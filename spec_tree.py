#!/usr/bin/env python3
"""spec_tree.py — TREE speculative decode on the flint int4 megakernel. The draft rolls a width-B beam to
depth D; verify_mega scores the whole tree in ONE int4 launch with a TREE ATTENTION MASK (each node attends
to its ancestors + the cached prefix). Accept the longest root-to-leaf path the target's greedy agrees with,
then KV-surgery the accepted path to contiguous slots. Measures the REAL tree acceptance (vs the 4.28 ceiling)
and net tok/s. Draft is eager here (correctness first); optimize once acceptance justifies it.

    python spec_tree.py --selftest --draft /root/eagle_ll8k/draft_ms.pt --B 2 --D 6
"""
import os, sys, time, argparse, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "train"))
from draft_model import DraftHead, rmsnorm
from chat import load_packed, DIM, INTER, NH, NKV, HD, VOCAB, NL, SCALE, ROPE, RESID, LOGS, EMB_MULT, MODEL
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="/root/eagle_ll8k/draft_ms.pt")
    ap.add_argument("--B", type=int, default=2); ap.add_argument("--D", type=int, default=6); ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--maxseq", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    dev = "cuda"; torch.set_grad_enabled(False)

    print("compiling megakernel + verify ...", flush=True)
    dec = load(name="flint_dec", sources=[HERE + "/kernels/megakernel_decode.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
    ver = load(name="flint_ver", sources=[HERE + "/kernels/verify_mega.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
    W = load_packed(dev)
    tok = AutoTokenizer.from_pretrained(MODEL); eos = tok.eos_token_id
    kc = torch.zeros(NL, a.maxseq, NKV, HD, device=dev, dtype=torch.bfloat16)
    vc = torch.zeros(NL, a.maxseq, NKV, HD, device=dev, dtype=torch.bfloat16)
    embed = W["embed"]; nf = W["nf"]
    ck = torch.load(a.draft, map_location=dev)
    draft = DraftHead(ck["dim"], ck["nh"], ck["inter"], fuse=ck.get("fuse", 1)).to(dev).to(torch.bfloat16)
    draft.load_state_dict(ck["state"]); draft.eval()

    def tlog(fp):  return F.linear(rmsnorm(fp[..., -DIM:], nf), embed) / LOGS   # feature -> full logits

    def dec1(token, pos):
        h = (embed[token].float() * EMB_MULT).to(torch.bfloat16)
        return dec.decode_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
            W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, pos, SCALE, ROPE, RESID, LOGS)

    def verM(tokens, pos, depth, amask):
        M = len(tokens)
        h = (embed[torch.tensor(tokens, device=dev)].float() * EMB_MULT).to(torch.bfloat16)
        return ver.verify_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
            W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc,
            depth.int(), amask, pos, M, SCALE, ROPE, RESID, LOGS)

    def tree_draft(feats, base_pos, first_tok):
        # eager width-B beam to depth D. returns tree nodes: tok[], depth[], parent[] (parent = tree-node idx or -1=root)
        lo = max(0, feats.shape[0] - a.W); fctx = feats[lo:]; Wc = fctx.shape[0]
        pctx = torch.arange(base_pos - Wc, base_pos, device=dev)
        def fwd(hist_f, hist_t, last_tok):                 # full-window draft forward -> next feature
            seq_f = torch.cat([fctx] + ([torch.stack(hist_f)] if hist_f else []))       # [Wc+len, fdim]
            toks = torch.zeros(seq_f.shape[0], dtype=torch.long, device=dev)
            toks[Wc - 1] = first_tok
            for i, t in enumerate(hist_t): toks[Wc + i] = t
            pos = torch.arange(base_pos - Wc, base_pos - Wc + seq_f.shape[0], device=dev)
            ec = F.embedding(toks, embed).to(torch.bfloat16)
            fp = draft(seq_f.unsqueeze(0), ec.unsqueeze(0), pos)
            return fp[0, -1]
        tok_a, dep_a, par_a = [], [], []
        f0 = fwd([], [], first_tok); lg = tlog(f0); top = lg.topk(a.B)
        beams = []
        for b in range(a.B):
            t = int(top.indices[b]); tok_a.append(t); dep_a.append(1); par_a.append(-1)
            beams.append({"lp": float(top.values[b]), "node": len(tok_a) - 1, "hist_f": [f0], "hist_t": [t]})
        for d in range(2, a.D + 1):
            cand = []
            for bm in beams:
                fn = fwd(bm["hist_f"], bm["hist_t"], None); lg = tlog(fn); top = lg.topk(a.B)
                for b in range(a.B):
                    cand.append((bm["lp"] + float(top.values[b]), bm, int(top.indices[b]), fn))
            cand.sort(key=lambda x: -x[0]); cand = cand[:a.B]
            beams = []
            for lp, bm, t, fn in cand:
                tok_a.append(t); dep_a.append(d); par_a.append(bm["node"])
                beams.append({"lp": lp, "node": len(tok_a) - 1, "hist_f": bm["hist_f"] + [fn], "hist_t": bm["hist_t"] + [t]})
        return tok_a, dep_a, par_a

    def run(ids, cb):
        pos = 0
        for t in ids[:-1]: dec1(t, pos); pos += 1
        d0 = torch.arange(1, dtype=torch.int32, device=dev); a0 = torch.ones(1, 1, dtype=torch.uint8, device=dev)
        lg, feat = verM([ids[-1]], pos, d0, a0); feats = feat.clone(); pos += 1; nxt = int(lg[0].argmax())
        n = 0; passes = 0; t0 = time.perf_counter(); t_draft = 0.0; t_ver = 0.0
        while n < a.max_new and pos < a.maxseq - a.D - 2:
            cb(nxt); n += 1
            if nxt == eos: break
            torch.cuda.synchronize(); _td = time.perf_counter()
            tok_a, dep_a, par_a = tree_draft(feats, pos, nxt)
            torch.cuda.synchronize(); t_draft += time.perf_counter() - _td
            N = len(tok_a)
            cand = [nxt] + tok_a                            # index 0 = root
            depth = [0] + dep_a
            cpar = [-1] + [0 if p == -1 else 1 + p for p in par_a]   # parent in cand-index space (root's parent = none)
            amask = torch.zeros(N + 1, N + 1, dtype=torch.uint8, device=dev)
            for i in range(N + 1):                          # ancestor-or-self mask
                j = i
                while j != -1: amask[i, j] = 1; j = cpar[j]
            _tv = time.perf_counter()
            vlog, vfeat = verM(cand, pos, torch.tensor(depth, device=dev), amask)
            torch.cuda.synchronize(); t_ver += time.perf_counter() - _tv
            tgt = vlog.argmax(-1).tolist()                  # target greedy per cand node
            # walk the longest path the target agrees with, from root
            children = {i: [] for i in range(N + 1)}
            for k in range(N): children[cpar[k + 1]].append(k + 1)
            path = [0]; cur = 0
            while True:
                want = tgt[cur]; nxtnode = None
                for ch in children[cur]:
                    if cand[ch] == want: nxtnode = ch; break
                if nxtnode is None: break
                path.append(nxtnode); cur = nxtnode
            if os.environ.get("DBG") and passes < 3:
                print(f"\n[dbg pass {passes}] pos={pos} cand={cand} depth={depth} cpar={cpar}", file=sys.stderr)
                print(f"  tgt={tgt}  path={path}  accepted={[cand[p] for p in path]}", file=sys.stderr)
            L = len(path) - 1                               # accepted tree depth (tokens beyond root)
            for step in range(1, len(path)):
                cb(cand[path[step]]); n += 1
            # KV surgery: accepted node at cand slot pos+path[step] -> contiguous slot pos+step (RoPE already at pos+step)
            for step in range(1, len(path)):
                src = pos + path[step]; dst = pos + step
                if src != dst: kc[:, dst] = kc[:, src]; vc[:, dst] = vc[:, src]
            feats = torch.cat([feats, vfeat[path]])          # features along accepted path (root..leaf)
            pos += 1 + L; nxt = tgt[path[-1]]; passes += 1
        dt = time.perf_counter() - t0
        print(f"\n\033[2m[profile] draft {t_draft/passes*1000:.1f}ms | verify {t_ver/passes*1000:.1f}ms | passes {passes} | tree {N+1} nodes\033[0m")
        return n, dt, (n / passes if passes else 1)

    if a.selftest:
        text = tok.apply_chat_template([{"role": "user", "content": "Write a python function that returns the nth Fibonacci number, with a short docstring."}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        print(f"[selftest] prompt {len(ids)} tok, B={a.B} D={a.D}\n", flush=True)
        buf = {"ids": [], "prev": ""}
        def cb(t):
            buf["ids"].append(t); s = tok.decode(buf["ids"])
            sys.stdout.write(s[len(buf["prev"]):]); sys.stdout.flush(); buf["prev"] = s
        n, dt, tpp = run(ids, cb)
        if os.environ.get("DUMP"): print("\nEMITIDS", buf["ids"][:30], file=sys.stderr)
        print(f"\n\033[2m⚡ {n} tokens · {n/dt:.1f} tok/s · {tpp:.2f} tokens/pass (TREE spec-decode on megakernel)\033[0m")


if __name__ == "__main__":
    main()
