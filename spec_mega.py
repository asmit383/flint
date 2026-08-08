#!/usr/bin/env python3
"""spec_mega.py — speculative decode running ON the flint int4 megakernel (not a bf16 stand-in). The EAGLE
draft proposes K tokens; verify_mega scores K+1 in ONE persistent cooperative int4 launch (every weight read
once) and returns both logits and the post-final-norm feature the draft rolls on next. Net tok/s is the real
end-to-end number: accepted / (draft_roll + verify).

    python spec_mega.py --selftest --draft /root/eagle_ll8k/draft_ms.pt --K 3
"""
import os, sys, time, argparse, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "train"))
from draft_model import DraftHead, rmsnorm
from chat import load_packed, DIM, INTER, NH, NKV, HD, VOCAB, NL, SCALE, ROPE, RESID, LOGS, EMB_MULT, MODEL
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="/root/eagle_ll8k/draft_ms.pt")
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")     # bf16 __hfma2 verify accumulation (~22% faster)
    ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--maxseq", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    dev = "cuda"; torch.set_grad_enabled(False)

    print("compiling megakernel + verify ...", flush=True)
    dec = load(name="flint_dec", sources=[HERE + "/kernels/megakernel_decode.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
    _vf = ["-O3", "--use_fast_math"] + (["-DBF16ACC"] if a.bf16 else [])
    ver = load(name=("flint_ver_bf16" if a.bf16 else "flint_ver"), sources=[HERE + "/kernels/verify_mega.cu"], extra_cuda_cflags=_vf)
    print("loading + int4-packing Granite ...", flush=True)
    W = load_packed(dev)
    tok = AutoTokenizer.from_pretrained(MODEL); eos = tok.eos_token_id
    kc = torch.zeros(NL, a.maxseq, NKV, HD, device=dev, dtype=torch.bfloat16)
    vc = torch.zeros(NL, a.maxseq, NKV, HD, device=dev, dtype=torch.bfloat16)
    embed = W["embed"]; nf = W["nf"]                       # tied lm_head + final norm (draft's token head)

    ck = torch.load(a.draft, map_location=dev)
    draft = DraftHead(ck["dim"], ck["nh"], ck["inter"], fuse=ck.get("fuse", 1)).to(dev).to(torch.bfloat16)
    draft.load_state_dict(ck["state"]); draft.eval(); draft.setup_cache(a.W + a.K + 2, dev)

    def head_tok(fp):                                      # feature -> token (target norm + tied lm_head)
        return (F.linear(rmsnorm(fp[..., -DIM:], nf), embed) / LOGS).argmax(-1)

    def dec1(token, pos):
        h = (embed[token].float() * EMB_MULT).to(torch.bfloat16)
        return dec.decode_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
            W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, pos, SCALE, ROPE, RESID, LOGS)

    def verM(tokens, pos):                                 # -> (logits[M,VOCAB], feature[M,DIM])
        M = len(tokens)
        h = (embed[torch.tensor(tokens, device=dev)].float() * EMB_MULT).to(torch.bfloat16)
        depth = torch.arange(M, dtype=torch.int32, device=dev)                 # chain: depth = index
        amask = torch.tril(torch.ones(M, M, dtype=torch.uint8, device=dev))    # chain: causal
        return ver.verify_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
            W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, depth, amask, pos, M, SCALE, ROPE, RESID, LOGS)

    # manual CUDA graph for the rollout step — robust to the interleaved cooperative megakernel launches that
    # poison torch.compile's cudagraph pool (which made the step re-capture every pass = 73ms). Static buffers.
    fdim = draft.fdim
    _sf = torch.zeros(1, 1, fdim, device=dev, dtype=torch.bfloat16)
    _se = torch.zeros(1, 1, DIM, device=dev, dtype=torch.bfloat16)
    _sp = torch.zeros(1, dtype=torch.long, device=dev); _ss = torch.zeros(1, dtype=torch.long, device=dev)
    def _step_raw():
        fp = draft(_sf, _se, _sp, input_pos=_ss); fn = fp[:, -1]
        return fn, head_tok(fn)[0]
    for _ in range(3): _step_raw()
    torch.cuda.synchronize()
    _g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(_g): _of, _ot = _step_raw()

    def draft_roll(feats, positions, first_tok, K):        # fixed-cache rollout, graphed step
        lo = max(0, feats.shape[0] - a.W)
        fctx = feats[lo:]; pctx = positions[lo:]; Wc = fctx.shape[0]
        tctx = torch.zeros(Wc, dtype=torch.long, device=dev); tctx[-1] = first_tok
        emb_ctx = F.embedding(tctx, embed).to(torch.bfloat16)
        fp = draft(fctx.unsqueeze(0), emb_ctx.unsqueeze(0), pctx, input_pos=torch.arange(Wc, device=dev))  # prime (eager)
        f_new = fp[:, -1]; t_new = head_tok(f_new)[0]
        out = [t_new]; cpos = int(pctx[-1]) + 1; slot = Wc
        for _ in range(K - 1):
            _sf.copy_(f_new.view(1, 1, -1))
            _se.copy_(F.embedding(t_new.view(1), embed).to(torch.bfloat16).view(1, 1, -1))
            _sp.fill_(cpos); _ss.fill_(slot)
            _g.replay()
            f_new = _of.clone(); t_new = _ot.clone()
            out.append(t_new); cpos += 1; slot += 1
        return torch.stack(out)

    def warm_prime():                                      # pay per-Wc cuBLAS/SDPA autotune ONCE, outside timing
        for wc in range(1, a.W + 1):
            df = torch.zeros(1, wc, fdim, device=dev, dtype=torch.bfloat16)
            de = torch.zeros(1, wc, DIM, device=dev, dtype=torch.bfloat16)
            draft(df, de, torch.arange(wc, device=dev), input_pos=torch.arange(wc, device=dev))
        torch.cuda.synchronize()

    def run(ids, cb):
        pos = 0
        for t in ids[:-1]: dec1(t, pos); pos += 1          # prefill bulk (M=1 megakernel)
        lg, feat = verM([ids[-1]], pos)                    # last prompt tok -> logits + feature
        feats = feat.clone(); pos += 1; nxt = lg[0].argmax().view(1)
        warm_prime()                                       # warm all Wc prime shapes BEFORE timing
        draft_roll(feats, torch.arange(pos - feats.shape[0], pos, device=dev), nxt, a.K)   # warmup (positions aligned to feats)
        torch.cuda.synchronize()
        n = 0; passes = 0; t0 = time.perf_counter(); t_draft = 0.0; t_ver = 0.0
        while n < a.max_new and pos < a.maxseq - a.K - 2:
            cb(nxt.item()); n += 1
            if nxt.item() == eos: break
            torch.cuda.synchronize(); _td = time.perf_counter()
            d = draft_roll(feats, torch.arange(pos - feats.shape[0], pos, device=dev), nxt, a.K)
            torch.cuda.synchronize(); t_draft += time.perf_counter() - _td; _tv = time.perf_counter()
            cand = torch.cat([nxt, d]).tolist()            # M = K+1
            vlog, vfeat = verM(cand, pos)
            torch.cuda.synchronize(); t_ver += time.perf_counter() - _tv
            tgt = vlog.argmax(-1)                           # [M] target greedy per position
            acc = 0
            for j in range(a.K):
                if int(tgt[j]) == int(d[j]): acc += 1
                else: break
            for j in range(acc): cb(int(d[j])); n += 1
            feats = torch.cat([feats, vfeat[:acc + 1]])
            pos += 1 + acc; nxt = tgt[acc].view(1); passes += 1
        dt = time.perf_counter() - t0
        print(f"\n\033[2m[profile] draft {t_draft/passes*1000:.1f}ms/pass | verify {t_ver/passes*1000:.1f}ms/pass | passes {passes}\033[0m")
        return n, dt, (n / passes if passes else 1)

    if a.selftest:
        text = tok.apply_chat_template([{"role": "user", "content": "Write a python function that returns the nth Fibonacci number, with a short docstring."}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        print(f"[selftest] prompt {len(ids)} tok, K={a.K}\n", flush=True)
        buf = {"ids": [], "prev": ""}
        def cb(t):
            buf["ids"].append(t); s = tok.decode(buf["ids"])
            sys.stdout.write(s[len(buf["prev"]):]); sys.stdout.flush(); buf["prev"] = s
        n, dt, tpp = run(ids, cb)
        print(f"\n\033[2m⚡ {n} tokens · {n/dt:.1f} tok/s · {tpp:.2f} tokens/pass (spec-decode on megakernel)\033[0m")
        return


if __name__ == "__main__":
    main()
