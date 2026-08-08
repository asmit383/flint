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

    # WHOLE-rollout CUDA graph: prime(fixed W) + K-1 steps captured as ONE graph. Fixed shapes throughout
    # (RoPE is relative, so relative positions 0..W-1 are exact), so one capture, minimal launch overhead.
    fdim = draft.fdim; Wg = a.W
    _gfeat = torch.zeros(1, Wg, fdim, device=dev, dtype=torch.bfloat16)   # padded context features
    _gemb = torch.zeros(1, Wg, DIM, device=dev, dtype=torch.bfloat16)     # context token embeds (last = first_tok)
    _gpos = torch.arange(Wg, device=dev)
    _sps = [torch.tensor([Wg + i], device=dev) for i in range(a.K - 1)]   # fixed step positions/slots
    def _roll_raw():
        fp = draft(_gfeat, _gemb, _gpos, input_pos=_gpos); f_new = fp[:, -1]; t_new = head_tok(f_new)[0]
        outs = [t_new]
        for i in range(a.K - 1):
            emb1 = F.embedding(t_new.view(1), embed).to(torch.bfloat16).view(1, 1, -1)
            fp = draft(f_new.view(1, 1, -1), emb1, _sps[i], input_pos=_sps[i]); f_new = fp[:, -1]; t_new = head_tok(f_new)[0]
            outs.append(t_new)
        return torch.stack(outs)
    for _ in range(3): _roll_raw()
    torch.cuda.synchronize()
    _rg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(_rg): _gout = _roll_raw()

    def draft_roll(feats, first_tok):                      # feats [N, fdim] -> [K] proposed tokens (graphed)
        N = feats.shape[0]
        if N >= Wg:
            _gfeat.copy_(feats[N - Wg:].unsqueeze(0))
        else:                                              # pad front with the oldest feature (relative RoPE)
            _gfeat[0, Wg - N:].copy_(feats); _gfeat[0, :Wg - N].copy_(feats[0].unsqueeze(0).expand(Wg - N, -1))
        _gemb.zero_(); _gemb[0, -1].copy_(embed[first_tok.view(-1)[0]].to(torch.bfloat16))
        _rg.replay()
        return _gout.clone()

    def run(ids, cb):
        pos = 0
        for t in ids[:-1]: dec1(t, pos); pos += 1          # prefill bulk (M=1 megakernel)
        lg, feat = verM([ids[-1]], pos)                    # last prompt tok -> logits + feature
        feats = feat.clone(); pos += 1; nxt = lg[0].argmax().view(1)
        draft_roll(feats, nxt)                             # warmup the graph (already captured above)
        torch.cuda.synchronize()
        n = 0; passes = 0; t0 = time.perf_counter(); t_draft = 0.0; t_ver = 0.0
        while n < a.max_new and pos < a.maxseq - a.K - 2:
            cb(nxt.item()); n += 1
            if nxt.item() == eos: break
            torch.cuda.synchronize(); _td = time.perf_counter()
            d = draft_roll(feats, nxt)
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
