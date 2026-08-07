#!/usr/bin/env python3
"""Full-decode megakernel: correctness (NL=1 vs matched reference) then END-TO-END speed (NL=40, real
decode loop, tok/s vs the 237 baseline). Random weights (speed is value-independent; blocks already
correctness-validated). This is the honest end-to-end megakernel number.

    python bench/megakernel_decode_test.py --mode correct   # NL=1 assembly check
    python bench/megakernel_decode_test.py --mode speed --peak-bw 2.0e12   # NL=40 tok/s
"""
import os, time, argparse, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128; DIM = 2560; INTER = 8192; NH = 40; NKV = 8; HD = 64; VOCAB = 100352
SCALE = 0.015625; ROPE = 1e7; RESID = 0.22; LOGS = 10.0; MAXSEQ = 64

def bitcast_i32(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def qpc(W, G=128):
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bitcast_i32((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous(), (q * s).view(OUT, IN).to(torch.bfloat16)

def build_layers(NL, dev, want_dq=False):
    QKV = DIM + 2 * NKV * HD
    def stk(shape, n):
        return [torch.randn(*shape, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(n)]
    W = {"qkv": stk((QKV, DIM), NL), "o": stk((DIM, DIM), NL), "gu": stk((2 * INTER, DIM), NL), "d": stk((DIM, INTER), NL)}
    packed, dq = {}, {}
    for key in W:
        qs = [qpc(w) for w in W[key]]
        packed[key + "_q"] = torch.stack([a[0] for a in qs])
        packed[key + "_s"] = torch.stack([a[1] for a in qs])
        if want_dq: dq[key] = [a[2] for a in qs]
    n1 = torch.stack([torch.randn(DIM, device=dev, dtype=torch.bfloat16).abs() + 0.5 for _ in range(NL)])
    n2 = torch.stack([torch.randn(DIM, device=dev, dtype=torch.bfloat16).abs() + 0.5 for _ in range(NL)])
    return packed, n1, n2, dq

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="correct"); ap.add_argument("--peak-bw", type=float, default=2.0e12)
    ap.add_argument("--gen", type=int, default=32); a = ap.parse_args(); dev = "cuda"
    _hb = bool(os.environ.get("HANDROLL"))               # HANDROLL=1 -> hand-rolled (inline PTX) grid barrier
    _cf = ["-O3", "--use_fast_math"] + (["-DHANDROLL_BARRIER"] if _hb else [])
    m = load(name="flint_mkd" + ("_hb" if _hb else ""), sources=[os.path.join(HERE, "kernels/megakernel_decode.cu")],
             extra_cuda_cflags=_cf, verbose=False)
    torch.manual_seed(0)
    NL = 1 if a.mode == "correct" else 40
    P, n1, n2, dq = build_layers(NL, dev, want_dq=(a.mode == "correct"))
    nf = torch.randn(DIM, device=dev, dtype=torch.bfloat16).abs() + 0.5
    Wlm = torch.randn(VOCAB, DIM, device=dev, dtype=torch.bfloat16) * 0.02
    Wlm_q, Wlm_s, Wlm_dq = qpc(Wlm)
    kc = torch.zeros(NL, MAXSEQ, NKV, HD, device=dev, dtype=torch.bfloat16)
    vc = torch.zeros(NL, MAXSEQ, NKV, HD, device=dev, dtype=torch.bfloat16)

    def launch(h, pos):
        return m.decode_mega_launch(P["qkv_q"], P["qkv_s"], P["o_q"], P["o_s"], n1, P["gu_q"], P["gu_s"],
                                    P["d_q"], P["d_s"], n2, nf, Wlm_q, Wlm_s, h, kc, vc, pos, SCALE, ROPE, RESID, LOGS)

    if a.mode == "correct":
        h0 = torch.randn(DIM, device=dev, dtype=torch.bfloat16); pos = 0
        # reference (NL=1, pos=0: rope=identity, attention T=1 -> attn_out = v[kv])
        def rms(x, w): return (x.float() * torch.rsqrt(x.float().pow(2).mean() + 1e-5) * w.float())
        xn1 = rms(h0, n1[0]); qkv = dq["qkv"][0].float() @ xn1
        v = qkv[DIM + NKV * HD:].view(NKV, HD)
        ao = torch.stack([v[qh // (NH // NKV)] for qh in range(NH)]).view(DIM)
        h1 = h0.float() + (dq["o"][0].float() @ ao) * RESID
        xn2 = rms(h1.bfloat16(), n2[0]); gu = dq["gu"][0].float() @ xn2
        act = F.silu(gu[:INTER]) * gu[INTER:]
        h2 = h1 + (dq["d"][0].float() @ act) * RESID
        xnf = rms(h2.bfloat16(), nf); logits_ref = (Wlm_dq.float() @ xnf) / LOGS
        logits = launch(h0.clone(), pos).float(); torch.cuda.synchronize()
        rel = (logits - logits_ref).norm().item() / (logits_ref.norm().item() + 1e-9)
        cos = F.cosine_similarity(logits, logits_ref, dim=0).item()
        print(f"full-decode megakernel NL=1  rel-L2 {rel:.4f}  cos {cos:.5f}  {'OK' if rel < 0.05 else 'BAD'}")
        print(f"  argmax kernel={logits.argmax().item()}  ref={logits_ref.argmax().item()}")
    else:
        nparams = 40 * (( (DIM+2*NKV*HD)*DIM + DIM*DIM + 2*INTER*DIM + DIM*INTER)) + VOCAB*DIM
        bytes_tok = nparams * 0.5 + (nparams // 128) * 2
        h = torch.randn(DIM, device=dev, dtype=torch.bfloat16)
        for w in range(8):  # warmup
            lg = launch(h, w % MAXSEQ); h = torch.randn(DIM, device=dev, dtype=torch.bfloat16)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for it in range(a.gen):
            lg = launch(h, it % MAXSEQ)
            h = torch.randn(DIM, device=dev, dtype=torch.bfloat16)  # (random next-h; speed is value-indep)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        tps = a.gen / dt
        print(f"FULL-DECODE MEGAKERNEL end-to-end: {tps:.1f} tok/s   MBU {100*bytes_tok*tps/a.peak_bw:.1f}%   (baseline 237)")
