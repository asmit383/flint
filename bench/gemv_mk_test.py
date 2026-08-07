#!/usr/bin/env python3
"""Measure verify_cost(M) = time(M)/time(1) for the M=K int4 GEMV, L2-cold over all Granite linears (weights
stream once, shared across the M verified tokens). This is the number that sets real spec-decode tok/s:
  net tok/s = 275 * accepted / verify_cost(M).
Correctness-checked. If verify_cost stays near 1 for small M, the M=K verify is cheap and the megakernel
extension is worth it.

    python bench/gemv_mk_test.py
"""
import os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
LAYER = [("q", 2560, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 2560),
         ("gate", 8192, 2560), ("up", 8192, 2560), ("down", 2560, 8192)]
NLAYER = 40
LM = ("lm_head", 100352, 2560)
BASE_TPS = 275.0                                              # measured single-pass megakernel

def bit(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def pack(W, G=128):
    OUT, IN = W.shape; Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bit((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous(), (q * s).view(OUT, IN).to(torch.bfloat16)

def gtime(fn, iters=50):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000                   # us

if __name__ == "__main__":
    dev = "cuda"
    m = load(name="flint_mk", sources=[os.path.join(HERE, "kernels/int4_gemv_mk.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    specs = [s for _ in range(NLAYER) for s in LAYER] + [LM]
    print(f"building {len(specs)} linears L2-cold ...")
    W = []
    for _, OUT, IN in specs:
        Wq, sc, _ = pack(torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02)
        W.append((Wq, sc, IN))

    # correctness on one shape
    Wq, sc, IN = W[len(specs) // 2]; OUT = Wq.shape[0]
    _, _, Wdq = pack(torch.zeros(OUT, IN, device=dev, dtype=torch.bfloat16))  # placeholder to get shapes
    Wq2, sc2, Wdq2 = pack(torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02)
    X = torch.randn(4, IN, device=dev, dtype=torch.bfloat16) * 0.5
    Y = m.gemv_mk(Wq2, sc2, X, 4).float()
    ref = (X.float() @ Wdq2.float().T)                        # [4, OUT]
    rel = (Y - ref).norm().item() / (ref.norm().item() + 1e-9)
    print(f"correctness M=4: rel-L2 {rel:.5f}  {'OK' if rel < 1e-3 else 'BAD'}\n")

    Xs = {IN: {M: torch.randn(M, IN, device=dev, dtype=torch.bfloat16) for M in (1, 2, 4, 8)} for _, _, IN in specs}
    t1 = None
    print(f"  {'M':>2}  {'time(us)':>9}  {'verify_cost':>11}  {'net tok/s @acc':>16}")
    for M in (1, 2, 4, 8):
        def run(M=M):
            for Wq, sc, IN in W: m.gemv_mk(Wq, sc, Xs[IN][M], M)
        t = gtime(run)
        if M == 1: t1 = t
        c = t / t1
        # net tok/s at our current chain acceptance for this K (rough: acc grows a little with K)
        acc = {1: 1.0, 2: 1.4, 4: 1.7, 8: 1.87}[M]
        net = BASE_TPS * acc / c
        print(f"  {M:>2}  {t:9.1f}  {c:11.2f}  {net:12.0f}  (acc {acc})")
    print("\nverify_cost(M) is the real number; net = 275 * accepted / verify_cost. Small M near 1 = cheap verify.")
