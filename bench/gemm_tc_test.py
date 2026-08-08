#!/usr/bin/env python3
"""Tensor-core M=K int4 GEMM verify cost vs the scalar M=K GEMV. verify_cost(M)=time(M)/time(1), L2-cold over
all Granite linears. If tensor cores drop verify_cost enough, net = 275 * accepted/verify_cost clears 350/500.

    python bench/gemm_tc_test.py
"""
import os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128; BASE_TPS = 275.0
LAYER = [("q", 2560, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 2560),
         ("gate", 8192, 2560), ("up", 8192, 2560), ("down", 2560, 8192)]
NLAYER = 40; LM = ("lm_head", 100352, 2560)

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
    return a.elapsed_time(b) / iters * 1000

if __name__ == "__main__":
    dev = "cuda"
    tc = load(name="flint_tc", sources=[os.path.join(HERE, "kernels/int4_gemm_tc.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    sk = load(name="flint_mk", sources=[os.path.join(HERE, "kernels/int4_gemv_mk.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    specs = [s for _ in range(NLAYER) for s in LAYER] + [LM]
    print(f"building {len(specs)} linears L2-cold ...")
    W = []
    for _, OUT, IN in specs:
        Wq, sc, _ = pack(torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02)
        W.append((Wq, sc, IN))

    Wq2, sc2, Wdq2 = pack(torch.randn(2560, 2560, device=dev, dtype=torch.bfloat16) * 0.02)
    X = torch.randn(8, 2560, device=dev, dtype=torch.bfloat16) * 0.5
    Ytc = tc.gemm_tc(Wq2, sc2, X, 8).float(); ref = (X.float() @ Wdq2.float().T)
    rel = (Ytc - ref).norm().item() / (ref.norm().item() + 1e-9)
    print(f"tensor-core correctness M=8: rel-L2 {rel:.4f}  {'OK' if rel < 0.02 else 'BAD'}\n")

    Xs = {IN: {M: torch.randn(M, IN, device=dev, dtype=torch.bfloat16) for M in (1, 2, 4, 8, 16)} for _, _, IN in specs}
    acc = {1: 1.0, 2: 1.4, 4: 1.7, 8: 1.87, 16: 2.1}
    for name, mod, fn, Ms in (("scalar GEMV", sk, "gemv_mk", (1, 2, 4, 8)),
                              ("tensor-core", tc, "gemm_tc", (1, 2, 4, 8, 16))):
        print(f"  {name}:")
        t1 = None
        for M in Ms:
            def run(M=M, mod=mod, fn=fn):
                f = getattr(mod, fn)
                for Wq, sc, IN in W: f(Wq, sc, Xs[IN][M], M)
            t = gtime(run)
            if M == 1: t1 = t
            c = t / t1; net = BASE_TPS * acc[M] / c
            print(f"      M={M:>2}  {t:8.1f}us  verify_cost {c:5.2f}  net {net:5.0f} tok/s (acc {acc[M]})")
        print()
    print("net = 275 * accepted / verify_cost. If tensor-core verify_cost stays low, spec-decode clears 350/500.")
