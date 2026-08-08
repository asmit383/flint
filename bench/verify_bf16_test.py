#!/usr/bin/env python3
"""Is an OPTIMIZED GEMM (bf16 cuBLAS) a viable M=K verify? Measures bf16 torch.matmul verify time vs the
int4 scalar single-pass, L2-cold, and computes the REAL net tok/s = 275 * accepted * (single_pass /
verify_time). bf16 reads 4x the weight bytes but cuBLAS is near-peak and weights are read once across M.
Tells us if the tensor-core/GEMM verify direction can clear 350 WITHOUT hand-writing Marlin.

    python bench/verify_bf16_test.py
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
    return bit((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous()

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
    sk = load(name="flint_mk", sources=[os.path.join(HERE, "kernels/int4_gemv_mk.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    specs = [s for _ in range(NLAYER) for s in LAYER] + [LM]
    print(f"building {len(specs)} linears (int4 + bf16) L2-cold ...")
    Wint4, Wbf16 = [], []
    for _, OUT, IN in specs:
        w = torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02
        Wq, sc = pack(w); Wint4.append((Wq, sc, IN)); Wbf16.append(w.t().contiguous())   # [IN, OUT] for matmul

    Xs = {IN: {M: torch.randn(M, IN, device=dev, dtype=torch.bfloat16) for M in (1, 4, 8, 16)} for _, _, IN in specs}
    acc = {1: 1.0, 4: 1.7, 8: 1.87, 16: 2.1}

    # int4 scalar single-pass = the fast baseline (~ the megakernel single pass, 275 tok/s)
    def int4_1():
        for Wq, sc, IN in Wint4: sk.gemv_mk(Wq, sc, Xs[IN][1], 1)
    base = gtime(int4_1)
    print(f"\n  int4 scalar single-pass (M=1): {base:8.1f}us  == 275 tok/s reference\n")

    print(f"  bf16 cuBLAS verify:")
    for M in (1, 4, 8, 16):
        def run(M=M):
            for i, (_, _, IN) in enumerate(specs): torch.matmul(Xs[IN][M], Wbf16[i])
        t = gtime(run)
        net = BASE_TPS * acc[M] * base / t                  # accepted per verify, verify costs t/base single-passes
        print(f"      M={M:>2}  {t:8.1f}us  ({t/base:4.2f}x single-pass)  net {net:5.0f} tok/s (acc {acc[M]})")
    print("\n  net = 275 * accepted * (single_pass_time / verify_time). >350 = bf16 verify is viable.")
