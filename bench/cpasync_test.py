#!/usr/bin/env python3
"""Can cp.async deep-pipelining break the int4 B=1 MBU wall? Measure the cp.async GEMV L2-cold (all
Granite linears, distinct weights streaming from HBM), correctness-checked, sweeping pipeline depth,
vs the vec champion. If MBU jumps past ~28%, cp.async is the lever toward 60-70%.

    python bench/cpasync_test.py --peak-bw 2.0e12
"""
import os, argparse, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
LAYER = [("q", 2560, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 2560),
         ("gate", 8192, 2560), ("up", 8192, 2560), ("down", 2560, 8192)]
NLAYER = 40
LM = ("lm_head", 100352, 2560)

def bit(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def pack(W, G=128):
    OUT, IN = W.shape; Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    qu = (torch.clamp(torch.round(Wg / s), -8, 7) + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bit((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous()

def gtime(fn, iters=100):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--peak-bw", type=float, default=2.0e12); a = ap.parse_args()
    dev = "cuda"
    mv = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    mc = load(name="flint_cpa", sources=[os.path.join(HERE, "kernels/int4_gemv_cpasync.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    specs = [s for _ in range(NLAYER) for s in LAYER] + [LM]
    print(f"building {len(specs)} linears L2-cold ...")
    W = []
    for _, OUT, IN in specs:
        Wq, sc = pack(torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02)
        W.append((Wq, sc, IN))
    xs = {IN: torch.randn(1, IN, device=dev, dtype=torch.bfloat16) for _, _, IN in specs}
    tot = sum(O * I * 0.5 + (I // G) * O * 2 for _, O, I in specs)

    # correctness: cp.async vs vec on one shape
    Wq, sc, IN = W[len(specs) // 2]
    yv = mv.int4_gemv_v(Wq, sc, xs[IN].view(IN), 8).float()
    yc = mc.gemv_cpasync(Wq, sc, xs[IN].view(IN), 4).float()
    rel = (yc - yv).norm().item() / (yv.norm().item() + 1e-9)
    print(f"cp.async correctness vs vec: rel-L2 {rel:.5f}  {'OK' if rel < 1e-3 else 'BAD'}")

    def run_vec():
        for Wq, sc, IN in W: mv.int4_gemv_v(Wq, sc, xs[IN].view(IN), 8 if IN == 8192 else 6)
    tv = gtime(run_vec); print(f"\n  vec (champion)   {tv:7.1f} us   MBU {100*tot/(tv*1e-6)/a.peak_bw:4.1f}%")
    for ns in [2, 3, 4, 6, 8]:
        def run_cpa(ns=ns):
            for Wq, sc, IN in W: mc.gemv_cpasync(Wq, sc, xs[IN].view(IN), ns)
        tc = gtime(run_cpa); print(f"  cp.async ns={ns}  {tc:7.1f} us   MBU {100*tot/(tc*1e-6)/a.peak_bw:4.1f}%")
