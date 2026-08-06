#!/usr/bin/env python3
"""Build flint's int4 GEMV, verify correctness vs a bf16 reference, then graph-timed MBU sweep over
accumulator counts — head-to-head with tinygemm's ~37-45% at the same Granite shapes.

    python bench/int4_kernel_test.py --peak-bw 2.0e12
"""
import argparse, os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
SHAPES = {"gate": (8192, 2560), "up": (8192, 2560), "down": (2560, 8192),
          "q": (2560, 2560), "o": (2560, 2560)}   # skip kv (tiny, latency-bound)

def quant_pack(W, G=128):
    """W [OUT,IN] -> (Wq int32[OUT,IN/8] bit-packed uint32, scales bf16[OUT,IN/G]). Symmetric int4:
    q in [-8,7], stored offset (q+8) in [0,15]; dequant = (qu-8)*scale."""
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G)
    scale = Wg.abs().amax(-1, keepdim=True) / 7.0                 # levels -7..7 (leave -8 headroom)
    q = torch.clamp(torch.round(Wg / scale), -8, 7)              # [-8,7]
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)           # [0,15]
    shifts = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    packed = (qu << shifts).sum(-1)                              # 0 .. 2^32-1, int64
    packed_i32 = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return packed_i32.contiguous(), scale.squeeze(-1).to(torch.bfloat16).contiguous(), q, scale

def graph_time(fn, iters):
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(30): g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters                             # ms

def int4_bytes(OUT, IN, G=128):
    return OUT * IN * 0.5 + (IN // G) * OUT * 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-bw", type=float, default=2.0e12)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--nacc", type=int, nargs="+", default=[2, 4, 6, 8])
    a = ap.parse_args()
    dev = "cuda"
    print("building kernels/int4_gemv.cu ...")
    m = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

    for name, (OUT, IN) in SHAPES.items():
        torch.manual_seed(0)
        W = torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, IN, device=dev, dtype=torch.bfloat16)
        Wq, scales, q, scale = quant_pack(W, G)
        # bf16 reference from the SAME quantized weights (isolates kernel correctness, not quant error)
        Wdq = (q * scale).view(OUT, IN).to(torch.bfloat16)
        y_ref = (Wdq.float() @ x.float().view(IN)).view(OUT)

        y = m.int4_gemv(Wq, scales, x.view(IN), G, a.nacc[0])
        rel = ((y.float() - y_ref).abs() / (y_ref.abs() + 1e-3)).max().item()
        ok = "OK " if rel < 0.02 else "BAD"
        print(f"\n{name} [{OUT}x{IN}]  correctness rel-err {rel:.4f} {ok}")

        best = None
        for nacc in a.nacc:
            ms = graph_time(lambda: m.int4_gemv(Wq, scales, x.view(IN), G, nacc), a.iters)
            gbs = int4_bytes(OUT, IN) / (ms * 1e-3) / 1e9
            mbu = 100 * gbs * 1e9 / a.peak_bw
            tag = ""
            if best is None or mbu > best[1]: best = (nacc, mbu)
            print(f"    nacc={nacc}  {ms*1e3:7.2f} us  {gbs:7.1f} GB/s  MBU {mbu:5.1f}%")
        print(f"    best: nacc={best[0]}  MBU {best[1]:.1f}%   (tinygemm ref: gate/up 37%, down 45%, q/o 19%)")
