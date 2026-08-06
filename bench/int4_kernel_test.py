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
    """W [OUT,IN] -> interleaved int4 layout matching int4_gemv.cu.
      Wq     int32[OUT, IN/8]   word (blk,lane) packs K-indices {blk*256+lane+32m : m=0..7}
      scales bf16 [OUT, IN/128] per-(blk,half) group scale; group = 2*blk + (m>=4)
    Symmetric int4: q in [-8,7], stored offset (q+8); dequant = (qu-8)*scale. Returns Wdq for the ref."""
    OUT, IN = W.shape
    assert IN % 256 == 0 and G == 128
    nblk = IN // 256
    Wg = W.float().view(OUT, nblk, 2, 4, 32)                     # [o, blk, half, m_in_half, lane]
    scale = Wg.abs().amax(dim=(3, 4), keepdim=True) / 7.0        # per (o,blk,half) -> group of 128
    q = torch.clamp(torch.round(Wg / scale), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, nblk, 8, 32)          # m = half*4 + m_in_half
    shifts = (torch.arange(8, device=W.device) * 4).view(1, 1, 8, 1)
    packed = (qu << shifts).sum(dim=2).view(OUT, nblk * 32)      # [o, blk*32+lane], 0..2^32-1
    packed_i32 = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32).contiguous()
    scales = scale.view(OUT, nblk * 2).to(torch.bfloat16).contiguous()   # [OUT, IN/128], idx 2*blk+half
    Wdq = (q * scale).view(OUT, IN).to(torch.bfloat16)          # ref, in natural K order
    return packed_i32, scales, Wdq

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
    ap.add_argument("--only", default=None, help="run a single shape (e.g. down)")
    a = ap.parse_args()
    if a.only: SHAPES = {a.only: SHAPES[a.only]}
    dev = "cuda"
    print("building kernels/int4_gemv.cu ...")
    m = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

    for name, (OUT, IN) in SHAPES.items():
        torch.manual_seed(0)
        W = torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, IN, device=dev, dtype=torch.bfloat16)
        Wq, scales, Wdq = quant_pack(W, G)
        # bf16 reference from the SAME quantized weights (isolates kernel correctness, not quant error)
        y_ref = (Wdq.float() @ x.float().view(IN)).view(OUT)

        y = m.int4_gemv(Wq, scales, x.view(IN), G, a.nacc[0]).float()
        rel_l2 = (y - y_ref).norm().item() / (y_ref.norm().item() + 1e-9)   # robust to near-zero entries
        cos = torch.nn.functional.cosine_similarity(y, y_ref, dim=0).item()
        ok = "OK " if rel_l2 < 0.02 else "BAD"
        print(f"\n{name} [{OUT}x{IN}]  rel-L2 {rel_l2:.4f}  cos {cos:.5f} {ok}")
        if rel_l2 >= 0.02:
            print(f"    y[:4]={y[:4].tolist()}  ref[:4]={y_ref[:4].tolist()}")

        best = None
        for nacc in a.nacc:
            ms = graph_time(lambda: m.int4_gemv(Wq, scales, x.view(IN), G, nacc), a.iters)
            gbs = int4_bytes(OUT, IN) / (ms * 1e-3) / 1e9
            mbu = 100 * gbs * 1e9 / a.peak_bw
            if best is None or mbu > best[1]: best = (f"nacc={nacc}", mbu)
            print(f"    nacc={nacc}       {ms*1e3:7.2f} us  {gbs:7.1f} GB/s  MBU {mbu:5.1f}%")
        # no-shared variant: x from global/L2, frees shared -> higher occupancy
        for nacc in a.nacc:
            yg = m.int4_gemv_g(Wq, scales, x.view(IN), nacc).float()
            rel = (yg - y_ref).norm().item() / (y_ref.norm().item() + 1e-9)
            ms = graph_time(lambda: m.int4_gemv_g(Wq, scales, x.view(IN), nacc), a.iters)
            gbs = int4_bytes(OUT, IN) / (ms * 1e-3) / 1e9
            mbu = 100 * gbs * 1e9 / a.peak_bw
            if rel < 0.02 and (best is None or mbu > best[1]): best = (f"glob nacc={nacc}", mbu)
            print(f"    glob nacc={nacc}   {ms*1e3:7.2f} us  {gbs:7.1f} GB/s  MBU {mbu:5.1f}%  (rel {rel:.4f})")
        # split-K sweep (helps tall-K/few-row shapes that underfill: down)
        nblk = IN // 256
        for sk in [s for s in (2, 4, 8, 16) if nblk % s == 0]:
            ysk = m.int4_gemv_sk(Wq, scales, x.view(IN), sk, 4).float()
            rel = (ysk - y_ref).norm().item() / (y_ref.norm().item() + 1e-9)
            ms = graph_time(lambda: m.int4_gemv_sk(Wq, scales, x.view(IN), sk, 4), a.iters)
            gbs = int4_bytes(OUT, IN) / (ms * 1e-3) / 1e9
            mbu = 100 * gbs * 1e9 / a.peak_bw
            if rel < 0.02 and (best is None or mbu > best[1]): best = (f"splitk={sk}", mbu)
            print(f"    splitk={sk} nacc4  {ms*1e3:7.2f} us  {gbs:7.1f} GB/s  MBU {mbu:5.1f}%  (rel {rel:.4f})")
        print(f"    best: {best[0]}  MBU {best[1]:.1f}%   (tinygemm ref: gate/up 37%, down 45%, q/o 19%)")
