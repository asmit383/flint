#!/usr/bin/env python3
"""Does activation sparsity actually BUY speed? Build the column-major sparse int4 kernel, verify it
against a masked dense reference, and race it against the dense champion (int4_gemv_v) on down_proj at
a real ~50% sparse h. The whole flint thesis in one number: fewer bytes -> faster.

    python bench/sparse_test.py --peak-bw 2.0e12
"""
import argparse, os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
OUT, IN = 2560, 8192                 # down_proj

def bitcast_i32(packed):             # int64 (0..2^32-1) -> int32 bit pattern
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32).contiguous()

def quant_pack_col(W, G=128):
    """W[OUT,IN] -> column-major sparse layout. Wcol int32[IN, OUT/8] (input j packs 8 outputs),
    scale_col bf16[IN/G, OUT]. Group along IN (standard). Returns q,scale for the reference too."""
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G)
    scale = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / scale), -8, 7).view(OUT, IN)
    qu = (q + 8).to(torch.int64).t().contiguous().view(IN, OUT // 8, 8)   # [IN, OUT/8, 8] pack outputs
    shifts = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    Wcol = bitcast_i32((qu << shifts).sum(-1))                            # [IN, OUT/8]
    scale_col = scale.squeeze(-1).t().contiguous().to(torch.bfloat16)     # [IN/G, OUT]
    return Wcol, scale_col, q, scale.squeeze(-1)

def quant_pack_row(W, G=128):        # dense champion layout (int4_gemv_v)
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G)
    scale = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / scale), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    shifts = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    Wq = bitcast_i32((qu << shifts).sum(-1))
    return Wq, scale.squeeze(-1).to(torch.bfloat16)

def graph_time(fn, iters=500):
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
    return s.elapsed_time(e) / iters

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-bw", type=float, default=2.0e12)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--jsplit", type=int, nargs="+", default=[16, 32, 64, 128])
    a = ap.parse_args()
    dev = "cuda"
    m = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    W = torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02
    h = torch.randn(IN, device=dev, dtype=torch.bfloat16)
    thr = h.abs().float().quantile(a.sparsity).item()          # threshold -> ~sparsity fraction skipped
    frac_kept = (h.abs().float() > thr).float().mean().item()
    print(f"down_proj [{OUT}x{IN}]  sparsity target {int(a.sparsity*100)}%  (kept {100*frac_kept:.1f}%)")

    Wcol, scale_col, q, scale = quant_pack_col(W, G)
    # masked dense reference
    hm = torch.where(h.abs() > thr, h, torch.zeros_like(h)).float()
    Wdq = (q * scale.repeat_interleave(G, dim=1)).float()      # [OUT,IN] dequantized
    y_ref = (Wdq @ hm)

    y = m.int4_spmv(Wcol, scale_col, h, thr, OUT, a.jsplit[0]).float()
    rel = (y - y_ref).norm().item() / (y_ref.norm().item() + 1e-9)
    print(f"correctness rel-L2 {rel:.4f}  {'OK' if rel < 0.02 else 'BAD'}")

    # dense champion time on the same shape (reads ALL bytes)
    Wq, scales = quant_pack_row(W, G)
    dense_ms = graph_time(lambda: m.int4_gemv_v(Wq, scales, h, 8))
    dbytes = OUT * IN * 0.5 + (IN // G) * OUT * 2
    print(f"\ndense int4_gemv_v:  {dense_ms*1e3:6.2f} us  MBU {100*dbytes/(dense_ms*1e-3)/a.peak_bw:5.1f}%  (reads 100% of weights)")
    print(f"sparse int4_spmv (reads ~{100*frac_kept:.0f}% of weights):")
    for js in a.jsplit:
        ms = graph_time(lambda: m.int4_spmv(Wcol, scale_col, h, thr, OUT, js))
        eff = dbytes * frac_kept
        print(f"    jsplit={js:3d}  {ms*1e3:6.2f} us  speedup {dense_ms/ms:4.2f}x  "
              f"eff-MBU {100*eff/(ms*1e-3)/a.peak_bw:5.1f}%")
