#!/usr/bin/env python3
"""End-to-end single-pass projection: our int4 kernel vs tinygemm across a FULL Granite-3B decode step.
Builds every linear in one decode step (q/k/v/o/gate/up/down x 40 layers + LM head), 40 DISTINCT weight
sets (so weights stream from HBM, no fake L2 reuse), and CUDA-graph-times the whole sequence with each
kernel. Then projects end-to-end tok/s: the full tinygemm decode step is measured 4178us (=239 tok/s),
so non-linear overhead = 4178 - tinygemm_linear_total; our step = our_linear_total + that overhead.

    python bench/decode_projection.py --peak-bw 2.0e12
"""
import argparse, os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
# Granite-3B per-layer linear shapes (out, in), + LM head once.
LAYER = [("q", 2560, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 2560),
         ("gate", 8192, 2560), ("up", 8192, 2560), ("down", 2560, 8192)]
NLAYER = 40
LM_HEAD = ("lm_head", 100352, 2560)
TINYGEMM_STEP_US = 4178.0        # measured full int4 decode step (gpt-fast, mtp_scaling M=1) = 239 tok/s

def bitcast_i32(p):
    return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()

def pack_contig(W, G=128):        # our int4_gemv_v layout
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G); scale = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / scale), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bitcast_i32((qu << sh).sum(-1)), scale.squeeze(-1).to(torch.bfloat16).contiguous()

def graph_time(fn, iters=100):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000        # us

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--peak-bw", type=float, default=2.0e12)
    a = ap.parse_args(); dev = "cuda"
    m = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    from torchao.quantization import quantize_
    from torchao.quantization import Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
    tg_cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)

    # build all linears for a decode step: 40 distinct layers + LM head
    specs = [s for _ in range(NLAYER) for s in LAYER] + [LM_HEAD]
    print(f"building {len(specs)} linears (40 layers x 7 + LM head) ...")
    ours, tiny, xs = [], [], {}
    torch.manual_seed(0)
    for name, OUT, IN in specs:
        W = (torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02)
        Wq, sc = pack_contig(W)
        ours.append((Wq, sc, IN))
        lin = torch.nn.Linear(IN, OUT, bias=False).to(dev).to(torch.bfloat16); lin.weight.data = W
        quantize_(lin, tg_cfg); tiny.append(lin)
        if IN not in xs: xs[IN] = torch.randn(1, IN, device=dev, dtype=torch.bfloat16)

    def run_ours():
        for Wq, sc, IN in ours: m.int4_gemv_v(Wq, sc, xs[IN].view(IN), 8 if IN == 8192 else 6)
    def run_tiny():
        for lin, (_, _, IN) in zip(tiny, ours): lin(xs[IN])

    ot = graph_time(run_ours); tt = graph_time(run_tiny)
    tot_bytes = sum(OUT * IN * 0.5 + (IN // G) * OUT * 2 for _, OUT, IN in specs)
    overhead = TINYGEMM_STEP_US - tt
    print(f"\n  total weight bytes/step: {tot_bytes/1e9:.2f} GB")
    print(f"  linear-only time   ours {ot:7.1f} us   tinygemm {tt:7.1f} us   ({tt/ot:.2f}x faster)")
    print(f"  linear MBU         ours {100*tot_bytes/(ot*1e-6)/a.peak_bw:4.1f}%   tinygemm {100*tot_bytes/(tt*1e-6)/a.peak_bw:4.1f}%")
    print(f"\n  non-linear overhead (attn+norm+sample, from measured tinygemm step): {overhead:.0f} us")
    our_step = ot + overhead
    print(f"  projected decode step   ours {our_step:7.1f} us   tinygemm {TINYGEMM_STEP_US:.0f} us")
    print(f"  projected END-TO-END    ours {1e6/our_step:6.1f} tok/s   tinygemm {1e6/TINYGEMM_STEP_US:.0f} tok/s   "
          f"({(1e6/our_step)/(1e6/TINYGEMM_STEP_US):.2f}x)")
