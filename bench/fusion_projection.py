#!/usr/bin/env python3
"""Try everything: does FUSION lift the B=1 single-pass MBU? Fuse projections that share an input
(QKV -> one 2560->3584 GEMV, gate/up -> one 2560->16384 GEMV) so the tiny k/v projections (~5% MBU,
latency-bound) stop dragging the aggregate. Full Granite-3B decode step, L2-cold (distinct weights),
CUDA-graphed. tinygemm AND our kernel, unfused vs fused.

    python bench/fusion_projection.py --peak-bw 2.0e12
"""
import argparse, os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
UNFUSED = [("q", 2560, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 2560),
           ("gate", 8192, 2560), ("up", 8192, 2560), ("down", 2560, 8192)]
FUSED = [("qkv", 3584, 2560), ("o", 2560, 2560), ("gate_up", 16384, 2560), ("down", 2560, 8192)]
LM = ("lm_head", 100352, 2560)
NLAYER = 40
TINYGEMM_STEP_US = 4178.0

def bitcast_i32(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def pack_contig(W, G=128):
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    qu = (torch.clamp(torch.round(Wg / s), -8, 7) + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bitcast_i32((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous()

def graph_time(fn, iters=100):
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
    ap = argparse.ArgumentParser(); ap.add_argument("--peak-bw", type=float, default=2.0e12)
    a = ap.parse_args(); dev = "cuda"
    m = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
    cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)

    def build(layout):
        specs = [s for _ in range(NLAYER) for s in layout] + [LM]
        ours, tiny, xs = [], [], {}
        torch.manual_seed(0)
        for _, OUT, IN in specs:
            W = torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02
            Wq, sc = pack_contig(W); ours.append((Wq, sc, IN))
            lin = torch.nn.Linear(IN, OUT, bias=False).to(dev).to(torch.bfloat16); lin.weight.data = W
            quantize_(lin, cfg); tiny.append(lin)
            xs.setdefault(IN, torch.randn(1, IN, device=dev, dtype=torch.bfloat16))
        return specs, ours, tiny, xs

    tot = lambda specs: sum(O * I * 0.5 + (I // G) * O * 2 for _, O, I in specs)
    print("building unfused + fused (L2-cold) ...")
    results = {}
    for tag, layout in [("unfused", UNFUSED), ("fused", FUSED)]:
        specs, ours, tiny, xs = build(layout)
        no = 8 if False else 6
        def run_ours():
            for Wq, sc, IN in ours: m.int4_gemv_v(Wq, sc, xs[IN].view(IN), 8 if IN == 8192 else 6)
        def run_tiny():
            for lin, (_, _, IN) in zip(tiny, specs): lin(xs[IN])
        ot, tt = graph_time(run_ours), graph_time(run_tiny)
        tb = tot(specs)
        results[tag] = dict(ot=ot, tt=tt, tb=tb, nk=len(specs))
        print(f"\n[{tag}]  {len(specs)} kernels, {tb/1e9:.2f} GB")
        print(f"   tinygemm {tt:7.1f} us  MBU {100*tb/(tt*1e-6)/a.peak_bw:4.1f}%")
        print(f"   ours     {ot:7.1f} us  MBU {100*tb/(ot*1e-6)/a.peak_bw:4.1f}%")

    print("\n=== end-to-end projection (overhead from measured tinygemm step) ===")
    overhead = TINYGEMM_STEP_US - results["unfused"]["tt"]
    for tag in ["unfused", "fused"]:
        for who, key in [("tinygemm", "tt"), ("ours", "ot")]:
            step = results[tag][key] + overhead
            print(f"  {tag:8s} {who:9s} {1e6/step:6.1f} tok/s   x2 MTP -> {2e6/step:5.0f}")
