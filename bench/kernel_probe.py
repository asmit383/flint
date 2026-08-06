#!/usr/bin/env python3
"""Per-kernel meter: isolate ONE int4 weight-only GEMV at Granite-4.1 shapes, B=1, and report
achieved bandwidth + MBU. This is the harness every custom kernel gets A/B'd against — one matmul,
no compile/graph noise, so ncu can profile it cleanly:

    python bench/kernel_probe.py --peak-bw 2.0e12                 # timing (all Granite linears)
    ncu --set full -k _weight_int4pack_mm python bench/kernel_probe.py --shape down --iters 1

At B=1 the matmul is a GEMV: read the whole weight once, do ~nothing. So achieved GB/s ÷ peak = MBU,
and MBU is the number the kernel lives or dies by. tinygemm should land ~dequant-bound (low MBU, high
Math-Pipe-Throttle); our kernel's job is to push that toward bf16's ~65%.
"""
import argparse, torch

# Granite-4.1-3B linear shapes (out_features, in_features), B=1.  dim=2560, intermediate=8192,
# n_head=40*64=2560, n_local_heads=8*64=512.
SHAPES = {
    "gate": (8192, 2560),   # w1  — MLP up-projection (gate)
    "up":   (8192, 2560),   # w3  — MLP up-projection
    "down": (2560, 8192),   # w2  — MLP down-projection  (the big read)
    "q":    (2560, 2560),   # attn q
    "kv":   (512,  2560),   # attn k/v (GQA, small)
    "o":    (2560, 2560),   # attn out
}

def make_int4_linear(out_f, in_f, dev):
    lin = torch.nn.Linear(in_f, out_f, bias=False).to(dev).to(torch.bfloat16)
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
    except ImportError:
        from torchao.quantization import int4_weight_only
        cfg = int4_weight_only(group_size=128)
    quantize_(lin, cfg)
    return lin

def int4_bytes(out_f, in_f, group_size=128):
    # 4-bit weights + bf16 group scales (+ zeros, small). Weight dominates.
    return out_f * in_f * 0.5 + (in_f // group_size) * out_f * 2

def bench_one(name, out_f, in_f, peak, iters, dev):
    lin = make_int4_linear(out_f, in_f, dev)
    x = torch.randn(1, in_f, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        # CUDA-graph capture: eager launch overhead (~37us) dwarfs a B=1 GEMV and would mask the
        # kernel's real bandwidth. Graph replay is what the actual decode loop uses, so it's the
        # honest per-kernel time.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                y = lin(x)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = lin(x)
        for _ in range(30):                      # warmup replays
            g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    gbs = int4_bytes(out_f, in_f) / (ms * 1e-3) / 1e9
    mbu = 100 * gbs * 1e9 / peak
    print(f"  {name:5s} [{out_f:>5d}x{in_f:<5d}]  {ms*1e3:7.2f} us  {gbs:7.1f} GB/s  MBU {mbu:5.1f}%")
    return gbs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-bw", type=float, default=2.0e12)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--shape", choices=list(SHAPES) + ["all"], default="all")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"int4 GEMV probe (B=1, peak-bw={a.peak_bw/1e12:.2f} TB/s, iters={a.iters})")
    names = list(SHAPES) if a.shape == "all" else [a.shape]
    for n in names:
        bench_one(n, *SHAPES[n], a.peak_bw, a.iters, dev)
