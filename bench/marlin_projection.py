#!/usr/bin/env python3
"""The decisive measurement: Marlin int4 vs tinygemm across a FULL Granite-3B decode step, B=1, L2-cold
(281 distinct weights streaming from HBM). Marlin's cp.async pipeline is designed to stay memory-bound at
low batch — if it hits ~55-65% MBU here (vs tinygemm's 28%), single-pass ~600-700 tok/s is real and the
2x-MTP->1500 path is on.

    python bench/marlin_projection.py --peak-bw 2.0e12
"""
import argparse, torch, marlin

G = 128
LAYER = [("q", 2560, 2560), ("k", 512, 2560), ("v", 512, 2560), ("o", 2560, 2560),
         ("gate", 8192, 2560), ("up", 8192, 2560), ("down", 2560, 8192)]   # (out, in)
NLAYER = 40
LM = ("lm_head", 100352, 2560)
TINYGEMM_STEP_US = 4178.0        # measured full int4 decode step (gpt-fast) = 239 tok/s

def graph_time(fn, iters=100):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        for _ in range(10): g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters): g.replay()
        b.record(); torch.cuda.synchronize()
        t = a.elapsed_time(b) / iters * 1000
        if t < 5.0: raise RuntimeError("graph captured empty (custom-op stream issue) -> plain timing")
        return t
    except Exception as e:
        print(f"  [graph fallback: {e}]")
        for _ in range(5): fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters): fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / iters * 1000

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--peak-bw", type=float, default=2.0e12)
    a = ap.parse_args(); dev = "cuda"
    specs = [s for _ in range(NLAYER) for s in LAYER] + [LM]
    print(f"building {len(specs)} Marlin + tinygemm linears (L2-cold, distinct weights) ...")

    # Marlin: Layer allocates correctly-sized int4 B/s/workspace; call marlin.mul on preallocated A,C.
    mar = []
    for _, OUT, IN in specs:
        lyr = marlin.Layer(IN, OUT, groupsize=128).cuda().half()
        A = torch.randn(1, IN, dtype=torch.half, device=dev)
        C = torch.zeros(1, OUT, dtype=torch.half, device=dev)
        mar.append((A, lyr.B, C, lyr.s, lyr.workspace))
    def run_marlin():
        for A, B, C, s, ws in mar: marlin.mul(A, B, C, s, ws)

    # tinygemm (torchao TILE_PACKED = aten _weight_int4pack_mm), same shapes
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
    cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
    tiny, xs = [], {}
    torch.manual_seed(0)
    for _, OUT, IN in specs:
        lin = torch.nn.Linear(IN, OUT, bias=False).to(dev).to(torch.bfloat16)
        quantize_(lin, cfg); tiny.append(lin)
        if IN not in xs: xs[IN] = torch.randn(1, IN, device=dev, dtype=torch.bfloat16)
    def run_tiny():
        for lin, (_, _, IN) in zip(tiny, specs): lin(xs[IN])

    tot_bytes = sum(OUT * IN * 0.5 + (IN // G) * OUT * 2 for _, OUT, IN in specs)
    mt = graph_time(run_marlin); tt = graph_time(run_tiny)
    def mbu(us): return 100 * tot_bytes / (us * 1e-6) / a.peak_bw
    print(f"\n  total weight bytes/step: {tot_bytes/1e9:.2f} GB")
    print(f"  MARLIN   linear-only {mt:7.1f} us   MBU {mbu(mt):4.1f}%")
    print(f"  tinygemm linear-only {tt:7.1f} us   MBU {mbu(tt):4.1f}%   ({mt<tt and 'Marlin faster' or 'tinygemm faster'})")
    overhead = TINYGEMM_STEP_US - tt
    for name, lin in [("MARLIN", mt), ("tinygemm", tt)]:
        step = lin + overhead
        print(f"  projected END-TO-END  {name:8s} {1e6/step:6.1f} tok/s   (x2 MTP -> {2e6/step:.0f})")
