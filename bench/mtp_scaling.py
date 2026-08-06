#!/usr/bin/env python3
"""MTP headroom: how does a decode forward pass cost scale with M candidate tokens (the speculative
verify batch)? Decode is memory-bound — weights read ONCE per pass — so if the M-token pass costs ~the
same as M=1, verifying M tokens/pass gives up to M x throughput (realized = acceptance x M). At M>1 the
int4 GEMM uses tensor cores efficiently (the M=1 padding waste is gone) — the regime that makes MTP win.

Reports, per M:  T_M (us),  tok/s ceiling = M / T_M  (perfect acceptance),  and speedup vs M=1.

    python bench/mtp_scaling.py --peak-bw 2.0e12 --int4
"""
import sys, time, argparse, torch
sys.path.insert(0, "/root/gpt-fast")
from model import Transformer, ModelArgs

ap = argparse.ArgumentParser()
ap.add_argument("--peak-bw", type=float, default=2.0e12)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--int4", action="store_true")
ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 12, 16])
a = ap.parse_args()

dev = "cuda"
args = ModelArgs(block_size=a.seqlen, vocab_size=100352, n_layer=40, n_head=40,
                 n_local_heads=8, dim=2560, intermediate_size=8192, rope_base=10000000)
with torch.device(dev):
    model = Transformer(args).to(torch.bfloat16).eval()
label = "bf16"
if a.int4:
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
    except ImportError:
        from torchao.quantization import int4_weight_only
        cfg = int4_weight_only(group_size=128)
    quantize_(model, cfg); label = "int4"
with torch.device(dev):
    model = model.eval(); model.setup_caches(max_batch_size=1, max_seq_length=a.seqlen)

nparams = sum(p.numel() for p in model.parameters())
print(f"gpt-fast Granite-3B [{label}]  ({nparams/1e9:.2f}B params)  — MTP verify-pass scaling")

def time_M(M, start=64, iters=200):
    x = torch.randint(0, args.vocab_size, (1, M), device=dev)
    pos = torch.arange(start, start + M, device=dev)
    fwd = torch.compile(lambda idx, p: model(idx, p), mode="reduce-overhead", fullgraph=True)
    def step():
        torch.compiler.cudagraph_mark_step_begin()
        return fwd(x, pos).clone()
    with torch.no_grad():
        for _ in range(12): step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): step()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

print(f"\n  {'M':>3}  {'T_M (us)':>9}  {'tok/s ceiling':>14}  {'speedup vs M=1':>15}")
t1 = None
for M in a.Ms:
    torch._dynamo.reset()
    t = time_M(M)
    t1 = t1 or t
    print(f"  {M:>3}  {t*1e6:>9.1f}  {M/t:>14.0f}  {t1*M/t:>14.2f}x")
print("\n1500 tok/s needs: (tok/s ceiling at chosen M) x (acceptance rate) >= 1500.")
