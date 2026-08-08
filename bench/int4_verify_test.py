#!/usr/bin/env python3
"""Tier-2 de-risk: int4 tensor-core (tinygemm/torchao) M=K verify cost. If the compiled int4 forward at M=8 is
~flat (~4ms, 1x bytes vs bf16's 4x), spec-decode nets: chain 1.76/4ms=440, tree 2.46/4ms=615 -> clears 350/550.
Random weights (speed is value-independent), gpt-fast model, compiled + CUDA graphs.

    python bench/int4_verify_test.py
"""
import sys, time, torch
sys.path.insert(0, "/root/gpt-fast")
from model import Transformer, ModelArgs

dev = "cuda"
args = ModelArgs(block_size=2048, vocab_size=100352, n_layer=40, n_head=40, n_local_heads=8,
                 dim=2560, intermediate_size=8192, rope_base=10000000)
with torch.device(dev):
    model = Transformer(args).to(torch.bfloat16).eval()
from torchao.quantization import quantize_, Int4WeightOnlyConfig
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D))
with torch.device(dev):
    model = model.eval(); model.setup_caches(1, 2048)
fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
P = 512

@torch.no_grad()
def measure(M, iters=40):
    ids = torch.randint(0, args.vocab_size, (1, P), device=dev)
    model(ids, torch.arange(P, device=dev))                  # prefill
    x = torch.randint(0, args.vocab_size, (1, M), device=dev); pos = torch.arange(P, P + M, device=dev)
    for _ in range(6):
        torch.compiler.cudagraph_mark_step_begin(); fwd(x, pos).clone()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        torch.compiler.cudagraph_mark_step_begin(); fwd(x, pos).clone()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1000

print("int4 compiled verify (gpt-fast, tensor-core):")
t1 = measure(1)
for M in (1, 4, 8):
    t = measure(M)
    print(f"  M={M:>2}  {t:6.2f} ms  ({t/t1:4.2f}x M=1)")
print(f"\nspec-decode net = accepted / verify_time:")
t8 = measure(8)
for acc in (1.76, 2.46):
    print(f"  acc {acc}: {acc/t8*1000:.0f} tok/s")
print(f"(bf16 verify was 8.9ms; int4 should be ~4ms. 350 needs > ~2.8 tok/verify-ms-equiv.)")
