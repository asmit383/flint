#!/usr/bin/env python3
"""REAL end-to-end fused decode: build the gpt-fast Granite model, optionally fuse gate/up (w1+w3 -> one
2560->16384 GEMV; QKV is already fused as wqkv), quantize int4, CUDA-graph the decode step, and generate
tokens on a clock. This is the honest confirmation of the 302 projection — a measured tok/s, not a sum.

    python bench/e2e_fused.py --peak-bw 2.0e12 --int4
"""
import sys, time, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/root/gpt-fast")
from model import Transformer, ModelArgs
from generate import decode_one_token, prefill

ap = argparse.ArgumentParser()
ap.add_argument("--peak-bw", type=float, default=2.0e12)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--int4", action="store_true")
ap.add_argument("--gen", type=int, default=128)
a = ap.parse_args()
dev = "cuda"

class FusedFF(nn.Module):
    """gate/up fused into one GEMV, split in-register."""
    def __init__(self, ff):
        super().__init__()
        dim, inter = ff.w1.in_features, ff.w1.out_features
        self.w13 = nn.Linear(dim, 2 * inter, bias=False).to(ff.w1.weight.dtype).to(ff.w1.weight.device)
        with torch.no_grad():
            self.w13.weight.copy_(torch.cat([ff.w1.weight, ff.w3.weight]))
        self.w2 = ff.w2
    def forward(self, x):
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)

def build(fuse):
    args = ModelArgs(block_size=a.seqlen, vocab_size=100352, n_layer=40, n_head=40,
                     n_local_heads=8, dim=2560, intermediate_size=8192, rope_base=10000000)
    with torch.device(dev):
        model = Transformer(args).to(torch.bfloat16).eval()
    if fuse:
        for blk in model.layers:
            blk.feed_forward = FusedFF(blk.feed_forward)
    nparams = sum(p.numel() for p in model.parameters())
    bytes_tok = nparams * 2
    if a.int4:
        from torchao.quantization import quantize_, Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        quantize_(model, Int4WeightOnlyConfig(group_size=128,
                  int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D))
        bytes_tok = nparams * 0.5 + 100352 * 2560 * 2
    with torch.device(dev):
        model = model.eval(); model.setup_caches(max_batch_size=1, max_seq_length=a.seqlen)
    return model, args, bytes_tok

def measure(model, args):
    decode = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    x = torch.randint(0, args.vocab_size, (1, 8), device=dev)
    cur = prefill(model, x, torch.arange(8, device=dev)).view(1, 1)
    ipos = torch.tensor([8], device=dev)
    def step(cur, ipos):
        torch.compiler.cudagraph_mark_step_begin()
        out, _ = decode(model, cur, ipos)
        return out.clone().view(1, 1), ipos + 1
    with torch.no_grad():
        for _ in range(12): cur, ipos = step(cur, ipos)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(a.gen): cur, ipos = step(cur, ipos)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return a.gen / dt

print(f"gpt-fast Granite-3B [{'int4' if a.int4 else 'bf16'}] — REAL end-to-end decode (measured)\n")
for fuse in [False, True]:
    torch._dynamo.reset()
    model, args, bt = build(fuse)
    tps = measure(model, args)
    tag = "gate/up FUSED" if fuse else "baseline (qkv fused, gate/up separate)"
    print(f"  {tag:42s} {tps:6.1f} tok/s   MBU {100*bt*tps/a.peak_bw:4.1f}%")
