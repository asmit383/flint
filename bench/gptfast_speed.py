#!/usr/bin/env python3
"""Native gpt-fast decode speed at Granite-4.1-3B dims (bf16, torch.compile + CUDA graphs).
Speed is value-independent, so we use random weights + gpt-fast's real compiled decode path (classic
SDPA gpt-fast) — the honest "what does gpt-fast do here" number, no checkpoint/tokenizer needed.
Run:  python bench/gptfast_speed.py --peak-bw 1.55e12
"""
import sys, time, argparse, torch
sys.path.insert(0, "/root/gpt-fast")
from model import Transformer, ModelArgs
from generate import decode_one_token, prefill

ap = argparse.ArgumentParser()
ap.add_argument("--peak-bw", type=float, default=1.55e12)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--int4", action="store_true", help="quantize to gpt-fast int4 (tinygemm)")
a = ap.parse_args()

dev = "cuda"
args = ModelArgs(block_size=a.seqlen, vocab_size=100352, n_layer=40, n_head=40,
                 n_local_heads=8, dim=2560, intermediate_size=8192, rope_base=10000000)
with torch.device(dev):
    model = Transformer(args).to(torch.bfloat16).eval()
nparams = sum(p.numel() for p in model.parameters())
bytes_tok = nparams * 2
label = "bf16"
if a.int4:
    # torchao int4 = same aten _weight_int4pack_mm (tinygemm) kernel gpt-fast uses.
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Int4WeightOnlyConfig   # torchao >= 0.8 (config-based)
        # tile-packed (tensor-core tiled) = classic aten _weight_int4pack_mm tinygemm; the default
        # PLAIN format in torchao 0.18 needs an mslk lib we don't have.
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
    except ImportError:
        from torchao.quantization import int4_weight_only        # older torchao
        cfg = int4_weight_only(group_size=128)
    quantize_(model, cfg)
    bytes_tok = nparams * 0.5 + 100352 * 2560 * 2   # int4 linears + bf16 embeds (approx)
    label = "int4"
with torch.device(dev):
    model = model.eval()
    model.setup_caches(max_batch_size=1, max_seq_length=a.seqlen)
print(f"gpt-fast @ Granite-3B dims [{label}]: params={nparams/1e9:.2f}B  bytes/tok~{bytes_tok/1e9:.2f} GB")

decode = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

plen = 8
x = torch.randint(0, args.vocab_size, (1, plen), device=dev)
cur = prefill(model, x, torch.arange(plen, device=dev)).view(1, 1)
ipos = torch.tensor([plen], device=dev)

def step(cur, ipos):
    torch.compiler.cudagraph_mark_step_begin()           # torch 2.11+: new cudagraph step
    out, _ = decode(model, cur, ipos)
    return out.clone().view(1, 1), ipos + 1              # clone out of the graph pool before reuse

with torch.no_grad():
    for _ in range(10):                                  # warmup + compile
        cur, ipos = step(cur, ipos)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.gen):
        cur, ipos = step(cur, ipos)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

toks = a.gen / dt
print(f"  gpt-fast {label} (compiled) B=1: {toks:.1f} tok/s   MBU ~ {100*bytes_tok*toks/a.peak_bw:.1f}%")
