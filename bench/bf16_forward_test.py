#!/usr/bin/env python3
"""Tier-1 de-risk: how fast is the bf16 target FORWARD (the spec-decode verify) with SDPA attention, and does
it stay flat M=1 -> M=8? The cuBLAS GEMMs alone are ~4.7ms; the earlier 4.3 tok/s was eager attention + python
overhead. Measures a fixed-context forward at M=1/4/8 (KV cache cropped each iter). If M=8 ~= M=1 and both are
small, spec-decode verify is cheap and Tier 1 is viable.

    python bench/bf16_forward_test.py
"""
import time, torch
from transformers import AutoModelForCausalLM, DynamicCache

dev = "cuda"
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.1-3b", torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa", trust_remote_code=True).to(dev).eval()
vocab = model.config.vocab_size
P = 512

@torch.no_grad()
def measure(M, iters=30):
    ids = torch.randint(0, vocab, (1, P), device=dev)
    cache = DynamicCache()
    model(ids, past_key_values=cache, use_cache=True)                # prefill P
    x = torch.randint(0, vocab, (1, M), device=dev)
    for _ in range(5):
        cache.crop(P); model(x, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        cache.crop(P); model(x, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

print(f"bf16 Granite forward (SDPA), context {P}:")
t1 = measure(1)
for M in (1, 4, 8):
    t = measure(M)
    print(f"  M={M:>2}  {t:6.2f} ms   ({t/t1:4.2f}x M=1)   single-pass equiv {1000/t:4.0f} tok/s")
print(f"\nspec-decode net (if verify=this): accepted / verify_time.")
for M, acc in ((8, 1.87), (8, 2.46)):
    t = measure(M); print(f"  verify M={M} ({t:.1f}ms) x acc {acc} -> {acc/t*1000:.0f} tok/s")
print("compare: int4 megakernel single-pass = 275 tok/s. Verify must be cheap for spec-decode to win.")
