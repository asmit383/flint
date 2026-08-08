#!/usr/bin/env python3
"""Does torch.compile + static cache get Granite bf16 to the GEMM floor (~5-6ms)? Uncompiled HF was 45ms
(overhead). If compiled decode hits ~6-10ms, the bf16 verify is cheap and Tier-1 spec-decode is viable.

    python bench/bf16_compiled_test.py
"""
import time, torch
from transformers import AutoModelForCausalLM

dev = "cuda"
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.1-3b", torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa", trust_remote_code=True).to(dev).eval()
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
V = model.config.vocab_size
ids = torch.randint(0, V, (1, 64), device=dev)

with torch.no_grad():
    print("compiling (first generate)...", flush=True)
    model.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=0)   # warmup + compile
    for _ in range(2): model.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=0)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=192, do_sample=False, pad_token_id=0)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0

n = out.shape[1] - ids.shape[1]
print(f"compiled bf16 generate: {n/dt:.0f} tok/s   ({dt/n*1000:.1f} ms/token)")
print(f"(uncompiled was ~21 tok/s / 45ms. int4 megakernel single-pass = 275. GEMM floor ~5-6ms.)")
