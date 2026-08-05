#!/usr/bin/env python3
"""Quick B=1 decode throughput (transformers eager) — a first floor, not the optimized baseline.
Reports tok/s + MBU (weights streamed per token / peak BW). The real baseline is gpt-fast (next).
Run:  python bench/throughput.py --model ibm-granite/granite-4.1-3b --peak-bw 1.55e12
"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--peak-bw", type=float, default=1.55e12, help="A100-40 1.55e12; A100-80/H100-PCIe 2.0e12; H200 4.8e12")
a = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=dev,
                                             trust_remote_code=True).eval()
nparams = sum(p.numel() for p in model.parameters())
bytes_tok = nparams * 2   # bf16 weights streamed per decode step

ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
gen = dict(max_new_tokens=a.gen, do_sample=False, use_cache=True)

with torch.no_grad():
    model.generate(**{"inputs": ids, **{"max_new_tokens": 32, "do_sample": False, "use_cache": True}})  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(inputs=ids, **gen)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

n = out.shape[1] - ids.shape[1]
toks = n / dt
print(f"model={a.model}  params={nparams/1e9:.2f}B  bf16 bytes/tok={bytes_tok/1e9:.2f} GB")
print(f"  transformers eager B=1: {n} tok in {dt:.3f}s -> {toks:.1f} tok/s   MBU = {100*bytes_tok*toks/a.peak_bw:.1f}%")
