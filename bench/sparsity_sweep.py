#!/usr/bin/env python3
"""GO/NO-GO GATE: does Granite-4.1 tolerate activation sparsity?

Masks the smallest-magnitude MLP intermediate activations (the SwiGLU output feeding down_proj —
CATS/TEAL-style row-skip) at increasing fractions and measures WikiText-2 perplexity. If ppl stays
tolerable to ~40-50% sparsity, the byte-cutting differentiator is real -> build the kernel. If it
blows up by ~20% (Muon-flat activations), SWITCH MODELS before writing anything.

Run:  python bench/sparsity_sweep.py --model ibm-granite/granite-4.1-3b
"""
import argparse, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--chunks", type=int, default=20)
ap.add_argument("--ctx", type=int, default=2048)
a = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()

STATE = {"s": 0.0}


def mask_hook(module, inp):
    # inp[0] = down_proj input [.., inter]; zero the smallest-|.| fraction s per token (row).
    s = STATE["s"]
    if s <= 0:
        return inp
    x = inp[0]
    k = int(x.shape[-1] * s)
    if k <= 0:
        return inp
    thresh = x.abs().kthvalue(k, dim=-1, keepdim=True).values
    return (torch.where(x.abs() > thresh, x, torch.zeros_like(x)),) + tuple(inp[1:])


n = 0
for name, mod in model.named_modules():
    if name.endswith("down_proj") and isinstance(mod, torch.nn.Linear):
        mod.register_forward_pre_hook(mask_hook)
        n += 1
print(f"model={a.model}  hooked {n} down_proj modules")

pq = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                     filename="wikitext-2-raw-v1/test-00000-of-00001.parquet")
text = "\n\n".join(t for t in pd.read_parquet(pq)["text"].tolist() if t.strip())
enc = tok(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def ppl():
    nlls = []
    for i in range(a.chunks):
        ids = enc[i * a.ctx:(i + 1) * a.ctx].unsqueeze(0).to(dev)
        if ids.shape[1] < 2:
            break
        nlls.append(model(ids, labels=ids).loss.item())
    return math.exp(sum(nlls) / len(nlls))


base = None
print("=== activation-sparsity tolerance (MLP intermediate, WikiText-2 ppl) ===")
for s in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
    STATE["s"] = s
    p = ppl()
    base = base or p
    verdict = "" if s == 0 else ("  OK" if p / base - 1 < 0.06 else "  <-- too costly")
    print(f"  sparsity {int(s*100):2d}%   ppl {p:8.3f}   Δ {100*(p/base-1):+6.1f}%{verdict}")
print("GATE: build the kernel if ~40-50% stays under ~+6% ppl; else switch models.")
