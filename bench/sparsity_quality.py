#!/usr/bin/env python3
"""Where does the activation-sparsity ppl loss come from — the thresholding SCHEME or calibration?
Compares three ways to hit ~50% sparsity on the MLP intermediate (down_proj input), same eval set:

  per-token top-k   dynamic, exactly 50% per token, NO calibration   (what sparsity_sweep.py used)
  global threshold  one magnitude cutoff for ALL layers, calibrated to 50% avg
  per-layer thresh  each down_proj its own cutoff, calibrated to 50%  (TEAL-style)

Reports WikiText-2 ppl AND achieved avg sparsity for each. If global >> per-layer/top-k, the loss is a
scheme artifact (uncalibrated), not intrinsic to 50% sparsity.

Run:  python bench/sparsity_quality.py --model ibm-granite/granite-4.1-3b
"""
import argparse, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--target", type=float, default=0.5)
ap.add_argument("--eval-chunks", type=int, default=20)
ap.add_argument("--calib-chunks", type=int, default=8)
ap.add_argument("--ctx", type=int, default=2048)
a = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()

# ---- hooks: one shared state, indexed per down_proj module ----
mods = []
STATE = {"mode": "off", "target": a.target, "tglobal": 0.0, "tlayer": None,
         "collect": None, "zeros": 0, "total": 0}

def hook(idx):
    def f(module, inp):
        x = inp[0]
        if STATE["mode"] == "calib":
            STATE["collect"][idx].append(x.detach().abs().flatten()[::17].float().cpu())  # subsample
            return inp
        if STATE["mode"] == "off":
            return inp
        ax = x.abs()
        if STATE["mode"] == "topk":
            k = int(x.shape[-1] * STATE["target"])
            th = ax.kthvalue(k, dim=-1, keepdim=True).values
            mask = ax > th
        elif STATE["mode"] == "global":
            mask = ax > STATE["tglobal"]
        elif STATE["mode"] == "layer":
            mask = ax > STATE["tlayer"][idx]
        STATE["zeros"] += (~mask).sum().item(); STATE["total"] += mask.numel()
        return (torch.where(mask, x, torch.zeros_like(x)),) + tuple(inp[1:])
    return f

for name, mod in model.named_modules():
    if name.endswith("down_proj") and isinstance(mod, torch.nn.Linear):
        mod.register_forward_pre_hook(hook(len(mods))); mods.append(name)
NL = len(mods)
print(f"model={a.model}  hooked {NL} down_proj  target sparsity {int(a.target*100)}%")

pq = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                     filename="wikitext-2-raw-v1/test-00000-of-00001.parquet")
enc = tok("\n\n".join(t for t in pd.read_parquet(pq)["text"].tolist() if t.strip()),
          return_tensors="pt").input_ids[0]

@torch.no_grad()
def run_chunks(lo, hi):
    nlls = []
    for i in range(lo, hi):
        ids = enc[i * a.ctx:(i + 1) * a.ctx].unsqueeze(0).to(dev)
        if ids.shape[1] < 2: break
        nlls.append(model(ids, labels=ids).loss.item())
    return math.exp(sum(nlls) / len(nlls))

# ---- calibration pass (chunks AFTER the eval range, disjoint) ----
print("calibrating thresholds ...")
STATE["mode"] = "calib"; STATE["collect"] = [[] for _ in range(NL)]
with torch.no_grad():
    run_chunks(a.eval_chunks, a.eval_chunks + a.calib_chunks)
def qtile(t, q, cap=8_000_000):
    if t.numel() > cap: t = t[torch.randperm(t.numel())[:cap]]
    return t.quantile(q).item()
per_layer_cat = [torch.cat(c) for c in STATE["collect"]]
tlayer = torch.tensor([qtile(c, a.target) for c in per_layer_cat])
tglobal = qtile(torch.cat(per_layer_cat), a.target)
STATE["tlayer"] = tlayer.tolist(); STATE["tglobal"] = tglobal; STATE["collect"] = None
print(f"  global threshold {tglobal:.4f} | per-layer range [{tlayer.min():.4f}, {tlayer.max():.4f}]  "
      f"(spread {tlayer.max()/tlayer.min():.1f}x -> why global mis-sparsifies)")

# ---- eval each scheme ----
def evalmode(mode):
    STATE["mode"] = mode; STATE["zeros"] = 0; STATE["total"] = 0
    p = run_chunks(0, a.eval_chunks)
    sp = STATE["zeros"] / max(STATE["total"], 1) if STATE["total"] else a.target
    return p, sp

STATE["mode"] = "off"; base = run_chunks(0, a.eval_chunks)
print(f"\n  dense                 ppl {base:8.3f}")
for mode, label in [("topk", "per-token top-k"), ("global", "global threshold"), ("layer", "per-layer calib")]:
    p, sp = evalmode(mode)
    print(f"  {label:20s}  ppl {p:8.3f}   Δ {100*(p/base-1):+6.1f}%   (sparsity {100*sp:4.1f}%)")
