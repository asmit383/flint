#!/usr/bin/env python3
"""EAGLE-3 data — harvest from the TARGET's OWN GENERATIONS (self-distillation), not raw corpus. This is
the fix for the distribution mismatch that capped our WikiText-trained draft at ~1.1 tokens/pass: the
draft must see the same distribution it drafts on at inference (the model's greedy continuations).

Seed from diverse WikiText passages, greedy-generate with Granite (batched), then one forward with
output_hidden_states to grab the last-layer features along the generation.

    python train/harvest_gen.py --seqs 2000 --gen 224 --out /root/eagle_gen
"""
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--seqs", type=int, default=2000)
ap.add_argument("--seed-len", type=int, default=32)
ap.add_argument("--gen", type=int, default=224)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default="/root/eagle_gen")
ap.add_argument("--domain", default="code", choices=["wiki", "code"])  # regime to specialize the drafter on
ap.add_argument("--fuse", type=int, default=1)               # 1 = last-layer feature; 3 = EAGLE-3 low/mid/high
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()
dim = model.config.hidden_size
NL = model.config.num_hidden_layers
FUSE_LAYERS = [NL] if a.fuse == 1 else [NL // 4, NL // 2, NL]  # fuse=1: last-layer only; else low/mid/high
print(f"model={a.model} dim={dim} | fuse layers {FUSE_LAYERS} | generating {a.seqs} x {a.gen}-tok (bs={a.bs})")

if a.domain == "code":
    from datasets import load_dataset
    ds = load_dataset("flytech/python-codes-25k", split="train")   # Python code, ungated
    paras = [t for t in ds["output"] if len(t.split()) > 20]
else:
    pq = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                         filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    paras = [t for t in pd.read_parquet(pq)["text"].tolist() if len(t.split()) > 40]
# build seeds: first seed_len tokens of distinct passages
seeds = []
for p in paras:
    ids = tok(p, return_tensors="pt").input_ids[0]
    if ids.shape[0] >= a.seed_len:
        seeds.append(ids[:a.seed_len])
    if len(seeds) >= a.seqs: break
print(f"seeds: {len(seeds)}")

shard_t, shard_f, si = [], [], 0
def flush():
    global shard_t, shard_f, si
    if not shard_t: return
    torch.save({"ids": torch.cat(shard_t), "feat": torch.cat(shard_f)},
               os.path.join(a.out, f"gshard_{si:04d}.pt")); shard_t, shard_f = [], []; si += 1

with torch.no_grad():
    for b in range(0, len(seeds), a.bs):
        batch = torch.stack(seeds[b:b + a.bs]).to(dev)                # [bs, seed_len]
        gen = model.generate(batch, do_sample=False, max_new_tokens=a.gen,
                             min_new_tokens=a.gen,                     # force full length -> uniform shards
                             pad_token_id=tok.pad_token_id)            # [bs, seed+gen]
        hs = model(gen, output_hidden_states=True).hidden_states   # tuple len NL+1
        feat = torch.cat([hs[i] for i in FUSE_LAYERS], -1).to(torch.bfloat16).cpu()  # [bs, L, fuse*dim]
        shard_t.append(gen.cpu()); shard_f.append(feat)
        if sum(x.shape[0] for x in shard_t) >= 256: flush()
        if (b // a.bs) % 5 == 0: print(f"  {b+len(batch)}/{len(seeds)}", flush=True)
flush()
torch.save({"norm": model.model.norm.state_dict(), "lm_head": model.lm_head.weight.detach().cpu(),
            "embed": model.model.embed_tokens.weight.detach().cpu(),
            "logits_scaling": float(getattr(model.config, "logits_scaling", 1.0)),
            "dim": dim, "vocab": model.config.vocab_size,
            "n_heads": model.config.num_attention_heads,
            "fuse": len(FUSE_LAYERS), "fuse_layers": FUSE_LAYERS},
           os.path.join(a.out, "heads.pt"))
print(f"DONE: {si} shards + heads.pt in {a.out}")
