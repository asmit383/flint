#!/usr/bin/env python3
"""EAGLE step 1 — harvest training data. Run the target (Granite) on a corpus and save, per position,
its last decoder-layer hidden state (the "feature" the draft head extrapolates) + the input token ids.
EAGLE drafts on these rich features, not raw tokens — that's why it beats n-gram lookup.

    python train/harvest.py --model ibm-granite/granite-4.1-3b --seqs 2000 --out /root/eagle_data
"""
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--seqs", type=int, default=2000)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default="/root/eagle_data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()
dim = model.config.hidden_size
print(f"model={a.model} dim={dim} | harvesting {a.seqs} x {a.ctx}-tok seqs -> {a.out}")

# WikiText-103 TRAIN (~100M tokens; far more than wt-2 for a real draft)
parts = []
for fn in ["wikitext-103-raw-v1/train-00000-of-00002.parquet",
           "wikitext-103-raw-v1/train-00001-of-00002.parquet"]:
    try:
        pq = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset", filename=fn)
        parts.append(pd.read_parquet(pq))
    except Exception as e:
        print(f"  skip {fn}: {e}")
if not parts:                                              # fallback to wt-2
    pq = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                         filename="wikitext-2-raw-v1/train-00000-of-00001.parquet")
    parts = [pd.read_parquet(pq)]
df = pd.concat(parts, ignore_index=True)
text = "\n\n".join(t for t in df["text"].tolist() if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0]
nseq = min(a.seqs, ids.shape[0] // a.ctx)
ids = ids[:nseq * a.ctx].view(nseq, a.ctx)
print(f"corpus: {ids.shape[0]} sequences ({ids.numel()} tokens)")

shard_ids, shard_feat, si = [], [], 0
def flush():
    global shard_ids, shard_feat, si
    if not shard_ids: return
    torch.save({"ids": torch.cat(shard_ids), "feat": torch.cat(shard_feat)},
               os.path.join(a.out, f"shard_{si:04d}.pt"))
    shard_ids, shard_feat = [], []; si += 1

with torch.no_grad():
    for b in range(0, nseq, a.bs):
        batch = ids[b:b + a.bs].to(dev)
        out = model(batch, output_hidden_states=True)
        feat = out.hidden_states[-1].to(torch.bfloat16).cpu()   # [B, ctx, dim] last decoder-layer output
        shard_ids.append(batch.cpu()); shard_feat.append(feat)
        if sum(x.shape[0] for x in shard_ids) >= 256: flush()
        if (b // a.bs) % 10 == 0: print(f"  {b+len(batch)}/{nseq}", flush=True)
flush()
# also stash the frozen heads the draft reuses for token prediction
torch.save({"norm": model.model.norm.state_dict(), "lm_head": model.lm_head.weight.detach().cpu(),
            "embed": model.model.embed_tokens.weight.detach().cpu(),
            "logits_scaling": float(getattr(model.config, "logits_scaling", 1.0)),
            "dim": dim, "vocab": model.config.vocab_size,
            "n_heads": model.config.num_attention_heads},
           os.path.join(a.out, "heads.pt"))
print(f"DONE: {si} shards + heads.pt in {a.out}")
