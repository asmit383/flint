#!/usr/bin/env python3
"""EAGLE step 2 — train the draft head. One lightweight transformer block that, given the target's
feature f_i and the embedding of the actual next token t_{i+1}, predicts the next feature f_{i+1}.
Token prediction reuses the target's FROZEN norm + lm_head. Loss = SmoothL1 on features + CE on the
token the predicted feature implies. Draft is ~1/40th the target's depth -> cheap to roll K times.

    python train/train_eagle.py --data /root/eagle_data --epochs 3 --out /root/eagle_data/draft.pt
"""
import argparse, glob, os, sys, math, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_model import DraftHead, rmsnorm

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/eagle_data")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--feat-w", type=float, default=0.1)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--inter", type=int, default=4096)
ap.add_argument("--rollout", type=int, default=3)            # multi-step: train on the draft's OWN predictions (EAGLE-2 fix)
ap.add_argument("--out", default="/root/eagle_data/draft.pt")
a = ap.parse_args()
dev = "cuda"

H = torch.load(os.path.join(a.data, "heads.pt"), map_location="cpu")
dim, vocab, nh, lsc = H["dim"], H["vocab"], H["n_heads"], H["logits_scaling"]
fuse = H.get("fuse", 1)                                     # EAGLE-3 multi-layer fusion factor
embed = H["embed"].to(dev)                                  # [vocab, dim] frozen
lm_head = H["lm_head"].to(dev)                              # [vocab, dim] frozen (tied)
norm_w = H["norm"]["weight"].to(dev)                        # RMSNorm weight, frozen

draft = DraftHead(dim, nh, a.inter, fuse=fuse).to(dev).to(torch.bfloat16)
nparam = sum(p.numel() for p in draft.parameters())
print(f"draft head: {nparam/1e6:.1f}M params (dim={dim}, heads={nh}, inter={a.inter}, fuse={fuse})")

shards = sorted(glob.glob(os.path.join(a.data, "*shard_*.pt")))
data = [torch.load(s, map_location="cpu") for s in shards]
ids = torch.cat([d["ids"] for d in data]); feat = torch.cat([d["feat"] for d in data])
N, ctx = ids.shape
print(f"data: {N} seqs x {ctx} tok")

opt = torch.optim.AdamW(draft.parameters(), lr=a.lr, weight_decay=a.wd)
steps = (N // a.bs) * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)

def token_logits(fp):                                       # fused fp -> head reads HIGH slice (last dim)
    return F.linear(rmsnorm(fp[..., -dim:], norm_w), lm_head) / lsc

step = 0
for ep in range(a.epochs):
    perm = torch.randperm(N)
    for b in range(0, N - a.bs + 1, a.bs):
        idx = perm[b:b + a.bs]
        f = feat[idx].to(dev).to(torch.bfloat16)            # [B,ctx,dim]
        t = ids[idx].to(dev)
        # multi-step rollout: at depth d feed the draft's OWN predicted feature forward (not the target's),
        # so it learns to recover from its own error — matches inference, closes the exposure-bias gap.
        D = a.rollout; L = f.shape[1] - D - 1
        fcur = f[:, :L]; floss = 0.0; tloss = 0.0; logits = None; tok_tgt = None
        for d in range(D):
            emb_d = F.embedding(t[:, d + 1:d + 1 + L], embed).to(torch.bfloat16)
            pos_d = torch.arange(d, d + L, device=dev)
            fp = draft(fcur, emb_d, pos_d)
            f_tgt_d = f[:, d + 1:d + 1 + L]; tok_tgt_d = t[:, d + 2:d + 2 + L]
            floss = floss + F.smooth_l1_loss(fp.float(), f_tgt_d.float())
            lg = token_logits(fp).float()
            tloss = tloss + F.cross_entropy(lg.reshape(-1, vocab), tok_tgt_d.reshape(-1))
            if d == 0: logits, tok_tgt = lg, tok_tgt_d
            fcur = fp                                       # <- own prediction forward
        floss = floss / D; tloss = tloss / D
        loss = tloss + a.feat_w * floss
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            acc = (logits.argmax(-1) == tok_tgt).float().mean().item()
            print(f"  ep{ep} step{step}/{steps}  tok_ce {tloss.item():.3f}  feat {floss.item():.3f}  "
                  f"1-tok-acc {100*acc:.1f}%", flush=True)
        step += 1

torch.save({"state": draft.state_dict(), "dim": dim, "nh": nh, "inter": a.inter, "fuse": fuse}, a.out)
print(f"DONE -> {a.out}  (1-tok-acc is the ceiling for draft depth 1)")
