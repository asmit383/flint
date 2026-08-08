#!/usr/bin/env python3
"""harvest_mega.py — self-distill the drafter on the MEGAKERNEL's OWN int4 features (not HF bf16). The draft
infers on verify_mega's post-final-norm feature, so train on exactly that: greedy-generate with the int4
megakernel, capture (feature, token) at each position. Closes the train/infer gap on BOTH axes — the feature
precision (int4, not bf16) and the token distribution (the megakernel's own greedy path).

    python train/harvest_mega.py --seqs 2500 --gen 128 --out /root/eagle_mega
"""
import os, sys, argparse, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from chat import load_packed, DIM, NKV, HD, NL, VOCAB, NH, SCALE, ROPE, RESID, LOGS, EMB_MULT, MODEL
from transformers import AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--seqs", type=int, default=2500)
ap.add_argument("--seed-len", type=int, default=32)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--out", default="/root/eagle_mega")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"; torch.set_grad_enabled(False)

dec = load(name="flint_dec", sources=[HERE + "/kernels/megakernel_decode.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
ver = load(name="flint_ver", sources=[HERE + "/kernels/verify_mega.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
W = load_packed(dev)
tok = AutoTokenizer.from_pretrained(MODEL); eos = tok.eos_token_id
MAXP = a.seed_len + a.gen + 4
kc = torch.zeros(NL, MAXP, NKV, HD, device=dev, dtype=torch.bfloat16)
vc = torch.zeros(NL, MAXP, NKV, HD, device=dev, dtype=torch.bfloat16)
embed = W["embed"]
D1 = torch.zeros(1, dtype=torch.int32, device=dev); A1 = torch.ones(1, 1, dtype=torch.uint8, device=dev)

def dec1(t, p):
    h = (embed[t].float() * EMB_MULT).to(torch.bfloat16)
    return dec.decode_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
        W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, p, SCALE, ROPE, RESID, LOGS)

def ver1(t, p):                                        # M=1 verify -> (logits[VOCAB], feature[DIM]) post-norm
    h = (embed[t].float() * EMB_MULT).to(torch.bfloat16).view(1, DIM)
    lg, ft = ver.verify_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
        W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, D1, A1, p, 1, SCALE, ROPE, RESID, LOGS)
    return lg[0], ft[0]

ds = load_dataset("flytech/python-codes-25k", split="train")
paras = [t for t in ds["output"] if len(t.split()) > 20]
seeds = []
for p in paras:
    ii = tok(p, return_tensors="pt").input_ids[0]
    if ii.shape[0] >= a.seed_len: seeds.append(ii[:a.seed_len])
    if len(seeds) >= a.seqs: break
print(f"seeds: {len(seeds)} | gen {a.gen} | out {a.out}", flush=True)

shard_t, shard_f, si = [], [], 0
def flush():
    global shard_t, shard_f, si
    if not shard_t: return
    torch.save({"ids": torch.stack(shard_t), "feat": torch.stack(shard_f)},
               os.path.join(a.out, f"mshard_{si:04d}.pt")); shard_t, shard_f = [], []; si += 1

for n, seed in enumerate(seeds):
    seed = seed.to(dev)
    for i in range(a.seed_len - 1): dec1(int(seed[i]), i)   # prefill seed 0..len-2
    pos = a.seed_len - 1; cur = int(seed[-1])
    toks, feats = [], []
    for g in range(a.gen):                                  # fixed length (ignore EOS) -> uniform shards
        lg, ft = ver1(cur, pos)                             # feature@pos, logits->next
        toks.append(cur); feats.append(ft.to(torch.bfloat16).cpu())
        cur = int(lg.argmax()); pos += 1
    shard_t.append(torch.tensor(toks, dtype=torch.long)); shard_f.append(torch.stack(feats))
    if len(shard_t) >= 256: flush()
    if n % 100 == 0: print(f"  {n}/{len(seeds)}", flush=True)
flush()
torch.save({"norm": {"weight": W["nf"].cpu()}, "lm_head": embed.cpu(), "embed": embed.cpu(),
            "logits_scaling": float(LOGS), "dim": DIM, "vocab": VOCAB, "n_heads": NH, "fuse": 1},
           os.path.join(a.out, "heads.pt"))
print(f"DONE: {si} shards + heads.pt in {a.out}")
