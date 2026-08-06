#!/usr/bin/env python3
"""EAGLE step 3 — measure REAL acceptance of the trained draft head against the real Granite weights.
At each verify position we take the target's true feature f_i, roll the draft K steps (feeding its OWN
predicted feature + token forward), and count how many drafted tokens match the greedy truth. Exact
spec-decode => accepted iff == greedy. Reports tokens/pass and projected tok/s vs the measured ceiling.

    python train/eagle_accept.py --data /root/eagle_data --draft /root/eagle_data/draft.pt
"""
import argparse, os, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--data", default="/root/eagle_data")
ap.add_argument("--draft", default="/root/eagle_data/draft.pt")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--K", type=int, nargs="+", default=[4, 6, 8])
a = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()
H = torch.load(os.path.join(a.data, "heads.pt"), map_location="cpu")
dim, vocab, lsc = H["dim"], H["vocab"], H["logits_scaling"]
embed = H["embed"].to(dev); lm_head = H["lm_head"].to(dev); norm_w = H["norm"]["weight"].to(dev)

def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

class DraftHead(nn.Module):
    def __init__(self, dim, nh, inter):
        super().__init__(); self.nh = nh
        self.merge = nn.Linear(2 * dim, dim, bias=False)
        self.n1 = nn.Parameter(torch.ones(dim)); self.n2 = nn.Parameter(torch.ones(dim))
        self.q = nn.Linear(dim, dim, bias=False); self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False); self.o = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, inter, bias=False); self.up = nn.Linear(dim, inter, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)
    def forward(self, feat, emb):
        B, T, D = feat.shape
        x = self.merge(torch.cat([feat, emb], -1)); h = rmsnorm(x, self.n1)
        q = self.q(h).view(B, T, self.nh, D // self.nh).transpose(1, 2)
        k = self.k(h).view(B, T, self.nh, D // self.nh).transpose(1, 2)
        v = self.v(h).view(B, T, self.nh, D // self.nh).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(att.transpose(1, 2).reshape(B, T, D))
        h = rmsnorm(x, self.n2)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

ck = torch.load(a.draft, map_location="cpu")
draft = DraftHead(ck["dim"], ck["nh"], ck["inter"]).to(dev).to(torch.bfloat16)
draft.load_state_dict(ck["state"]); draft.eval()

def head_tok(fp):                                            # frozen head -> greedy token id
    return (F.linear(rmsnorm(fp, norm_w), lm_head) / lsc).argmax(-1)

T_M = {1: 4178e-6, 2: 5241e-6, 4: 5515e-6, 8: 6778e-6, 12: 7874e-6, 16: 9289e-6}
def t_for(M):
    for k in sorted(T_M):
        if k >= M: return T_M[k]
    return T_M[16]

PROMPTS = {
    "code":  "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "prose": "The history of the Roman Empire is a subject that has fascinated scholars for centuries.",
}

@torch.no_grad()
def roll(feats, tokens, i, K, W=64):                         # autoregressive draft WITH context window
    lo = max(0, i - W + 1)
    fseq = feats[lo:i + 1].clone()                           # f_lo..f_i  (real target features = context)
    tnext = tokens[lo + 1:i + 2].clone()                     # t_{lo+1}..t_{i+1}  (their next tokens)
    out = []
    for _ in range(K):
        emb = F.embedding(tnext, embed).to(torch.bfloat16)
        fp = draft(fseq.unsqueeze(0), emb.unsqueeze(0))[0]   # [w,dim]; last pos predicts the next feature
        f_new = fp[-1]
        t_new = head_tok(f_new.view(1, 1, -1))[0, 0]
        out.append(t_new.item())
        fseq = torch.cat([fseq, f_new.view(1, -1)], 0)       # extend: (f_new, emb(t_new)) as next input
        tnext = torch.cat([tnext, t_new.view(1)], 0)
    return out

@torch.no_grad()
def evaluate(prompt, K):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    plen = ids.shape[1]
    gen = model.generate(ids, do_sample=False, max_new_tokens=a.gen, pad_token_id=tok.eos_token_id)[0]
    feats = model(gen.unsqueeze(0), output_hidden_states=True).hidden_states[-1][0]  # [L, dim]
    L = gen.shape[0]; seq = gen.tolist()
    i, passes, advanced = plen - 1, 0, 0
    while i < L - 1:
        d = roll(feats, gen, i, K)                           # free token = seq[i+1]; draft seq[i+2..]
        passes += 1
        acc = 0
        for kk in range(len(d)):
            if i + 2 + kk < L and d[kk] == seq[i + 2 + kk]: acc += 1
            else: break
        adv = 1 + acc                                        # 1 free + accepted drafts
        advanced += adv; i += adv
    return advanced / passes

print(f"trained EAGLE draft acceptance (real Granite weights)\n")
for K in a.K:
    print(f"  K={K}:")
    for name, p in PROMPTS.items():
        tpp = evaluate(p, K)
        print(f"      {name:6s}  tokens/pass {tpp:4.2f}  -> ~{tpp/t_for(K+1):5.0f} tok/s")
print("\n(prompt-lookup was ~1.0-1.3 tokens/pass; EAGLE target is 3-5. 1500 needs tokens/pass x ceiling.)")
