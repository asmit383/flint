#!/usr/bin/env python3
"""Tree-verify acceptance (greedy). A tree of the draft's top-b candidates per node accepts the greedy path
as long as the true next token is among the draft's top-b at each step. So for greedy decoding the accepted
length is: how far along the true sequence the draft keeps the true token inside its top-b.
  branch=1 -> chain (reproduces eagle_accept ~1.77);  branch>1 -> tree.
This measures the acceptance CEILING of a tree with branching b (real trees approach it with a fixed node
budget; verify cost grows with total tree nodes, reported alongside).

    python train/tree_accept.py --data /root/eagle_gen_code --draft /root/eagle_gen_code/draft.pt
"""
import argparse, os, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_model import DraftHead, rmsnorm

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--data", default="/root/eagle_gen_code")
ap.add_argument("--draft", default="/root/eagle_gen_code/draft.pt")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--depth", type=int, default=10)             # max draft depth per pass
ap.add_argument("--branch", type=int, nargs="+", default=[1, 2, 4, 8])
a = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()
H = torch.load(os.path.join(a.data, "heads.pt"), map_location="cpu")
dim, vocab, lsc = H["dim"], H["vocab"], H["logits_scaling"]
embed = H["embed"].to(dev); lm_head = H["lm_head"].to(dev); norm_w = H["norm"]["weight"].to(dev)

ck = torch.load(a.draft, map_location="cpu")
draft = DraftHead(ck["dim"], ck["nh"], ck["inter"]).to(dev).to(torch.bfloat16)
draft.load_state_dict(ck["state"]); draft.eval()

def head_logits(fp):                                         # feature -> vocab logits (frozen head)
    return F.linear(rmsnorm(fp, norm_w), lm_head) / lsc

PROMPTS = {
    "prose":       "The history of the Roman Empire is a subject that has fascinated scholars for centuries.",
    "code_sort":   "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "code_class":  "class BankAccount:\n    def __init__(self, owner, balance=0):\n",
    "code_recur":  "def fib(n):\n    if n < 2:\n        return n\n",
    "code_io":     "def load_config(path):\n    import json\n    with open(path) as f:\n",
    "code_dict":   "def word_count(text):\n    counts = {}\n    for word in text.split():\n",
    "code_search": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n",
    "code_str":    "def is_palindrome(s):\n    s = s.lower()\n    return ",
}

W = 64
@torch.no_grad()
def roll_tree(feats, gen, i, K, b, L):                       # accepted length along the greedy path, top-b hedge
    lo = max(0, i - W + 1)
    fseq = feats[lo:i + 1].clone()
    tnext = gen[lo + 1:i + 2].clone()
    acc = 0
    for kk in range(K):
        if i + 2 + kk >= L: break
        emb = F.embedding(tnext, embed).to(torch.bfloat16)
        pos = torch.arange(lo, lo + fseq.shape[0], device=dev)
        fp = draft(fseq.unsqueeze(0), emb.unsqueeze(0), pos)[0]
        f_new = fp[-1]
        true_tok = gen[i + 2 + kk]
        topb = head_logits(f_new.view(1, -1)).topk(b, dim=-1).indices[0]
        if (topb == true_tok).any():                        # true token inside draft's top-b -> path survives
            acc += 1
            fseq = torch.cat([fseq, f_new.view(1, -1)], 0)
            tnext = torch.cat([tnext, true_tok.view(1)], 0)  # continue along the accepted (true) token
        else:
            break
    return acc

@torch.no_grad()
def evaluate(prompt, K, b):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    plen = ids.shape[1]
    gen = model.generate(ids, do_sample=False, max_new_tokens=a.gen, pad_token_id=tok.eos_token_id)[0]
    feats = model(gen.unsqueeze(0), output_hidden_states=True).hidden_states[-1][0]
    L = gen.shape[0]
    i, passes, advanced = plen - 1, 0, 0
    while i < L - 1:
        acc = roll_tree(feats, gen, i, K, b, L)
        passes += 1
        advanced += 1 + acc
        i += 1 + acc
    return advanced / passes

print("tree-verify acceptance (greedy, top-b hedge) — real Granite weights\n")
codes = [k for k in PROMPTS if k.startswith("code")]
for b in a.branch:
    print(f"  branch={b}:")
    tot = 0.0
    for name, p in PROMPTS.items():
        tpp = evaluate(p, a.depth, b)
        tot += tpp if name.startswith("code") else 0
        print(f"      {name:12s}  tokens/pass {tpp:4.2f}")
    print(f"      {'CODE avg':12s}  tokens/pass {tot/len(codes):4.2f}\n")
print("branch=1 is the chain baseline; branch>1 is tree (acceptance ceiling — verify cost grows with tree nodes).")
