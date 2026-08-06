#!/usr/bin/env python3
"""Real MTP acceptance rate for a drafter, against the REAL Granite weights. Exact greedy speculative
decoding accepts a drafted token iff it equals the model's greedy next token — so we generate the true
greedy continuation ONCE, then simulate the drafter offline against it (no cache-rollback needed).

Starts with prompt-lookup (n-gram) drafting: zero training, works today. Reports avg tokens advanced
per verify pass, then projects realized tok/s using the measured verify-cost ceiling from mtp_scaling.

    python bench/spec_accept.py --model ibm-granite/granite-4.1-3b
"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--ngram", type=int, nargs="+", default=[2, 3])
ap.add_argument("--K", type=int, nargs="+", default=[4, 8, 12, 16])
a = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()

# verify-pass cost ceiling (from mtp_scaling.py, gpt-fast int4, H100 PCIe) -> project realized tok/s.
# T_M in seconds for M candidate tokens; tok/s = tokens_advanced_per_pass / T_M.
T_M = {1: 4178e-6, 2: 5241e-6, 4: 5515e-6, 8: 6778e-6, 12: 7874e-6, 16: 9289e-6}
def t_for(M):  # nearest measured M >= needed
    for k in sorted(T_M):
        if k >= M: return T_M[k]
    return T_M[16]

PROMPTS = {
    "code":   "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "repeat": "The report states the following items must be reviewed: item 1, item 2, item 3,",
    "prose":  "The history of the Roman Empire is a subject that has fascinated scholars for centuries.",
}

@torch.no_grad()
def greedy(prompt, n):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, do_sample=False, max_new_tokens=n,
                         pad_token_id=tok.eos_token_id)
    return out[0].tolist()

def draft(seq, i, ngram, K):
    """prompt-lookup: most recent match of the ngram ending at i-1, return the K tokens that followed it."""
    if i < ngram: return []
    ng = seq[i - ngram:i]
    for j in range(i - ngram - 1, ngram - 2, -1):
        if seq[j:j + ngram] == ng:
            return seq[j + ngram: j + ngram + K]
    return []

def simulate(seq, plen, ngram, K):
    i, passes, advanced, drafted, accepted = plen, 0, 0, 0, 0
    while i < len(seq):
        d = draft(seq, i, ngram, K)
        passes += 1
        if d:
            a = 0
            for k in range(len(d)):
                if i + k < len(seq) and d[k] == seq[i + k]: a += 1
                else: break
            drafted += len(d); accepted += a
            adv = a + 1
        else:
            adv = 1
        advanced += adv; i += adv
    return advanced / passes, (accepted / drafted if drafted else 0.0), passes

print(f"model={a.model}  (real weights)  prompt-lookup speculative acceptance\n")
seqs = {name: greedy(p, a.gen) for name, p in PROMPTS.items()}
for ngram in a.ngram:
    for K in a.K:
        print(f"  ngram={ngram} K={K:2d}:")
        for name, seq in seqs.items():
            plen = len(tok(PROMPTS[name]).input_ids)
            tpp, acc, passes = simulate(seq, plen, ngram, K)
            realized = tpp / t_for(K + 1)
            print(f"      {name:7s}  tokens/pass {tpp:4.2f}  accept {100*acc:4.1f}%  -> ~{realized:5.0f} tok/s")
print("\n(realized = tokens/pass ÷ verify-cost at M=K+1; 1500 target. prose = worst case, code = best.)")
