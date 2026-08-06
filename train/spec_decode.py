#!/usr/bin/env python3
"""Wire it up: REAL speculative decoding (draft -> verify -> accept, with KV cache) vs plain greedy, on
the real Granite weights + trained EAGLE draft. Measures wall-clock tok/s of both and checks the spec
output is byte-identical to greedy (exact spec-dec => no quality loss). Eager transformers, so absolute
numbers are low; the RATIO (spec/plain) is the honest 'does the draft help yet' answer.

    python train/spec_decode.py --draft /root/eagle_data/draft.pt --new 200 --K 5
"""
import argparse, os, sys, time, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_model import DraftHead, rmsnorm

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
ap.add_argument("--data", default="/root/eagle_data")
ap.add_argument("--draft", default="/root/eagle_data/draft.pt")
ap.add_argument("--new", type=int, default=200)
ap.add_argument("--K", type=int, default=5)
a = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             device_map=dev, trust_remote_code=True).eval()
H = torch.load(os.path.join(a.data, "heads.pt"), map_location="cpu")
embed = H["embed"].to(dev); lm_head = H["lm_head"].to(dev); norm_w = H["norm"]["weight"].to(dev)
lsc = H["logits_scaling"]
ck = torch.load(a.draft, map_location="cpu")
draft = DraftHead(ck["dim"], ck["nh"], ck["inter"]).to(dev).to(torch.bfloat16)
draft.load_state_dict(ck["state"]); draft.eval()

def head_tok(fp):
    return (F.linear(rmsnorm(fp, norm_w), lm_head) / lsc).argmax(-1)

@torch.no_grad()
def draft_roll(ctx_feats, ctx_positions, first_tok, K):
    """roll K draft tokens; ctx_feats [W,dim] real feature context, first_tok = the free next token."""
    fseq = ctx_feats.clone(); pos = ctx_positions.clone(); tnext = first_tok.view(1)
    tnext = torch.cat([torch.zeros(fseq.shape[0] - 1, dtype=torch.long, device=dev), tnext])  # only last matters
    out = []
    for _ in range(K):
        emb = F.embedding(tnext, embed).to(torch.bfloat16)
        fp = draft(fseq.unsqueeze(0), emb.unsqueeze(0), pos)[0]
        f_new = fp[-1]; t_new = head_tok(f_new.view(1, 1, -1))[0, 0]
        out.append(t_new)
        fseq = torch.cat([fseq, f_new.view(1, -1)]); pos = torch.cat([pos, pos[-1:] + 1])
        tnext = torch.cat([tnext, t_new.view(1)])
    return torch.stack(out)

@torch.no_grad()
def plain(prompt, n):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(ids, do_sample=False, max_new_tokens=n, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new = out[0, ids.shape[1]:]
    return new.tolist(), n / dt

@torch.no_grad()
def spec(prompt, n, K, W=32):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    cache = DynamicCache()
    o = model(ids, use_cache=True, past_key_values=cache, output_hidden_states=True)
    feats = o.hidden_states[-1][0]                          # [L,dim]
    pos = ids.shape[1]
    nxt = o.logits[0, -1].argmax().view(1)                 # free token
    gen = [nxt.item()]
    torch.cuda.synchronize(); t0 = time.perf_counter(); passes = 0
    while len(gen) < n:
        lo = max(0, pos - W)
        cpos = torch.arange(lo, pos, device=dev)
        d = draft_roll(feats[lo:pos], cpos, nxt, K)        # K candidates
        vin = torch.cat([nxt, d]).view(1, -1)              # [1, K+1]
        vo = model(vin, past_key_values=cache, use_cache=True, output_hidden_states=True)
        tgt = vo.logits[0].argmax(-1)                      # [K+1] target next-token per position
        acc = 0
        for j in range(K):
            if tgt[j].item() == d[j].item(): acc += 1
            else: break
        corr = tgt[acc]                                    # target's true token at first mismatch
        gen += d[:acc].tolist() + [corr.item()]
        cache.crop(pos + 1 + acc)                          # keep free tok + acc drafts
        newf = vo.hidden_states[-1][0]                     # [K+1,dim]
        feats = torch.cat([feats, newf[:acc + 1]])         # append accepted features
        pos += 1 + acc; nxt = corr.view(1); passes += 1
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return gen[:n], n / dt, (n / passes)

PROMPT = "The history of the Roman Empire is a subject that has fascinated scholars for centuries."
print(f"model={a.model}  draft K={a.K}  new={a.new}\n")
g_ids, g_tps = plain(PROMPT, a.new)
s_ids, s_tps, tpp = spec(PROMPT, a.new, a.K)
match = sum(x == y for x, y in zip(g_ids, s_ids)) / len(g_ids)
print(f"  plain greedy  : {g_tps:6.1f} tok/s")
print(f"  spec-decode   : {s_tps:6.1f} tok/s   ({tpp:.2f} tokens/pass, {a.K} drafted)")
print(f"  speedup       : {s_tps/g_tps:.2f}x")
print(f"  output match  : {100*match:.1f}%  (exact spec-dec should be 100%)")
