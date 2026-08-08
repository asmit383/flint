#!/usr/bin/env python3
"""Tier-1 spec-decode chat: compiled bf16 gpt-fast Granite (the ~6ms verify) + EAGLE draft, live streaming.
Draft proposes K tokens from the target's own features; target verifies K+1 in ONE compiled forward; accept
the longest greedy-matching prefix; advance. Exact spec-decode => byte-identical to greedy, but faster.

    python spec_chat.py --draft /root/eagle_ll8k/draft.pt --K 6
    python spec_chat.py --selftest --draft /root/eagle_ll8k/draft.pt
"""
import os, sys, time, argparse, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
sys.path.insert(0, os.path.join(HERE, "train"))
from model import Transformer
from draft_model import DraftHead, rmsnorm
from transformers import AutoTokenizer

MODEL = "ibm-granite/granite-4.1-3b"; CKPT = os.path.join(HERE, "engine/checkpoints/granite-4.1-3b/model.pth")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="/root/eagle_ll8k/draft.pt")
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--W", type=int, default=32)  # K=3 = chain net optimum (acc saturates ~2, draft scales with K)
    ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--maxseq", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    dev = "cuda"; torch.set_grad_enabled(False)

    print("loading gpt-fast Granite bf16 ...", flush=True)
    model = Transformer.from_name("granite-4.1-3b")
    sd = torch.load(CKPT, weights_only=True)
    model.load_state_dict(sd, strict=False, assign=True)
    model = model.to(torch.bfloat16).to(dev).eval()
    with torch.device(dev): model.setup_caches(1, a.maxseq)
    embed = model.tok_embeddings.weight; lm_head = model.output.weight
    norm_w = model.norm.weight; lsc = model.config.logits_scaling
    tok = AutoTokenizer.from_pretrained(MODEL)

    ck = torch.load(a.draft, map_location="cpu")
    draft = DraftHead(ck["dim"], ck["nh"], ck["inter"], fuse=ck.get("fuse", 1)).to(dev).to(torch.bfloat16)
    draft.load_state_dict(ck["state"]); draft.eval()

    def head_tok(fp):                                          # feature -> greedy token (model's frozen head)
        return (F.linear(rmsnorm(fp, norm_w), lm_head) / lsc).argmax(-1)

    verify = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    draft.setup_cache(a.W + a.K + 2, dev)                     # fixed KV cache -> the rollout step CUDA-graphs

    def _step(f1, emb1, pos1, slot):                          # one graphed rollout step (fixed [1,1,*] shapes)
        fp = draft(f1, emb1, pos1, input_pos=slot)
        f_new = fp[:, -1]
        return f_new, head_tok(f_new)[0]
    _step_c = torch.compile(_step, mode="reduce-overhead", fullgraph=True)

    def draft_roll(feats, positions, first_tok, K):           # fixed-cache rollout: prime context, then graphed steps
        lo = max(0, feats.shape[0] - a.W)
        fctx = feats[lo:]; pctx = positions[lo:]; Wc = fctx.shape[0]
        tctx = torch.zeros(Wc, dtype=torch.long, device=dev); tctx[-1] = first_tok
        emb_ctx = F.embedding(tctx, embed).to(torch.bfloat16)
        fp = draft(fctx.unsqueeze(0), emb_ctx.unsqueeze(0), pctx, input_pos=torch.arange(Wc, device=dev))  # prime
        f_new = fp[:, -1]; t_new = head_tok(f_new)[0]
        out = [t_new]; cpos = pctx[-1:] + 1; slot = torch.tensor([Wc], device=dev)
        for _ in range(K - 1):
            emb1 = F.embedding(t_new.view(1), embed).to(torch.bfloat16).view(1, 1, -1)
            torch.compiler.cudagraph_mark_step_begin()
            f_new, t_new = _step_c(f_new.view(1, 1, -1), emb1, cpos, slot)
            f_new = f_new.clone(); t_new = t_new.clone()      # copy graph outputs before the next replay reuses them
            out.append(t_new); cpos = cpos + 1; slot = slot + 1
        return torch.stack(out)

    def spec_generate(ids, cb):                               # streaming spec-decode; cb(token) per accepted token
        ids = ids.view(1, -1); P = ids.shape[1]
        ipos = torch.arange(P, device=dev)
        logits, h = model(ids, ipos, return_hidden=True)      # prefill (uncompiled, once)
        feats = h[0].clone()                                  # [P, dim] target features
        pos = P; nxt = logits[0, -1].argmax().view(1)
        draft_roll(feats, torch.arange(pos, device=dev), nxt, a.K)   # warmup: compile the draft step BEFORE timing
        torch.compiler.cudagraph_mark_step_begin()            # warmup: compile the verify BEFORE timing
        verify(torch.zeros(1, a.K + 1, dtype=torch.long, device=dev), torch.arange(pos, pos + a.K + 1, device=dev), return_hidden=True)
        torch.cuda.synchronize()
        n = 0; passes = 0; t0 = time.perf_counter(); t_draft = 0.0; t_ver = 0.0
        while n < a.max_new and pos < a.maxseq - a.K - 2:
            cb(nxt.item()); n += 1
            if nxt.item() == tok.eos_token_id: break
            torch.cuda.synchronize(); _td = time.perf_counter()
            d = draft_roll(feats, torch.arange(pos, device=dev), nxt, a.K)   # K candidates
            torch.cuda.synchronize(); t_draft += time.perf_counter() - _td; _tv = time.perf_counter()
            vin = torch.cat([nxt, d]).view(1, -1)             # [1, K+1]
            vpos = torch.arange(pos, pos + a.K + 1, device=dev)
            torch.compiler.cudagraph_mark_step_begin()
            vlog, vh = verify(vin, vpos, return_hidden=True)
            vlog = vlog.clone(); vh = vh[0].clone()
            torch.cuda.synchronize(); t_ver += time.perf_counter() - _tv
            tgt = vlog[0].argmax(-1)                          # [K+1] target greedy per position
            acc = 0
            for j in range(a.K):
                if tgt[j].item() == d[j].item(): acc += 1
                else: break
            for j in range(acc): cb(d[j].item()); n += 1      # stream accepted drafts
            feats = torch.cat([feats, vh[:acc + 1]])          # append accepted features (+ the free-next position)
            pos += 1 + acc; nxt = tgt[acc].view(1); passes += 1
        dt = time.perf_counter() - t0
        print(f"\n\033[2m[profile] draft_roll {t_draft/passes*1000:.1f}ms/pass | verify {t_ver/passes*1000:.1f}ms/pass | passes {passes}\033[0m")
        return n, dt, (n / passes if passes else 1)

    def run(prompt_ids):
        buf = {"ids": [], "prev": ""}
        def cb(t):
            buf["ids"].append(t); s = tok.decode(buf["ids"])
            sys.stdout.write(s[len(buf["prev"]):]); sys.stdout.flush(); buf["prev"] = s
        n, dt, tpp = spec_generate(torch.tensor(prompt_ids, device=dev), cb)
        print(f"\n\033[2m⚡ {n} tokens · {n/max(dt,1e-9):.1f} tok/s · {tpp:.2f} tokens/pass (spec-decode bf16)\033[0m")

    if a.selftest:
        text = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that reverses a linked list."}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        print(f"\n[selftest] compiling + generating ...\n", flush=True); run(ids); return

    print("\nflint spec-decode chat (bf16) — Ctrl-C to quit. First reply compiles (~1 min).\n")
    msgs = []
    while True:
        try: user = input("\033[1myou›\033[0m ")
        except (EOFError, KeyboardInterrupt): print(); break
        if not user.strip(): continue
        msgs.append({"role": "user", "content": user})
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        sys.stdout.write("\033[1mflint›\033[0m ")
        pre = []
        def cb(t, pre=pre):
            pre.append(t); s = tok.decode(pre); sys.stdout.write(s[len(cb.prev):]); sys.stdout.flush(); cb.prev = s
        cb.prev = ""
        n, dt, tpp = spec_generate(torch.tensor(ids, device=dev), cb)
        print(f"\n\033[2m⚡ {n} tok · {n/max(dt,1e-9):.1f} tok/s · {tpp:.2f} tok/pass\033[0m\n")
        msgs.append({"role": "assistant", "content": tok.decode(pre)})

if __name__ == "__main__":
    main()
