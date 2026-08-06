#!/usr/bin/env python3
"""chat.py — interactive chat on the flint megakernel: our from-scratch int4 decode (the ~284 tok/s path),
full 40-layer Granite-4.1-3B forward fused into ONE persistent cooperative CUDA launch. Real weights, our
own contiguous int4 packing. QKV is packed RAW (unpermuted) because the megakernel's RoPE is HF/NEOX
half-split — NOT gpt-fast's interleaved convention. Type to chat; watch the tokens fly.

    python chat.py                 # greedy, feel the speed
    python chat.py --selftest      # one fixed prompt, check coherence + tok/s, exit
    python chat.py --temp 0.7 --topk 40
"""
import os, sys, time, argparse, torch
from glob import glob
from torch.utils.cpp_extension import load
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
DIM, INTER, NH, NKV, HD, VOCAB, NL = 2560, 8192, 40, 8, 64, 100352, 40
SCALE, ROPE, RESID, LOGS, EMB_MULT = 0.015625, 1e7, 0.22, 10.0, 12.0   # Granite-4.1 scalar multipliers
MODEL = "ibm-granite/granite-4.1-3b"

def qpc(W, G=128):                        # our int4 pack: q[OUT,IN/8] int32 (8 nibbles/word) + s[OUT,IN/G] bf16
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    p = (qu << (torch.arange(8, device=W.device) * 4).view(1, 1, 8)).sum(-1)
    qi = torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
    return qi, s.squeeze(-1).to(torch.bfloat16).contiguous()

def load_packed(dev):
    snap = snapshot_download(MODEL)
    sd = {}
    for f in glob(os.path.join(snap, "*.safetensors")):
        sd.update(load_file(f, device=dev))
    def stackpack(mk):                    # pack per-layer weight, stack to [NL, OUT, IN/8] + [NL, OUT, IN/G]
        qs = [qpc(mk(l)) for l in range(NL)]
        return torch.stack([a[0] for a in qs]), torch.stack([a[1] for a in qs])
    P = lambda l: f"model.layers.{l}."
    W = {}
    W["qkv_q"], W["qkv_s"] = stackpack(lambda l: torch.cat([sd[P(l)+"self_attn.q_proj.weight"],   # RAW, no permute
        sd[P(l)+"self_attn.k_proj.weight"], sd[P(l)+"self_attn.v_proj.weight"]], 0))
    W["o_q"], W["o_s"]   = stackpack(lambda l: sd[P(l)+"self_attn.o_proj.weight"])
    W["gu_q"], W["gu_s"] = stackpack(lambda l: torch.cat([sd[P(l)+"mlp.gate_proj.weight"], sd[P(l)+"mlp.up_proj.weight"]], 0))
    W["d_q"], W["d_s"]   = stackpack(lambda l: sd[P(l)+"mlp.down_proj.weight"])
    W["n1"] = torch.stack([sd[P(l)+"input_layernorm.weight"] for l in range(NL)]).contiguous()
    W["n2"] = torch.stack([sd[P(l)+"post_attention_layernorm.weight"] for l in range(NL)]).contiguous()
    W["nf"] = sd["model.norm.weight"].contiguous()
    W["embed"] = sd["model.embed_tokens.weight"].contiguous()          # bf16, for the embedding lookup
    W["lm_q"], W["lm_s"] = qpc(W["embed"])                             # tied -> LM head is the embedding, int4
    return W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512); ap.add_argument("--maxseq", type=int, default=4096)
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    dev = "cuda"; torch.set_grad_enabled(False)

    print("compiling megakernel ...", flush=True)
    m = load(name="flint_chat", sources=[os.path.join(HERE, "kernels/megakernel_decode.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    print("loading + int4-packing Granite-4.1-3B ...", flush=True)
    W = load_packed(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id
    kc = torch.zeros(NL, a.maxseq, NKV, HD, device=dev, dtype=torch.bfloat16)
    vc = torch.zeros(NL, a.maxseq, NKV, HD, device=dev, dtype=torch.bfloat16)

    def logits_at(token, pos):            # embed*mult -> full 40-layer megakernel -> logits[VOCAB]
        h = (W["embed"][token].float() * EMB_MULT).to(torch.bfloat16)
        return m.decode_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
            W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, pos, SCALE, ROPE, RESID, LOGS)

    def pick(logits):
        if a.temp <= 0: return int(logits.argmax())
        lg = logits / a.temp
        if a.topk > 0:
            v, _ = torch.topk(lg, a.topk); lg[lg < v[-1]] = -float("inf")
        p = torch.softmax(lg, -1); return int(torch.multinomial(p, 1))

    def respond(prompt_ids):
        pos = 0
        for t in prompt_ids[:-1]:         # prefill all but last (fills KV cache)
            logits_at(t, pos); pos += 1
        logits = logits_at(prompt_ids[-1], pos); pos += 1
        nxt = pick(logits); gen = []; prev = ""
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(gen) < a.max_new and nxt != eos:
            gen.append(nxt)
            text = tok.decode(gen)
            sys.stdout.write(text[len(prev):]); sys.stdout.flush(); prev = text
            logits = logits_at(nxt, pos); pos += 1; nxt = pick(logits)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        print(f"\n\033[2m⚡ {len(gen)} tokens · {len(gen)/dt:.1f} tok/s (megakernel int4, decode only)\033[0m")

    if a.selftest:
        text = tok.apply_chat_template([{"role": "user", "content": "In two sentences, what is a GPU and why is it fast?"}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        print(f"\n[selftest] prompt {len(ids)} tok\n", flush=True); respond(ids); return

    print("\nflint chat — Granite-4.1-3B int4 on the megakernel. Ctrl-C to quit.\n")
    msgs = []
    while True:
        try: user = input("\033[1myou›\033[0m ")
        except (EOFError, KeyboardInterrupt): print(); break
        if not user.strip(): continue
        msgs.append({"role": "user", "content": user})
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if len(ids) >= a.maxseq - a.max_new:
            print("\033[2m(context full — restarting conversation)\033[0m"); msgs = msgs[-1:]
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        sys.stdout.write("\033[1mflint›\033[0m ")
        # capture the response text to keep chat history coherent
        reply_ids = []
        pos = 0
        for t in ids[:-1]: logits_at(t, pos); pos += 1
        logits = logits_at(ids[-1], pos); pos += 1
        nxt = pick(logits); prev = ""
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(reply_ids) < a.max_new and nxt != eos and pos < a.maxseq:
            reply_ids.append(nxt); s = tok.decode(reply_ids)
            sys.stdout.write(s[len(prev):]); sys.stdout.flush(); prev = s
            logits = logits_at(nxt, pos); pos += 1; nxt = pick(logits)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        print(f"\n\033[2m⚡ {len(reply_ids)} tokens · {len(reply_ids)/max(dt,1e-9):.1f} tok/s\033[0m\n")
        msgs.append({"role": "assistant", "content": tok.decode(reply_ids)})

if __name__ == "__main__":
    main()
