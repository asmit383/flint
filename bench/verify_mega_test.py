#!/usr/bin/env python3
"""Validate + time the M=K verify megakernel. Reference = the M=1 decode megakernel run sequentially over a
greedy chain of M tokens (ground truth). verify_mega must reproduce those per-position logits in ONE launch.
Then verify_cost(M) = time(M)/time(1): if ~flat (latency-bound => extra math hides), spec-decode nets
accepted/verify_cost * (single-pass tok/s). Beats 275 when accepted/verify_cost > 1.

    python bench/verify_mega_test.py
"""
import os, sys, time, torch
sys.path.insert(0, "/root/flint")
from torch.utils.cpp_extension import load
from chat import load_packed, DIM, INTER, NH, NKV, HD, VOCAB, NL, SCALE, ROPE, RESID, LOGS, EMB_MULT, MODEL
from transformers import AutoTokenizer

dev = "cuda"; torch.set_grad_enabled(False); HERE = "/root/flint"
dec = load(name="flint_dec", sources=[HERE + "/kernels/megakernel_decode.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
ver = load(name="flint_ver", sources=[HERE + "/kernels/verify_mega.cu"], extra_cuda_cflags=["-O3", "--use_fast_math"])
W = load_packed(dev)
kc = torch.zeros(NL, 2048, NKV, HD, device=dev, dtype=torch.bfloat16)
vc = torch.zeros(NL, 2048, NKV, HD, device=dev, dtype=torch.bfloat16)

def dec1(token, pos):
    h = (W["embed"][token].float() * EMB_MULT).to(torch.bfloat16)
    return dec.decode_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
        W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, pos, SCALE, ROPE, RESID, LOGS)

def verM(tokens, pos):
    M = len(tokens)
    h = (W["embed"][torch.tensor(tokens, device=dev)].float() * EMB_MULT).to(torch.bfloat16)  # [M, DIM]
    depth = torch.arange(M, dtype=torch.int32, device=dev)                 # chain: depth = index
    amask = torch.tril(torch.ones(M, M, dtype=torch.uint8, device=dev))    # chain: causal
    return ver.verify_mega_launch(W["qkv_q"], W["qkv_s"], W["o_q"], W["o_s"], W["n1"], W["gu_q"], W["gu_s"],
        W["d_q"], W["d_s"], W["n2"], W["nf"], W["lm_q"], W["lm_s"], h, kc, vc, depth, amask, pos, M, SCALE, ROPE, RESID, LOGS)

tok = AutoTokenizer.from_pretrained(MODEL)
text = tok.apply_chat_template([{"role": "user", "content": "Write a python function to sort a list."}],
                               add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
P = len(ids)
for i, t in enumerate(ids[:-1]): dec1(t, i)                # prefill 0..P-2
lg = dec1(ids[-1], P - 1)                                  # last prompt token at P-1 -> first gen logits

# reference: greedy chain of M tokens via the M=1 megakernel (ground truth)
M = 8; cur = int(lg.argmax()); toks = []; ref = []
for m in range(M):
    toks.append(cur); l = dec1(cur, P + m); ref.append(l.clone()); cur = int(l.argmax())

# verify: same M tokens in ONE launch (overwrites cache[P..P+M-1] with identical K/V)
vlog, _vf = verM(toks, P)                                  # [M, VOCAB], [M, DIM]
print(f"prompt {P} tok | M={M} verify @ pos {P}")
print("per-position rel-L2 (verify vs sequential M=1) and argmax match:")
ok = True
for m in range(M):
    rl = (vlog[m] - ref[m]).norm() / ref[m].norm()
    am = int(vlog[m].argmax()) == int(ref[m].argmax())
    ok = ok and am and rl < 0.02
    print(f"  pos {P+m}: rel-L2 {rl:.4f}  argmax {'OK' if am else 'MISMATCH'} "
          f"(ver {int(vlog[m].argmax())} vs ref {int(ref[m].argmax())})")
print(f"  => {'PASS' if ok else 'FAIL'}")

def timeM(m, iters=30):
    t = toks[:m] if m <= len(toks) else (toks * ((m // len(toks)) + 1))[:m]
    for _ in range(5): verM(t, P)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): verM(t, P)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1000

print("\nverify_cost(M) = time(M)/time(1):")
t1 = timeM(1)
for m in (1, 2, 3, 4, 6, 8):
    tm = timeM(m); print(f"  M={m:>2}  {tm:6.2f} ms  ({tm/t1:4.2f}x)  single-pass = {1000/tm*m:5.0f} equiv tok/s if all accepted")
print(f"\nnet at acc: chain 2.03 -> {2.03/ (timeM(4)):.0f}... (net = accepted / verify_time_ms * 1000)")
for acc, mm in [(2.03, 3), (2.52, 3), (4.28, 8)]:
    print(f"  acc {acc} @ M={mm}: {acc / timeM(mm) * 1000:.0f} tok/s  (+ draft ~2-3ms not yet included)")
