# flint

> Batch-1 LLM decode made faster by reading fewer bytes — int4 + dynamic activation sparsity.

Batch-1 LLM decode is **memory-bandwidth-bound** — you read the weights and do almost no math. So the
way to go faster isn't a faster kernel (production kernels are already good); it's **reading fewer
bytes**. This project does that on a dense model via **int4 weights + dynamic activation sparsity**
(skip the weight rows whose activation is ~0) — a byte-cutting, opt-in quality trade that engines like
vLLM don't ship — driven from a tight, self-owned CUDA-graph decode loop.

- **Model:** Granite-4.1 dense (standard GQA/RoPE/SwiGLU transformer, Apache 2.0). Prototype 3B, then 8B.
- **Base engine:** fork of [gpt-fast](https://github.com/pytorch-labs/gpt-fast) (minimal B=1 decode with
  int4 + CUDA graphs + speculative decode).
- **Hardware:** H100 PCIe (2 TB/s, Hopper — for TMA / warp-specialization). MBU % is peak-independent,
  so it also holds on cheaper cards; absolute tok/s scales with bandwidth.

## Baselines — measured (Granite-3B, B=1, H100 PCIe 2 TB/s)

| path | tok/s | MBU | bytes/token |
|---|---|---|---|
| transformers eager | 5.6 | 1.9% | 6.81 GB |
| gpt-fast bf16 (compiled) | 176.6 | **64.6%** | 7.32 GB |
| **gpt-fast int4** (tinygemm) | **233.8** | **27.4%** | 2.34 GB |

Two facts these pin down:
- **bf16 hits ~65% MBU** → that's the achievable ceiling at B=1 on this box; it's real, not a fantasy.
- **int4 sits at only ~27% MBU** — 1.3× faster than bf16 despite 3× fewer bytes, because it's
  **dequant-bound** (the tinygemm int4 kernel leaves ~37 points of MBU on the floor to on-the-fly unpack).

**That gap is the whole thesis.** flint's job: close int4's dequant gap (multi-accumulator dequant, TMA +
warp specialization, split-K) **+ read fewer bytes** (dynamic activation sparsity — Granite-4.1 tolerates
~50% @ +2% ppl, measured) **+ MTP**. Target: **~1500 tok/s** (int4 @ ~45% MBU × ~40% sparsity ×
speculative decode), stated *with* the ppl cost, never hidden. The baseline to beat is **233.8 tok/s.**

## Layout
```
engine/   gpt-fast fork + Granite-4.1 support (+ its LICENSE / NOTICE, BSD-3)
kernels/  custom CUDA — sparse-int4 GEMV, fused rmsnorm  (the work)
bench/    measurement harnesses (sparsity gate, baselines) — source of truth
setup/    one-command box provisioning + baselines
notes/, kdoc/   working notes (gitignored)
```

## Order of operations
1. `setup/setup_box.sh` — deps + model + gpt-fast. Then `setup/baselines.sh <peak-bw>`.
2. `bench/sparsity_sweep.py` — **GO/NO-GO gate** ✅ *passed* (Granite-4.1 tolerates ~50% @ +2% ppl).
3. gpt-fast baselines ✅ *measured* (above).
4. **← we are here:** write the sparse-int4 GEMV, drive it in a self-owned CUDA-graph loop, and climb
   int4 MBU from 27% toward ~45–65% — every technique A/B'd against `ncu`. Then activation sparsity, then MTP.
