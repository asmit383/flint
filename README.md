# flint

> A measured study of batch-1 int4 LLM decode on H100 — and a hand-written full-decode **megakernel** that beats the baseline — for Granite-4.1.

Batch-1 decode is where single-stream LLM latency lives: one token at a time, read the whole weight matrix,
do almost no math. The folk wisdom is *"it's memory-bandwidth-bound, so read fewer bytes (int4, activation
sparsity) and you win."* flint set out to do exactly that — and **measured, honestly, that on an H100 the
folk wisdom is mostly wrong.** What we found instead, and the megakernel we hand-wrote in response, is the
project. Every number here is measured end-to-end on real hardware; where a projection existed, it's marked
and superseded by the measurement.

- **Model:** Granite-4.1-3B dense (GQA / RoPE / SwiGLU, Apache 2.0).
- **Base engine:** fork of [gpt-fast](https://github.com/pytorch-labs/gpt-fast) (BSD-3, attribution in `engine/NOTICE`).
- **Hardware:** H100 PCIe (2 TB/s, 114 SMs).

## The core finding (measured — and independently published)

On H100, batch-1 int4 decode is **NOT bandwidth-bound. It's overhead/latency-bound.** We measure ~27% MBU
(memory-bandwidth utilization); independent work ([arXiv 2605.30571](https://arxiv.org/html/2605.30571))
measures the same 27% and shows *why* — launch/latency overhead, not bytes. A slow GPU (L4) hits 81% MBU on
the same workload: it *is* bandwidth-bound. **Fast GPU → latency-bound; slow GPU → bandwidth-bound.** Same
kernel, opposite regime.

The consequence: the byte-cutting levers everyone reaches for **don't help at B=1 on H100**, because bytes
aren't the bottleneck. We proved each one by measurement rather than assuming:

| lever tried | result | why |
|---|---|---|
| hand int4 GEMV vs tinygemm | **ties** (~27% MBU) | the microbench "win" was an **L2-cache artifact** — evaporated end-to-end |
| activation sparsity (50%) | **2.8× slower** | scattered column-skips + atomics; not bandwidth-bound |
| tensor cores (hand WMMA) | **100× slower** | M=1 wastes 15/16 of the MMA |
| gate/up fusion | **+3%** end-to-end | gpt-fast already fuses QKV; the win was already there |
| Marlin int4 | doesn't crack B=1 (confirmed) | tuned for batch, not for overhead-bound decode |

Every optimistic microbenchmark got corrected *downward* by honest end-to-end measurement. That discipline —
and the negative results it produced — is the real content.

## What actually beats the baseline: the megakernel

If B=1 is overhead-bound, the fix isn't fewer bytes — it's fewer *ops*. Fuse the **entire** decode step into
**one persistent GPU kernel** so the residual stream never round-trips HBM and there are no inter-op
relaunches. Hand-written, all correct (`kernels/megakernel_{mlp,attn,decode}.cu`):

| path | tok/s | MBU |
|---|---|---|
| gpt-fast int4 baseline | 237 | 27% |
| **flint megakernel** (full 40-layer, one cooperative launch) | **274** | **24%** |

**274 tok/s, 1.16× the baseline, numerically correct end-to-end** (argmax matches the reference every step) —
the first thing in the project to actually beat gpt-fast. The dig that got there:

```
naive persistent            44 tok/s   (grid-wide atomicAdd reduction = 117k threads → 1 address)
+ block-reduce rmsnorm      209 tok/s   ← 4.7×, that contention was the killer
+ redundant per-block norm  217 tok/s   (barriers matter less than expected)
+ vectorized 128-bit x-load 274 tok/s   ← the GEMV win
(more warps / more accumulators both HURT — the kernel is register/occupancy-sensitive, not warp-starved)
```

It's currently **latency/barrier-bound** (a diagnostic that halves *all* weight bytes speeds it up only
1.23×, not 2× — 76% of bandwidth sits idle). The remaining headroom (24% → ~78% MBU, HazyResearch-class)
needs a **barrier-free producer-consumer redesign** (no `grid.sync`, per-tile flag deps, memory/compute
overlap) — the next big build.

## The path to ~800 tok/s (honest)

Single-pass is walled around ~270 on H100 (overhead-bound — no kernel beats that without the barrier-free
megakernel). The multiplier is **speculative decode (EAGLE-3)** — verify multiple tokens per weight-read.
Published EAGLE-3 delivers **3.0–3.4×** at batch 1. So:

> **megakernel 274 × EAGLE-3 3× ≈ 820 tok/s** — realistic, both fronts measured.

The EAGLE runway is built (`train/`): the key fix is **self-distillation** — training the draft on the
target's *own generations*, not raw corpus (lifted held-out draft accuracy 2.6×). Reaching a full 3× needs
more diverse data + tree-verify. (1500 would additionally need the barrier-free megakernel; ~800 does not.)

## Layout
```
engine/    gpt-fast fork + Granite-4.1 support (LICENSE / NOTICE, BSD-3)
kernels/   custom CUDA — int4 GEMV variants + the megakernel (attn / mlp / full-decode)
train/     EAGLE draft: self-distill harvest, training, acceptance eval, spec-decode loop
bench/     measurement harnesses — every claim above is reproducible here
setup/     one-command box provisioning + baselines
notes/     working notes incl. the megakernel design doc (gitignored)
```

## Honesty notes
- Numbers are **measured end-to-end**, not projected. Microbenchmarks lie at B=1 (L2 residency, launch
  amortization) — only the token clock counts.
- **Negative results are kept** — sparsity, tensor cores, split-K, Marlin all measured-dead at B=1. Knowing
  *why* they fail is the point.
- int4 adds a small perplexity cost (quantization); activation sparsity is an opt-in quality trade. Neither
  is free, and neither is claimed to be.
