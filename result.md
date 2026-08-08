# Results — measured

Every number here is measured end-to-end on H100 PCIe (2 TB/s, 114 SMs), Granite-4.1-3B, int4. Negative
results kept. Single-pass baseline is gpt-fast int4 (237 tok/s).

## 1. Single-pass decode megakernel

The whole 40-layer decode fused into one persistent cooperative launch (residual stays L2-resident, no
inter-op relaunches).

| path | tok/s | MBU |
|---|---|---|
| gpt-fast int4 baseline | 237 | 27% |
| **flint megakernel** (real Granite weights, coherent) | **275** | 25% |

Correct (argmax matches reference every step). The dig: 44 → 209 (block-reduce rmsnorm; a grid-wide
atomicAdd was serializing 117k threads) → 217 → 274 (128-bit vector activation load) → ~284 (flash-decode
multi-warp attention). More warps / multi-accumulator both HURT — the kernel is register-limited (50%
occupancy), not warp-starved.

## 2. Where the time goes (latency decomposition)

Two diagnostics bracket it:
- **NOSYNC** (barriers compiled out) → 404 tok/s ⇒ barriers ≈ 30%. But not the primitive: moving the
  balanced rmsnorm barriers to shared memory gained 0% — the cost is **load-imbalance at post-GEMV syncs**,
  which the reduction dependencies force.
- **HALFBYTES** (read half the weight rows) → 1.23× ⇒ weight-load latency ≈ ⅓, and it's *latency* not
  bandwidth (MBU 25% = pipe a quarter full; register-capped at 50% occupancy, cp.async only bought +1%).

So: launch latency killed by the megakernel; what remains is ~⅓ memory latency, ~⅓ barrier imbalance,
~⅓ compute. Batch-1 on a fast GPU is **latency-bound, not bandwidth-bound** (an L4 runs the same kernel at
81% MBU = bandwidth-bound; opposite regime).

## 3. Hand-rolled grid barrier vs cooperative groups

| barrier | ns/barrier (isolation) | in the megakernel |
|---|---|---|
| `cg::grid.sync()` | 2308 | **287 tok/s** |
| hand-rolled C++ (atomics+fence) | 1958 (**0.85×**) | 262 tok/s (0.91×) |
| hand-rolled inline PTX | 1957 (**0.85×**) | — |

Hand-rolling is practical (**15% faster in isolation**), and **inline PTX ≡ C++ intrinsics** (identical to
the nanosecond — `atomicInc`/`__threadfence` already compile to `atom.inc`/`membar`). But **in the real
kernel it's 9% SLOWER** — the `membar.gl` fence is cheap with no traffic but expensive fencing in-flight
GEMV writes. Microbench didn't transfer; only the token clock counts. (C++ is also the portable form —
`hipify` translates it to CDNA; inline PTX does not.)

## 4. Single-pass ceiling — measured dead ends

~275 is the single-pass B=1 ceiling. Every lever measured:

| lever | result | why |
|---|---|---|
| hand int4 GEMV vs tinygemm | ties (~27% MBU) | the microbench "win" was an L2-cache artifact |
| activation sparsity (50%) | 2.8× slower | scatter+atomics floor; not bandwidth-bound |
| tensor cores (WMMA) at M=1 | 100× slower | M=1 fills 1 of 16 MMA rows |
| cp.async pipeline | +1% | latency-bound, not load-parallelism-bound |
| producer-consumer (barrier-free) | 1.34× slower | splitting warps starves phases; reduction dep = no overlap; the barrier is cheaper than removing it |
| barrier-count reduction (shared-xn) | 0% | removable barriers are cheap; expensive ones are dependency-forced |

## 5. Speculative decode — the multiplier (coding regime)

Self-distillation drafter: seed Granite with Python-code prefixes, harvest **its own** greedy continuations
+ hidden features, train an EAGLE draft to mimic them. Specializing the regime is the acceptance lever.

**Acceptance (tokens/pass), held-out code prompts, 8k self-distill data:**

| verify | single-step train | **multi-step (rollout) train** |
|---|---|---|
| general (wiki/prose) | ~1.1 | — |
| **chain** (code avg) | 1.76 | **2.52** |
| tree branch=4 | 2.27 | 3.67 |
| **tree branch=8** | 2.46 | **4.28** |

**The acceptance breakthrough — exposure-bias fix.** The draft trained *single-step* (teacher-forced on the
target's TRUE features) but ran *multi-step* at inference (autoregress on its OWN predicted features) — a
textbook EAGLE-2 gap: errors compound and chains broke ~token 3. Retraining with a **K-step rollout** (feed
the draft's own prediction forward at each depth, so it learns to recover from its own error) closed it:
held-out code chain **1.76 → 2.52**, tree **2.46 → 4.28**. This removes acceptance as the wall.

## 6. Wiring the multiplier — the verify-cost de-risk

Naive projection is `acceptance × 275`. It's **wrong** — the M=K verify pass (verifying K tokens in one
forward, weights read once) costs more than M=1. Measured `verify_cost(M) = time(M)/time(1)`, L2-cold,
scalar M=K int4 GEMV:

| M | verify_cost | net tok/s = 275 × accepted / verify_cost (acc ~1.8) |
|---|---|---|
| 1 | 1.00 | 275 |
| 2 | 1.45 | 266 |
| 4 | 1.86 | 251 |
| 8 | 2.66 | 194 |

**At current acceptance, spec-decode LOSES** — every M below 275, because `accepted/verify_cost < 1`. The
scalar M=K GEMV is register/compute-bound. Measuring this *before* building the full M=K megakernel avoided
shipping a slower chat.

**Fix — tensor-core M=K GEMM:** the "tensor cores waste 15/16 at M=1" reverses at M≥4 (fill 8/16 at M=8), so
a tensor-core verify could drop `verify_cost(4)` from 1.86 → ~1.2.

## 7. The path to 800

```
net tok/s = accepted / (draft_roll_time + verify_time)
```
With acceptance solved (chain 2.52 / tree 4.28), the projection with a well-wired draft + verify:
- chain 2.52, draft 2ms + int4 verify 7ms:  2.52/9ms ≈ **280** (beats the 275 single-pass ceiling)
- tree 4.28, draft 3ms + bf16 flat verify 8.9ms:  4.28/11.9ms ≈ **360** (clears 350)
- + int4 M=K megakernel verify at tree acceptance: → path to **800**

**The one remaining bottleneck: the draft rollout.** Live spec-decode is draft-bound — the K-step rollout
runs ~20ms of eager Python (K tiny forwards, each launching ~15 kernels) and dominates the pass. The
acceptance (2.52/4.28) is there but not yet realized as tok/s. The fix is a CUDA-graphed draft with a
fixed preallocated KV cache (gpt-fast style) — a `torch.compile` over a *growing* cache recompiles per
step; a fixed cache graphs cleanly. That collapses the rollout to ~2ms and converts the acceptance win
directly into the ~280 (chain) / ~360 (tree) above.

**Build order:** [done] self-distill drafter · [done] M=K GEMV cost measurement · [done] **multi-step
rollout training (acceptance 1.76→2.52 / 2.46→4.28)** · [next] CUDA-graphed fixed-cache draft rollout ·
[then] int4 M=K megakernel verify → 800.
