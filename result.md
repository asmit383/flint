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

| verify | tokens/pass |
|---|---|
| general (wiki/prose) | ~1.1 |
| **chain** (code avg) | **1.87** |
| tree branch=2 | 2.09 |
| tree branch=4 | 2.27 |
| **tree branch=8** | **2.46** (best prompt 2.70) |

Specialization + tree-verify: **1.1 → 2.46**, measured, held-out (spread 2.1–2.7 across 7 code prompts —
generalizes, not overfit to one style). Tree-verify: for greedy decoding only the greedy path is
acceptable, so a top-b tree accepts while the true token stays in the draft's top-b (no tree-attention-mask
needed to measure the ceiling).

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

## 7. The path to 800 (measured, honest)

```
net tok/s = 275 (single-pass) × accepted / verify_cost
```
- tensor-core verify only, current acceptance: 275 × 1.7/1.2 ≈ **390** (clears 350)
- + EAGLE-3 acceptance (~3.5): 275 × 3.5/1.2 ≈ **800**

Needs **both** a cheap verify (tensor-core M=K GEMM) **and** higher acceptance (EAGLE-3). Neither alone gets
there. EAGLE-3 multi-layer feature fusion was attempted and parked — a fused 2k draft overfit (98% train,
2.10 held-out) vs the last-layer 8k draft's 2.46; the real gain likely needs token-prediction recurrence
(not feature-prediction) + more data.

**Build order:** (1) tensor-core M=K GEMM verify [milestone 1 — the M=K GEMV + this cost measurement — done];
(2) full M=K megakernel decode; (3) spec serving loop + wire chat.py, measure net; (4) EAGLE-3 acceptance.
