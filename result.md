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

## 6. Spec-decode ON the megakernel — built, measured end-to-end

Two pieces built and wired onto the int4 engine (not a bf16 stand-in):
- **M=K verify megakernel** (`kernels/verify_mega.cu`): scores K+1 candidates in ONE cooperative int4
  launch, every weight row read once. **Bit-exact vs the M=1 megakernel** (rel-L2 0.0000, all argmax match).
  Returns logits + the post-final-norm feature the draft rolls on. Batching all M queries into 2 grid-syncs
  (not 2M) flattened the cost a lot: `verify_cost(M) = time(M)/time(1)`, L2-cold, on the megakernel:

  | M | verify_cost | verify ms |
  |---|---|---|
  | 1 | 1.00 | 3.9 |
  | 3 | 1.64 | 6.6 |
  | 4 | 1.96 | 7.9 |
  | 8 | 2.65 | 10.6 |

- **`spec_mega.py`** — full loop: megakernel prefill → draft K → verify_mega M=K+1 → accept → update.
  Coherent output. **Measured 116 tok/s @ K=3** (draft 9.8ms + verify 8.0ms, acc 2.11).

**The measured wall: at B=1, chain spec-decode loses to single-pass.**
```
net = accepted / (draft + verify);  even with a FREE draft: 2.11 / 8.0ms = 264 < 275
```
Verify(M=4) is 1.96× a single token — verifying 4 tokens costs ~2 tokens, and greedy chain accepts ~2, so
it breaks even at best. The verify is **not flat** (register/occupancy-bound: M accumulators/thread drop
occupancy, so the extra math doesn't fully hide in the 27%-MBU latency slack). Acceptance being solved
(2.52/4.28) is necessary but not sufficient — the flat verify is what's missing.

## 7. The path above 275 — what it actually needs

```
net tok/s = accepted / (draft_roll_time + verify_time)
```
Both remaining levers are real builds:
- **Flat M=K verify** (`verify_cost(M) → ~1.1–1.2`): the whole-project thesis (B=1 is latency-bound, extra
  math should hide) says it's possible, but the current kernel is register-bound. With a flat verify: chain
  2.52/(2ms + 4.5ms) ≈ **388**; tree 4.28/(3ms + 6ms) ≈ **475**.
- **Tree-verify serving loop** to cash in the 4.28 tree acceptance (tree attention mask in verify_mega,
  longest-path accept, KV surgery). With today's *non-flat* verify: 4.28/(3ms + 10.6ms) ≈ **315** — the one
  path that clears 275 without the flat-verify kernel, if live tree acceptance holds near 4.28.

**Build order:** [done] self-distill drafter · [done] multi-step rollout training (acc 1.76→2.52 / 2.46→4.28)
· [done] fixed-cache CUDA-graphed draft rollout · [done] **M=K verify megakernel + end-to-end spec_mega
(116 tok/s, chain loses at B=1)** · [next] flat M=K verify (register/occupancy) AND/OR tree-verify loop.
