# Results — hand-rolled grid barrier (inline PTX vs C++ vs cooperative groups)

Question: is a resident-grid barrier practical to hand-roll (vs `cg::grid.sync()`), and does inline PTX
beat C++ intrinsics? Measured on H100 PCIe.

## 1. Barrier in isolation (`bench/grid_barrier_test.py`)

Pure barrier cost, no work between barriers.

| barrier | ns / barrier | vs grid.sync |
|---|---|---|
| `cg::grid.sync()` | 2308 | 1.00× |
| hand-rolled, C++ intrinsics (`atomicInc` + `__threadfence`) | 1958 | **0.85×** |
| hand-rolled, inline PTX (`atom.inc.u32` / `membar.gl` / `ld.volatile`) | 1958 | **0.85×** |

- Hand-rolling is **practical** and ~15% **faster** than `grid.sync()` in isolation (leaner: one thread/block
  on a counter vs cg syncing all threads).
- **Inline PTX ≡ C++ intrinsics** — identical to the nanosecond. `atomicInc`/`__threadfence` already compile
  to `atom.inc`/`membar`, so PTX buys nothing over the intrinsics here.

## 2. Same barrier inside the full megakernel (`bench/megakernel_decode_test.py`, `HANDROLL=1`)

Real 40-layer decode, ~440 barriers/token with GEMV traffic in flight. Both numerically correct (argmax 17653).

| barrier | tok/s | MBU | vs baseline |
|---|---|---|---|
| `cg::grid.sync()` | **286.7** | 25.2% | 1.00× |
| hand-rolled inline PTX | 262.2 | 23.0% | **0.91× (~9% slower)** |

## Conclusion

The microbench win **did not transfer** — 15% faster in isolation, 9% slower end-to-end. The `membar.gl`
fence is nearly free in the empty microbench (no memory traffic to order) but expensive in the real kernel,
where every barrier must fence hundreds of in-flight GEMV writes; `grid.sync()`'s hardware semantics win
once there is real work. Same lesson as the L2-cache artifact: only the token clock counts.

Portability note: C++ intrinsics ≡ inline PTX in speed, and the C++ form is the portable one (`hipify`
translates `atomicInc`/`__threadfence` to CDNA; inline PTX does not port). So hand-rolling the barrier is
the right fallback on hardware where cooperative groups differs — but on NVIDIA, `cg::grid.sync()` is better
for the real workload, and there is no reason to reach for PTX here.
