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
- **Claim (to be measured, honestly, same harness):** +X% B=1 decode over the int4 baseline at a stated
  ppl cost — with the sparsity/ppl trade shown, not hidden.

## Layout
```
bench/    measurement harnesses  (sparsity gate, baselines) — local source of truth
kernels/  custom CUDA (sparse-int4 GEMV, fused rmsnorm)
engine/   gpt-fast fork + integration
scripts/  box setup / run scripts
notes/    plan + working notes (gitignored)
kdoc/     prior-project notes (gitignored)
```

## Order of operations
1. `scripts/setup_box.sh` — deps + model + gpt-fast.
2. `bench/sparsity_sweep.py` — **GO/NO-GO gate:** does Granite tolerate ~40-50% activation sparsity?
   (If not, switch models before writing anything.)
3. gpt-fast baseline (B=1 tok/s + MBU, bf16 then int4) — the honest floor.
4. Port the sparse-int4 GEMV, wire into the tight loop, measure the delta.
