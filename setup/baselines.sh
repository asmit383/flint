#!/usr/bin/env bash
# flint — measure the B=1 decode baselines (the honest floor our kernel must beat).
#     bash setup/baselines.sh <PEAK_BW> [MODEL]
# PEAK_BW (HBM, for MBU):  A100-40=1.55e12  A100-80/H100-PCIe=2.0e12  H100-SXM=3.35e12  H200=4.8e12
set -euo pipefail

PEAK="${1:-2.0e12}"
MODEL="${2:-ibm-granite/granite-4.1-3b}"
PY=/root/venv/bin/python
FLINT="${FLINT_DIR:-/root/flint}"
export PYTHONPATH="$FLINT"

echo "===== flint B=1 baselines  (peak-bw=$PEAK, model=$MODEL) ====="
echo "-- transformers eager (naive floor) --"
$PY "$FLINT/bench/throughput.py"    --model "$MODEL" --peak-bw "$PEAK" 2>&1 | grep -E "params|tok/s" || true
echo "-- gpt-fast bf16 (torch.compile + CUDA graphs) --"
$PY "$FLINT/bench/gptfast_speed.py"        --peak-bw "$PEAK" 2>&1 | grep -E "params|tok/s" || true
echo "-- gpt-fast int4 (torchao tinygemm) --"
$PY "$FLINT/bench/gptfast_speed.py" --int4 --peak-bw "$PEAK" 2>&1 | grep -E "params|tok/s" || true
echo "BASELINES-DONE"
