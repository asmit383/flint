#!/usr/bin/env bash
# flint — one-shot box provisioning. Run on a FRESH GPU box:
#     bash setup/setup_box.sh [MODEL]
# Installs venv + torch + deps, downloads the model, clones gpt-fast (pre-flex SDPA build).
# Assumes a MODERN driver (>= 550 / CUDA 12.4+) so torch's default cu-wheel just works.
# (Old driver <= 535? see NOTE at the bottom — pin the cu wheel.)
set -euo pipefail

MODEL="${1:-ibm-granite/granite-4.1-3b}"
GPTFAST_COMMIT="091515ab5b06f91c0d6a3b92f9c27463f738cc9b"   # pre-FlexAttention: SDPA, compiles clean on any torch 2.x
VENV=/root/venv

echo "== [1/4] driver check =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "== [2/4] venv + deps (torch + inference + quant tooling) =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip uv
# torch pinned to cu128 (CUDA 12.8): matches driver >= 570. The DEFAULT 'latest' wheel is cu130
# (CUDA 13.0) which needs driver >= 580 and fails on most current boxes ("driver too old").
TORCH_CU="${TORCH_CU:-cu128}"
"$VENV/bin/uv" pip install --python "$VENV/bin/python" torch --index-url "https://download.pytorch.org/whl/${TORCH_CU}"
"$VENV/bin/uv" pip install --python "$VENV/bin/python" \
    transformers datasets accelerate huggingface_hub torchao \
    tiktoken blobfile numpy pandas pyarrow sentencepiece
"$VENV/bin/python" - <<'PY'
import torch, torchao
print(f"torch {torch.__version__} (cuda {torch.version.cuda}) | avail: {torch.cuda.is_available()} | torchao {torchao.__version__}")
assert torch.cuda.is_available(), "CUDA not available -> driver/torch cu-wheel mismatch (see NOTE)"
print("gpu:", torch.cuda.get_device_name(0))
PY

echo "== [3/4] model: $MODEL =="
"$VENV/bin/hf" download "$MODEL"

echo "== [4/4] gpt-fast (pre-flex SDPA build) =="
cd /root && [ -d gpt-fast ] || git clone -q https://github.com/pytorch-labs/gpt-fast.git
cd /root/gpt-fast && git checkout -q "$GPTFAST_COMMIT"

echo "SETUP-DONE  |  venv=$VENV  model=$MODEL  gpt-fast=/root/gpt-fast"
# NOTE (old driver <= 535): default torch wheel fails at runtime. Pin a matching cu wheel, e.g.:
#   uv pip install --python /root/venv/bin/python --reinstall torch --index-url https://download.pytorch.org/whl/cu121
#   uv pip install --python /root/venv/bin/python "torchao==0.7.0"
