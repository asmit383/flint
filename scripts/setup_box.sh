#!/usr/bin/env bash
# One-shot box setup for the sparse-int4 dense-decode project (A100/H100).
# Installs deps, downloads Granite-4.1, clones gpt-fast. Run:  bash scripts/setup_box.sh [MODEL]
set -e
MODEL="${1:-ibm-granite/granite-4.1-3b}"

echo "=== venv + core deps ==="
python3 -m venv /root/venv
/root/venv/bin/pip install -q --upgrade pip uv
# torch (default cu wheel) + everything the sparsity sweep and gpt-fast need
/root/venv/bin/uv pip install --python /root/venv/bin/python \
    torch transformers datasets accelerate huggingface_hub sentencepiece numpy
/root/venv/bin/python -c "import torch;print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"

echo "=== download model: $MODEL ==="
/root/venv/bin/hf download "$MODEL"

echo "=== clone gpt-fast (fork base) ==="
cd /root && [ -d gpt-fast ] || git clone https://github.com/pytorch-labs/gpt-fast.git

echo "SETUP-DONE ($MODEL ready; gpt-fast at /root/gpt-fast)"
