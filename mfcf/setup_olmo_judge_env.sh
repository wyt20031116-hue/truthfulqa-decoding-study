#!/bin/bash -l

set -eo pipefail

module load anaconda3

if ! conda env list | awk '{print $1}' | grep -qx truthfulqa-olmo-judge; then
  conda create -y -n truthfulqa-olmo-judge \
    --override-channels \
    --channel conda-forge \
    python=3.10 \
    pip
fi

conda activate truthfulqa-olmo-judge
python -m pip install --upgrade pip
python -m pip install \
  "torch>=2.4,<2.7" \
  "transformers>=4.48,<5" \
  "accelerate>=1.0,<2" \
  "bitsandbytes>=0.45,<1" \
  "pandas>=2.2,<3" \
  "tqdm>=4.66,<5" \
  "safetensors>=0.4,<1"

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY
