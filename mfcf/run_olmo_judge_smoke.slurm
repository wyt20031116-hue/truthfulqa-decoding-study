#!/bin/bash -l
#SBATCH --job-name=tqa_olmo_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_olmo_smoke_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
mkdir -p outputs

python -c "import torch; print(torch.cuda.get_device_name(0)); print('GPU GB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))"

python scripts/judge_with_olmo_local.py \
  --input generations.csv \
  --output outputs/olmo_calibration_15.csv \
  --one-per-prompt \
  --overwrite

python - <<'PY'
import pandas as pd

rows = pd.read_csv("outputs/olmo_calibration_15.csv")
assert len(rows) == 15
assert rows["prompt_id"].nunique() == 15
print(rows[[
    "question",
    "generated_text",
    "correctness",
    "informativeness",
    "correctness_reason",
    "informativeness_reason",
    "judge_id",
]].to_string(index=False))
PY
