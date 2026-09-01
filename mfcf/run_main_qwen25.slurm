#!/bin/bash -l
#SBATCH --job-name=tqa_qwen25_main
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_main_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p runs/main_50q_145settings_25rep

python scripts/run_experiment_qwen25_7b.py \
  --full-grid \
  --device cuda

python - <<'PY'
import pandas as pd

path = "runs/main_50q_145settings_25rep/generations.csv"
rows = pd.read_csv(path)
keys = [
    "prompt_id",
    "decoding_mode",
    "temperature",
    "top_p",
    "top_k",
    "repetition_id",
]
assert len(rows) == 181250, f"Expected 181250 rows, got {len(rows)}"
assert rows["prompt_id"].nunique() == 50
assert not rows.duplicated(keys).any()
print("Main-run validation passed: 181,250 unique generations.")
PY
