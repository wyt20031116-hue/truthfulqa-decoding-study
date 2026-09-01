#!/bin/bash -l
#SBATCH --job-name=tqa_pure_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_pure_smoke_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p runs/smoke_2q_5pure_settings_2rep

python scripts/run_experiment_qwen25_7b.py \
  --pure-only \
  --device cuda \
  --num-prompts 2 \
  --repetitions 2 \
  --limit-settings 5 \
  --output runs/smoke_2q_5pure_settings_2rep/generations.csv \
  --sample-output runs/smoke_2q_5pure_settings_2rep/sampled_questions.csv

python - <<'PY'
import json
import pandas as pd

root = "runs/smoke_2q_5pure_settings_2rep"
rows = pd.read_csv(f"{root}/generations.csv")
metadata = json.load(open(f"{root}/generations.csv.metadata.json"))
assert len(rows) == 20, f"Expected 20 rows, got {len(rows)}"
assert rows["prompt_id"].nunique() == 2
assert rows["repetition_id"].nunique() == 2
assert set(rows["decoding_mode"]) == {"pure_top_p", "pure_top_k"}
assert metadata["experiment_config"]["planned_generations"] == 20
print("Pure smoke validation passed: 20 generations across both pure modes.")
PY
