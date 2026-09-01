#!/bin/bash -l
#SBATCH --job-name=tqa_qwen25_pure
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_pure_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p runs/main_50q_45pure_settings_25rep

python scripts/run_experiment_qwen25_7b.py \
  --pure-only \
  --device cuda

python - <<'PY'
import pandas as pd

path = "runs/main_50q_45pure_settings_25rep/generations.csv"
rows = pd.read_csv(path)
keys = [
    "prompt_id",
    "decoding_mode",
    "temperature",
    "top_p",
    "top_k",
    "repetition_id",
]
settings = rows[
    ["decoding_mode", "temperature", "top_p", "top_k"]
].drop_duplicates()

assert len(rows) == 56250, f"Expected 56,250 rows, got {len(rows)}"
assert rows["prompt_id"].nunique() == 50
assert rows["repetition_id"].nunique() == 25
assert not rows.duplicated(keys).any()
assert set(rows["decoding_mode"]) == {"pure_top_p", "pure_top_k"}
assert len(settings) == 45, f"Expected 45 settings, got {len(settings)}"
assert (settings["decoding_mode"] == "pure_top_p").sum() == 20
assert (settings["decoding_mode"] == "pure_top_k").sum() == 25
assert rows.loc[rows["decoding_mode"].eq("pure_top_p"), "top_k"].eq(0).all()
assert rows.loc[rows["decoding_mode"].eq("pure_top_k"), "top_p"].eq(1.0).all()
print("Pure-run validation passed: 56,250 unique generations across 45 settings.")
PY
