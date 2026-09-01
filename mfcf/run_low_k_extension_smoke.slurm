#!/bin/bash -l
#SBATCH --job-name=tqa_q25_lowk_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_low_k_smoke_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

ROOT="runs/low_k_extension_smoke_50q_1temp_1rep"
mkdir -p "$ROOT"

python scripts/run_experiment_qwen25_7b.py \
  --low-k-extension \
  --device cuda \
  --num-prompts 50 \
  --repetitions 1 \
  --temperatures 0.7 \
  --top-ks 1 2 5 \
  --output "$ROOT/generations.csv" \
  --sample-output "$ROOT/sampled_questions.csv"

python - <<'PY'
import pandas as pd

root = "runs/low_k_extension_smoke_50q_1temp_1rep"
old_root = "runs/main_50q_45pure_settings_25rep"
x = pd.read_csv(f"{root}/generations.csv", low_memory=False)
new_q = pd.read_csv(f"{root}/sampled_questions.csv")
old_q = pd.read_csv(f"{old_root}/sampled_questions.csv").reset_index(drop=True)

pd.testing.assert_frame_equal(new_q.reset_index(drop=True), old_q, check_dtype=False)
assert len(x) == 150
assert x["prompt_id"].nunique() == 50
assert set(x["top_k"]) == {1, 2, 5}
assert set(x["temperature"]) == {0.7}
assert x["top_p"].eq(1.0).all()
assert x["repetition_id"].eq(0).all()
print("Low-k extension smoke validation passed: 150 rows and exact 50-question match.")
PY
