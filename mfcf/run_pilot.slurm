#!/bin/bash -l
#SBATCH --job-name=tqa_qwen7b_150
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen7b_150_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/pilot_15q_10rep

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('GPU GB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))"

python scripts/run_experiment_qwen25_7b.py \
  --device cuda \
  --no-resume \
  --output runs/pilot_15q_10rep/generations.csv \
  --sample-output runs/pilot_15q_10rep/sampled_questions.csv

python - <<'PY'
import pandas as pd

path = "runs/pilot_15q_10rep/generations.csv"
rows = pd.read_csv(path)
assert len(rows) == 150, f"Expected 150 rows, got {len(rows)}"
assert rows["question"].nunique() == 15
assert rows.groupby("question").size().eq(10).all()
print("Pilot validation passed: 15 questions x 10 repetitions = 150 rows")
print(rows[["question", "generated_text"]].to_string(index=False))
PY
