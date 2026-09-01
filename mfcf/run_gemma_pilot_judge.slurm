#!/bin/bash -l
#SBATCH --job-name=tqa_gemma_150
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_gemma_pilot_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p outputs

python scripts/judge_generations_local.py \
  --model google/gemma-2-27b-it \
  --input generations.csv \
  --output outputs/gemma2_27b_judgments_qwen25_150.csv

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

path = Path("outputs/gemma2_27b_judgments_qwen25_150.csv")
x = pd.read_csv(path)
assert len(x) == 150, f"Expected 150 judgments, found {len(x)}"
assert x["prompt_id"].nunique() == 15
assert not x.duplicated([
    "experiment_id",
    "prompt_id",
    "temperature",
    "top_p",
    "top_k",
    "repetition_id",
]).any()
summary = {
    "judge_id": x["judge_id"].iloc[0],
    "n": len(x),
    "n_prompts": int(x["prompt_id"].nunique()),
    "correctness_counts": {
        str(k): int(v)
        for k, v in x["correctness"].value_counts().sort_index().items()
    },
    "mean_correctness": float(x["correctness"].mean()),
    "strict_correct_rate": float(x["correctness"].eq(1).mean()),
    "mean_informativeness": float(x["informativeness"].mean()),
    "raw_responses_constant": bool(x["raw_response"].nunique() == 1),
}
Path("outputs/gemma2_27b_qwen25_150_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY
