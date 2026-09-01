#!/bin/bash -l
#SBATCH --job-name=tqa_olmo_judge
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_olmo_judge_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARED="$ROOT/scripts"
mkdir -p "$ROOT/outputs"
cd "$ROOT"

python -c "import torch; print(torch.cuda.get_device_name(0)); print('GPU GB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))"

python scripts/judge_with_olmo_local.py \
  --input generations.csv \
  --output outputs/olmo_judgments_qwen25.csv

python scripts/judge_with_olmo_local.py \
  --input distilgpt2_matched_150.csv \
  --output outputs/olmo_judgments_distilgpt2.csv

python "$SHARED/score_generated_truthfulness.py" \
  --input generations.csv \
  --judgments outputs/olmo_judgments_qwen25.csv \
  --scored-output outputs/qwen25_with_olmo_scores.csv \
  --summary-output outputs/qwen25_olmo_summary.csv

python "$SHARED/score_generated_truthfulness.py" \
  --input distilgpt2_matched_150.csv \
  --judgments outputs/olmo_judgments_distilgpt2.csv \
  --scored-output outputs/distilgpt2_with_olmo_scores.csv \
  --summary-output outputs/distilgpt2_olmo_summary.csv

python - <<'PY'
import pandas as pd

for name in ["qwen25", "distilgpt2"]:
    path = f"outputs/{name}_with_olmo_scores.csv"
    data = pd.read_csv(path)
    assert len(data) == 150, (name, len(data))
    print(f"\n{name}: 150 rows validated")
    print(
        data[["correctness", "informativeness", "quality_product"]]
        .agg(["mean", "std"])
        .to_string()
    )
    print(data["correctness"].value_counts().sort_index().to_string())
PY
