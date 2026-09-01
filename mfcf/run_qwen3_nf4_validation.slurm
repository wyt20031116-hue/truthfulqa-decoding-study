#!/bin/bash -l
#SBATCH --job-name=tqa_q3_nf4_val
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen3_nf4_validation_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
OUTPUT_DIR="outputs/qwen3_nf4_validation"
mkdir -p "$OUTPUT_DIR"

python scripts/judge_qwen3_nf4_batched.py \
  --model Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --input data/nf4_validation/combined_adjudicated_55.csv \
  --output "$OUTPUT_DIR/combined_adjudicated_55_nf4.csv" \
  --metrics "$OUTPUT_DIR/combined_adjudicated_55_nf4_runtime.json" \
  --batch-size 4 \
  --max-new-tokens 160 \
  --reason-mode compact \
  --overwrite

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/qwen3_nf4_validation")
x = pd.read_csv(root / "combined_adjudicated_55_nf4.csv")
expected = {"boundary_adjudicated_15": 15, "priority_adjudicated_40": 40}
for split, n in expected.items():
    selected = x[x["validation_split"].eq(split)].copy()
    assert len(selected) == n and selected["case_id"].nunique() == n
    selected.to_csv(root / f"{split}_nf4.csv", index=False)
PY

python scripts/evaluate_qwen3_nf4_validation.py \
  --boundary "$OUTPUT_DIR/boundary_adjudicated_15_nf4.csv" \
  --priority "$OUTPUT_DIR/priority_adjudicated_40_nf4.csv" \
  --output "$OUTPUT_DIR/report.json"

echo "NF4 validation completed. Review $OUTPUT_DIR/report.json before any bulk judging."
