#!/bin/bash -l
#SBATCH --job-name=tqa_bound_cal
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_boundary_calibration_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p outputs/judge_boundary_calibration

run_candidate() {
  local short_name="$1"
  local model_name="$2"

  python scripts/calibrate_local_judge.py \
    --model "$model_name" \
    --input data/judge_boundary_calibration_15.csv \
    --output "outputs/judge_boundary_calibration/${short_name}.csv" \
    --overwrite

  python scripts/evaluate_judge_calibration.py \
    --input "outputs/judge_boundary_calibration/${short_name}.csv" \
    --report "outputs/judge_boundary_calibration/${short_name}_report.json" || true
}

run_candidate gemma2_27b google/gemma-2-27b-it
run_candidate qwen3_32b Qwen/Qwen3-32B

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/judge_boundary_calibration")
candidates = [
    ("google/gemma-2-27b-it", root / "gemma2_27b_report.json"),
    ("Qwen/Qwen3-32B", root / "qwen3_32b_report.json"),
]
selected = None
for model, path in candidates:
    report = json.loads(path.read_text())
    if report["accepted"]:
        selected = model
        break
if selected is None:
    raise SystemExit("Neither candidate passed the boundary calibration")
(root / "selected_judge.txt").write_text(selected + "\n")
print(f"SELECTED {selected}")
PY
