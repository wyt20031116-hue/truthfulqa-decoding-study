#!/bin/bash -l
#SBATCH --job-name=tqa_gemma_cal
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_gemma_calibration_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p outputs/judge_calibration

python scripts/calibrate_local_judge.py \
  --model google/gemma-2-27b-it \
  --input data/judge_calibration_15.csv \
  --output outputs/judge_calibration/gemma2_27b.csv \
  --overwrite

python scripts/evaluate_judge_calibration.py \
  --input outputs/judge_calibration/gemma2_27b.csv \
  --report outputs/judge_calibration/gemma2_27b_report.json

printf '%s\n' "google/gemma-2-27b-it" \
  > outputs/judge_calibration/selected_judge.txt
