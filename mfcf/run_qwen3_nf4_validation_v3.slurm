#!/bin/bash -l
#SBATCH --job-name=tqa_q3_nf4_v3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen3_nf4_validation_v3_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
OUTPUT_DIR="outputs/qwen3_nf4_validation_v3"
mkdir -p "$OUTPUT_DIR"

python scripts/judge_qwen3_nf4_batched.py \
  --model Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --input data/nf4_validation/combined_v3_95.csv \
  --output "$OUTPUT_DIR/combined_v3_95_nf4.csv" \
  --metrics "$OUTPUT_DIR/runtime.json" \
  --batch-size 4 \
  --max-new-tokens 192 \
  --reason-mode compact \
  --prompt-version v3 \
  --overwrite

python scripts/evaluate_qwen3_nf4_v3.py \
  --input "$OUTPUT_DIR/combined_v3_95_nf4.csv" \
  --output "$OUTPUT_DIR/report.json"

echo "NF4 v3 final validation completed. Review $OUTPUT_DIR/report.json."
