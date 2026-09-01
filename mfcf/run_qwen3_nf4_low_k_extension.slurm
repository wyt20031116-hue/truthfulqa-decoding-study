#!/bin/bash -l
#SBATCH --job-name=tqa_q3_lowk
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen3_nf4_low_k_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

INPUT="runs/low_k_extension_50q_k1_2_5_5rep/generations_with_k10_anchor.csv"
OUTPUT_DIR="outputs/qwen3_nf4_low_k_extension_5000"
mkdir -p "$OUTPUT_DIR"

# Use the exact frozen judge configuration used for the 56,250-row study.
python scripts/judge_qwen3_nf4_batched.py \
  --model Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --input "$INPUT" \
  --output "$OUTPUT_DIR/judgments.csv" \
  --metrics "$OUTPUT_DIR/runtime.json" \
  --batch-size 4 \
  --max-new-tokens 160 \
  --reason-mode compact \
  --prompt-version v1

python scripts/validate_low_k_extension_judgments.py \
  --input "$INPUT" \
  --judgments "$OUTPUT_DIR/judgments.csv" \
  --output "$OUTPUT_DIR/validation.json"

echo "Qwen3-NF4 low-k extension judging and validation completed."
