#!/bin/bash -l
#SBATCH --job-name=tqa_q3_nf4_56k
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen3_nf4_pure_56250_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

INPUT="runs/main_50q_45pure_settings_25rep/generations.csv"
OUTPUT_DIR="outputs/qwen3_nf4_pure_56250"
mkdir -p "$OUTPUT_DIR"

# Frozen provisional production configuration.  V1 compact is used because it
# is the fastest already-validated NF4 configuration.  The hard-set limitation
# is documented in PROVISIONAL_JUDGE_NOTICE.md and must be reported.
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

python scripts/validate_qwen3_nf4_pure_56250.py \
  --input "$INPUT" \
  --judgments "$OUTPUT_DIR/judgments.csv" \
  --output "$OUTPUT_DIR/validation.json"

python scripts/analyze_pure_pk_provisional.py \
  --input "$OUTPUT_DIR/judgments.csv" \
  --output-dir analysis/pure_pk_provisional_qwen3_nf4

echo "Provisional NF4 judging and pure-p/pure-k analysis completed."
