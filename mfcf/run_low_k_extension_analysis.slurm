#!/bin/bash -l
#SBATCH --job-name=tqa_lowk_analysis
#SBATCH --partition=cpu_pr3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=truthfulqa_low_k_analysis_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python scripts/analyze_low_k_extension.py \
  --input outputs/qwen3_nf4_low_k_extension_5000/judgments.csv \
  --audit outputs/qwen3_nf4_audit_validation_200/item_level_validation_200.csv \
  --output-dir analysis/low_k_extension

echo "Low-k setting-level analysis completed."
