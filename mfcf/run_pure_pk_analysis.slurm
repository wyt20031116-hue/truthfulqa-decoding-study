#!/bin/bash -l
#SBATCH --job-name=tqa_pure_analysis
#SBATCH --partition=cpu_pr3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=truthfulqa_pure_analysis_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

cd "${SLURM_SUBMIT_DIR:-$PWD}"
python scripts/validate_qwen3_nf4_pure_56250.py \
  --input runs/main_50q_45pure_settings_25rep/generations.csv \
  --judgments outputs/qwen3_nf4_pure_56250/judgments.csv \
  --output outputs/qwen3_nf4_pure_56250/validation.json
python scripts/analyze_pure_pk_provisional.py \
  --input outputs/qwen3_nf4_pure_56250/judgments.csv \
  --output-dir analysis/pure_pk_provisional_qwen3_nf4
