#!/bin/bash -l
#SBATCH --job-name=tqa_k1_audit
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=truthfulqa_k1_forensic_%j.out

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load anaconda3
conda activate truthfulqa

python scripts/audit_k1_determinism.py

echo "k=1 forensic audit completed: analysis/k1_forensic/report.json"
