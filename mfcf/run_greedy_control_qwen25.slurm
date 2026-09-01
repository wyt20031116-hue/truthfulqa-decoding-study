#!/bin/bash -l
#SBATCH --job-name=tqa_greedy50
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=truthfulqa_greedy_control_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

cd "${SLURM_SUBMIT_DIR:-$PWD}"
python scripts/run_greedy_control_qwen25.py

echo "Greedy control completed: runs/greedy_control_50q_5rep/validation.json"
