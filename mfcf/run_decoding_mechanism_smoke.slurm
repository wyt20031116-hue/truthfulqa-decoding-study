#!/bin/bash -l
#SBATCH --job-name=tqa_mech_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_mechanism_smoke_%j.out

set -eo pipefail
module load anaconda3
conda activate truthfulqa
cd "${SLURM_SUBMIT_DIR:-$PWD}"

python scripts/run_decoding_mechanism_qwen25.py \
  --device cuda \
  --limit-questions 2 \
  --repetitions 1 \
  --output-dir runs/mechanism_smoke_2q_21settings_1rep
