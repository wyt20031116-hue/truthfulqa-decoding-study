#!/bin/bash -l
#SBATCH --job-name=tqa_qwen7b_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen7b_smoke_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/smoke

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('GPU GB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))"

python scripts/run_experiment_qwen25_7b.py \
  --device cuda \
  --num-prompts 2 \
  --repetitions 2 \
  --limit-settings 3 \
  --max-new-tokens 48 \
  --no-resume \
  --output runs/smoke/generations.csv \
  --sample-output runs/smoke/sampled_questions.csv
