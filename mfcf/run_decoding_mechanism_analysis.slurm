#!/bin/bash -l
#SBATCH --job-name=tqa_mech_analysis
#SBATCH --partition=cpu_pr3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=truthfulqa_mechanism_analysis_%j.out

set -eo pipefail
module load anaconda3
conda activate truthfulqa-olmo-judge
cd "${SLURM_SUBMIT_DIR:-$PWD}"

python scripts/analyze_decoding_mechanism.py
