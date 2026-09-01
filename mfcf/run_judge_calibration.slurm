#!/bin/bash -l
#SBATCH --job-name=tqa_judge_cal
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_judge_calibration_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p outputs/judge_calibration
: > outputs/judge_calibration/attempts.log
rm -f outputs/judge_calibration/selected_judge.txt

run_candidate() {
  local slug="$1"
  local model="$2"
  local quantization_flag="${3:-}"
  local csv="outputs/judge_calibration/${slug}.csv"
  local report="outputs/judge_calibration/${slug}_report.json"

  echo "START ${slug} ${model}" | tee -a outputs/judge_calibration/attempts.log
  if python scripts/calibrate_local_judge.py \
      --model "$model" \
      --input data/judge_calibration_15.csv \
      --output "$csv" \
      --overwrite \
      $quantization_flag; then
    if python scripts/evaluate_judge_calibration.py \
        --input "$csv" \
        --report "$report"; then
      printf '%s\n' "$model" > outputs/judge_calibration/selected_judge.txt
      echo "ACCEPT ${slug} ${model}" | tee -a outputs/judge_calibration/attempts.log
      return 0
    fi
    echo "REJECT ${slug} calibration thresholds failed" \
      | tee -a outputs/judge_calibration/attempts.log
  else
    echo "REJECT ${slug} model run failed" \
      | tee -a outputs/judge_calibration/attempts.log
  fi
  return 1
}

# Reproduce OLMo first and preserve its literal decoded response and token IDs.
if run_candidate olmo allenai/OLMo-2-1124-13B-Instruct; then
  exit 0
fi

# Professor Qin's preferred independent-family candidates.
# Llama 70B uses NF4 to fit on one A100. Its gated repository may require
# access through an HF_TOKEN environment variable.
if run_candidate llama33_70b meta-llama/Llama-3.3-70B-Instruct --load-in-4bit; then
  exit 0
fi
if run_candidate gemma2_27b google/gemma-2-27b-it; then
  exit 0
fi

# Explicit final fallback requested by Professor Qin.
if run_candidate qwen3_32b Qwen/Qwen3-32B; then
  exit 0
fi

echo "No candidate passed calibration." | tee -a outputs/judge_calibration/attempts.log
exit 2
