#!/bin/bash -l
#SBATCH --job-name=tqa_q3_screen
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen3_nf4_audit_screen_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DIR="outputs/qwen3_nf4_audit_screen"
mkdir -p "$DIR"

python scripts/build_balanced_audit_screen.py \
  --generations runs/main_50q_45pure_settings_25rep/generations.csv \
  --partial-judgments outputs/qwen3_nf4_pilot_partial/judgments_partial_58566.csv \
  --output-dir "$DIR"

python scripts/judge_qwen3_nf4_batched.py \
  --model Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --input "$DIR/screen_to_judge.csv" \
  --output "$DIR/screen_judgments.csv" \
  --metrics "$DIR/screen_runtime.json" \
  --batch-size 4 \
  --max-new-tokens 160 \
  --reason-mode compact \
  --prompt-version v1 \
  --overwrite

python scripts/assemble_balanced_audit_frame.py \
  --existing "$DIR/existing_judged_balanced.csv" \
  --screen-judgments "$DIR/screen_judgments.csv" \
  --output "$DIR/balanced_judged_frame_1000.csv" \
  --manifest "$DIR/balanced_frame_manifest.json"

python scripts/build_stratified_audit_pilot.py \
  --input "$DIR/balanced_judged_frame_1000.csv" \
  --output-dir outputs/qwen3_nf4_audit_pilot_200_50q

echo "Balanced 50-question audit pilot is ready."
