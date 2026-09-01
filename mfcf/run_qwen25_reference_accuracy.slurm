#!/bin/bash -l
#SBATCH --job-name=tqa_q25_refacc
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_reference_accuracy_%j.out

set -eo pipefail
module load anaconda3
conda activate truthfulqa
cd "${SLURM_SUBMIT_DIR:-$PWD}"

ROOT="outputs/qwen25_reference_accuracy"
mkdir -p "$ROOT/main_pure" "$ROOT/low_k"

# Intrinsic reference preference: binary, MC1, and MC2, without decoding truncation.
python scripts/evaluate_accuracy_qwen25.py \
  --device cuda \
  --output "$ROOT/mc_by_question.csv" \
  --summary-output "$ROOT/mc_summary.csv" \
  --metadata-output "$ROOT/mc_metadata.json"

# Reference likelihood under the 45 settings actually used in the main pure-p/pure-k run.
python scripts/evaluate_accuracy_by_decoding_qwen25.py \
  --device cuda \
  --settings-input runs/main_50q_45pure_settings_25rep/generations.csv \
  --output "$ROOT/main_pure/by_question_setting.csv" \
  --summary-output "$ROOT/main_pure/by_setting.csv" \
  --metadata-output "$ROOT/main_pure/metadata.json"

# The same supplementary metric for the 20-cell low-k follow-up.
python scripts/evaluate_accuracy_by_decoding_qwen25.py \
  --device cuda \
  --settings-input runs/low_k_extension_50q_k1_2_5_5rep/generations_with_k10_anchor.csv \
  --output "$ROOT/low_k/by_question_setting.csv" \
  --summary-output "$ROOT/low_k/by_setting.csv" \
  --metadata-output "$ROOT/low_k/metadata.json"

python - <<'PY'
import json
import pandas as pd
root = "outputs/qwen25_reference_accuracy"
mc = pd.read_csv(f"{root}/mc_by_question.csv")
main = pd.read_csv(f"{root}/main_pure/by_question_setting.csv")
low = pd.read_csv(f"{root}/low_k/by_question_setting.csv")
assert len(mc) == 50 and mc["prompt_id"].nunique() == 50
assert len(main) == 2250 and main["prompt_id"].nunique() == 50
assert main[["decoding_mode", "temperature", "top_p", "top_k"]].drop_duplicates().shape[0] == 45
assert len(low) == 1000 and low["prompt_id"].nunique() == 50
assert low[["decoding_mode", "temperature", "top_p", "top_k"]].drop_duplicates().shape[0] == 20
for relative, expected in [
    ("main_pure/metadata.json", (45, 2250)),
    ("low_k/metadata.json", (20, 1000)),
]:
    metadata = json.load(open(f"{root}/{relative}"))
    assert (metadata["n_settings"], metadata["n_rows"]) == expected
print("Qwen2.5 reference-accuracy full validation passed.")
PY
