#!/bin/bash -l
#SBATCH --job-name=tqa_q25_acc_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_reference_accuracy_smoke_%j.out

set -eo pipefail
module load anaconda3
conda activate truthfulqa
cd "${SLURM_SUBMIT_DIR:-$PWD}"

ROOT="outputs/qwen25_reference_accuracy_smoke"
mkdir -p "$ROOT"

python scripts/evaluate_accuracy_qwen25.py \
  --device cuda --limit 2 \
  --output "$ROOT/mc_by_question.csv" \
  --summary-output "$ROOT/mc_summary.csv" \
  --metadata-output "$ROOT/mc_metadata.json"

python scripts/evaluate_accuracy_by_decoding_qwen25.py \
  --device cuda --limit 2 \
  --settings-input runs/low_k_extension_50q_k1_2_5_5rep/generations_with_k10_anchor.csv \
  --output "$ROOT/low_k_by_question_setting.csv" \
  --summary-output "$ROOT/low_k_by_setting.csv" \
  --metadata-output "$ROOT/low_k_metadata.json"

python - <<'PY'
import json
import pandas as pd
root = "outputs/qwen25_reference_accuracy_smoke"
mc = pd.read_csv(f"{root}/mc_by_question.csv")
low = pd.read_csv(f"{root}/low_k_by_question_setting.csv")
meta = json.load(open(f"{root}/low_k_metadata.json"))
assert len(mc) == 2
assert len(low) == 40
assert low["prompt_id"].nunique() == 2
assert low[["decoding_mode", "temperature", "top_p", "top_k"]].drop_duplicates().shape[0] == 20
assert meta["n_settings"] == 20 and meta["n_rows"] == 40
assert "mc2_score_defined" in low
assert low.loc[low["mc2_score_defined"].eq(0), "mc2_truncated_score"].isna().all()
print("Qwen2.5 reference-accuracy smoke validation passed.")
PY
