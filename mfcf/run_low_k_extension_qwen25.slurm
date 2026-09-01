#!/bin/bash -l
#SBATCH --job-name=tqa_q25_lowk
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen25_low_k_extension_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

NEW_ROOT="runs/low_k_extension_50q_k1_2_5_5rep"
OLD_ROOT="runs/main_50q_45pure_settings_25rep"
mkdir -p "$NEW_ROOT"

python scripts/run_experiment_qwen25_7b.py \
  --low-k-extension \
  --device cuda \
  --top-ks 1 2 5

python - <<'PY'
import hashlib
import json
from pathlib import Path

import pandas as pd

new_root = Path("runs/low_k_extension_50q_k1_2_5_5rep")
old_root = Path("runs/main_50q_45pure_settings_25rep")
new_questions = pd.read_csv(new_root / "sampled_questions.csv")
old_questions = pd.read_csv(old_root / "sampled_questions.csv")

# The follow-up must use exactly the original 50-question sampling frame.
pd.testing.assert_frame_equal(
    new_questions.reset_index(drop=True),
    old_questions.reset_index(drop=True),
    check_dtype=False,
)

new = pd.read_csv(new_root / "generations.csv", low_memory=False)
keys = [
    "prompt_id", "decoding_mode", "temperature", "top_p", "top_k", "repetition_id"
]
assert len(new) == 3750, len(new)
assert not new.duplicated(keys).any()
assert new["prompt_id"].nunique() == 50
assert set(new["temperature"]) == {0.1, 0.3, 0.7, 1.0, 1.5}
assert set(new["top_k"]) == {1, 2, 5}
assert new["top_p"].eq(1.0).all()
assert set(new["repetition_id"]) == set(range(5))
assert new["decoding_mode"].eq("pure_top_k").all()

# Reuse the original k=10 observations as an unchanged comparison anchor.
old = pd.read_csv(old_root / "generations.csv", low_memory=False)
anchor = old[
    old["decoding_mode"].eq("pure_top_k")
    & old["top_k"].eq(10)
    & old["repetition_id"].lt(5)
].copy()
assert len(anchor) == 1250, len(anchor)

combined = pd.concat([new, anchor], ignore_index=True)
assert len(combined) == 5000
assert not combined.duplicated(keys).any()
assert set(combined["top_k"]) == {1, 2, 5, 10}
combined = combined.sort_values(keys).reset_index(drop=True)
combined_path = new_root / "generations_with_k10_anchor.csv"
combined.to_csv(combined_path, index=False)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "design": "post_hoc_low_k_follow_up",
    "new_k_values": [1, 2, 5],
    "anchor_k_value": 10,
    "temperatures": [0.1, 0.3, 0.7, 1.0, 1.5],
    "n_questions": 50,
    "repetitions_per_cell": 5,
    "new_generation_rows": 3750,
    "reused_anchor_rows": 1250,
    "combined_rows": 5000,
    "question_frame_matches_original_exactly": True,
    "original_questions_sha256": sha256(old_root / "sampled_questions.csv"),
    "new_questions_sha256": sha256(new_root / "sampled_questions.csv"),
    "combined_sha256": sha256(combined_path),
}
with open(new_root / "validation_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
print(json.dumps(manifest, indent=2))
print("Low-k extension validation passed.")
PY
