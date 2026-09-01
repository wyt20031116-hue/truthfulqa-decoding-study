#!/bin/bash -l
#SBATCH --job-name=tqa_qwen3_150
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=truthfulqa_qwen3_pilot_%j.out

set -eo pipefail

module load anaconda3
conda activate truthfulqa-olmo-judge

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p outputs/qwen3_pilot_150

# No --overwrite: a restarted job resumes from already completed unique keys.
python scripts/judge_generations_local.py \
  --model Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --input generations.csv \
  --output outputs/qwen3_pilot_150/judgments.csv

python - <<'PY'
import hashlib
import json
from pathlib import Path

import pandas as pd

output_dir = Path("outputs/qwen3_pilot_150")
judgment_path = output_dir / "judgments.csv"
manual_path = Path("qwen25_15q_manual_audit.csv")
rubric_path = Path("scripts/judge_with_olmo_local.py")
x = pd.read_csv(judgment_path)
manual = pd.read_csv(manual_path)
keys = [
    "experiment_id",
    "prompt_id",
    "temperature",
    "top_p",
    "top_k",
    "repetition_id",
]

assert len(x) == 150, f"Expected 150 judgments, found {len(x)}"
assert x["prompt_id"].nunique() == 15
assert not x.duplicated(keys).any()
assert x["raw_response"].notna().all()
assert x["generated_token_ids"].notna().all()
assert x["correctness"].isin([0, 0.5, 1]).all()
assert x["informativeness"].between(0, 1).all()

by_prompt = (
    x.assign(qwen3_strict_correct=x["correctness"].eq(1).astype(int))
    .groupby(["prompt_id", "question"], as_index=False)
    .agg(
        qwen3_strict_correct_n=("qwen3_strict_correct", "sum"),
        qwen3_mean_correctness=("correctness", "mean"),
        qwen3_mean_informativeness=("informativeness", "mean"),
    )
)
comparison = manual.merge(by_prompt, on=["prompt_id", "question"], validate="one_to_one")
comparison["strict_count_difference"] = (
    comparison["qwen3_strict_correct_n"] - comparison["strict_correct_n"]
)
comparison.to_csv(output_dir / "comparison_with_manual_by_prompt.csv", index=False)

summary = {
    "judge_id": str(x["judge_id"].iloc[0]),
    "rubric_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
    "n": len(x),
    "n_prompts": int(x["prompt_id"].nunique()),
    "correctness_counts": {
        str(k): int(v)
        for k, v in x["correctness"].value_counts().sort_index().items()
    },
    "mean_correctness": float(x["correctness"].mean()),
    "strict_correct_n": int(x["correctness"].eq(1).sum()),
    "strict_correct_rate": float(x["correctness"].eq(1).mean()),
    "mean_informativeness": float(x["informativeness"].mean()),
    "manual_strict_correct_n": int(manual["strict_correct_n"].sum()),
    "prompt_level_strict_count_mae": float(
        comparison["strict_count_difference"].abs().mean()
    ),
    "comparison_scope": (
        "Prompt-level strict-correct counts only; the existing manual audit "
        "does not contain row-level correctness or informativeness labels."
    ),
    "raw_responses_constant": bool(x["raw_response"].nunique() == 1),
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("Qwen3 150-row pilot validation and prompt-level manual comparison passed.")
PY
