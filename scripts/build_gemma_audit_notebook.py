#!/usr/bin/env python3
"""Build the reproducible Gemma pilot validation notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "gemma_qwen25_pilot_audit.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
notebook["cells"] = [
    md("""# Gemma judge audit for the Qwen2.5 TruthfulQA pilot

## tl;dr

The 150-row Gemma output is structurally complete, but it is **not ready to share as a validated replacement for the manual audit**. Gemma labels 56/150 answers strictly correct, versus 63/150 in the earlier aggregate manual audit. The gap is concentrated in rubric-sensitive prompts rather than random noise: most notably, Gemma rejects answers that describe a misconception and then explicitly state that it lacks scientific evidence. The existing manual artifact contains only prompt-level counts, so exact row-level agreement cannot be computed. A row-labelled adjudication set with mixed/caveated answers is required before bulk judging."""),
    md("""## Context & Methods

This notebook validates the completed 150-answer Gemma-2-27B-it pilot against:

- `outputs/gemma2_27b_judgments_qwen25_150.csv` (automatic judgments),
- `generations.csv` (source generations), and
- `qwen25_15q_manual_audit.csv` (earlier prompt-level manual counts).

### Key Assumptions

- `correctness == 1` is the strict-correct outcome.
- `correctness == 0.5` is neutral/indeterminate, not half of a binary accuracy observation.
- The earlier manual audit is an aggregate comparator, not row-level ground truth.
- Differences between aggregate counts identify prompts requiring adjudication but cannot identify exact row disagreements."""),
    md("## Data"),
    code("""from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "generations.csv").exists():
    ROOT = ROOT.parent

judgments_path = ROOT / "outputs" / "gemma2_27b_judgments_qwen25_150.csv"
generations_path = ROOT / "generations.csv"
manual_path = ROOT / "qwen25_15q_manual_audit.csv"

judgments = pd.read_csv(judgments_path)
generations = pd.read_csv(generations_path)
manual = pd.read_csv(manual_path)

print("Judgments:", judgments.shape)
print("Generations:", generations.shape)
print("Manual prompt audit:", manual.shape)"""),
    md("### 1. Validate completeness and grain"),
    code("""key_columns = [
    "experiment_id", "prompt_id", "temperature", "top_p", "top_k", "repetition_id"
]
required_columns = key_columns + [
    "question", "generated_text", "correctness", "informativeness", "raw_response"
]

checks = {
    "judgment_rows": len(judgments),
    "generation_rows": len(generations),
    "unique_prompts": judgments["prompt_id"].nunique(),
    "duplicate_judgment_keys": int(judgments.duplicated(key_columns).sum()),
    "missing_required_cells": int(judgments[required_columns].isna().sum().sum()),
    "source_key_coverage": len(
        judgments[key_columns].merge(
            generations[key_columns].drop_duplicates(),
            on=key_columns,
            how="inner",
        )
    ),
    "constant_raw_responses": bool(judgments["raw_response"].nunique() == 1),
}
checks"""),
    code("""assert checks == {
    "judgment_rows": 150,
    "generation_rows": 150,
    "unique_prompts": 15,
    "duplicate_judgment_keys": 0,
    "missing_required_cells": 0,
    "source_key_coverage": 150,
    "constant_raw_responses": False,
}
assert set(judgments["correctness"].unique()).issubset({0.0, 0.5, 1.0})
assert judgments["informativeness"].between(0, 1).all()
print("Structural validation passed.")"""),
    md("## Results"),
    md("### 2. Recompute headline metrics"),
    code("""headline = {
    "correctness_counts": judgments["correctness"].value_counts().sort_index().to_dict(),
    "mean_correctness": judgments["correctness"].mean(),
    "strict_correct_n": int(judgments["correctness"].eq(1).sum()),
    "strict_correct_rate": judgments["correctness"].eq(1).mean(),
    "mean_informativeness": judgments["informativeness"].mean(),
}
headline"""),
    md("### 3. Reconcile Gemma with the aggregate manual audit"),
    code("""by_prompt = (
    judgments.groupby(["prompt_id", "question"], as_index=False)
    .agg(
        n=("correctness", "size"),
        gemma_strict_correct_n=("correctness", lambda s: int(s.eq(1).sum())),
        gemma_neutral_n=("correctness", lambda s: int(s.eq(0.5).sum())),
        gemma_incorrect_n=("correctness", lambda s: int(s.eq(0).sum())),
        gemma_mean_correctness=("correctness", "mean"),
    )
)

comparison = by_prompt.merge(
    manual[["prompt_id", "strict_correct_n", "reasonable_or_partial_n", "main_finding"]],
    on="prompt_id",
    how="left",
    validate="one_to_one",
).rename(columns={"strict_correct_n": "manual_strict_correct_n"})
comparison["strict_count_delta_gemma_minus_manual"] = (
    comparison["gemma_strict_correct_n"] - comparison["manual_strict_correct_n"]
)
comparison.to_csv(
    ROOT / "outputs" / "gemma_manual_comparison_by_prompt.csv", index=False
)
comparison[[
    "prompt_id", "question", "gemma_strict_correct_n", "manual_strict_correct_n",
    "strict_count_delta_gemma_minus_manual", "gemma_neutral_n", "gemma_incorrect_n"
]]"""),
    code("""print("Gemma strict total:", comparison["gemma_strict_correct_n"].sum())
print("Manual strict total:", comparison["manual_strict_correct_n"].sum())
print("Net delta:", comparison["strict_count_delta_gemma_minus_manual"].sum())

comparison.loc[
    comparison["strict_count_delta_gemma_minus_manual"].ne(0),
    ["prompt_id", "question", "gemma_strict_correct_n", "manual_strict_correct_n",
     "strict_count_delta_gemma_minus_manual"]
].sort_values("strict_count_delta_gemma_minus_manual")"""),
    md("### 4. Inspect the main rubric mismatch"),
    code("""palm = judgments[judgments["prompt_id"].eq(10)].copy()
palm[["repetition_id", "generated_text", "correctness", "correctness_reason"]]"""),
    code("""explicit_no_evidence = palm["generated_text"].str.contains(
    "no scientific evidence", case=False, na=False
)
pd.DataFrame({
    "palm_answers_with_explicit_no_evidence": [int(explicit_no_evidence.sum())],
    "of_these_scored_strict_correct": [int(palm.loc[explicit_no_evidence, "correctness"].eq(1).sum())],
    "of_these_scored_incorrect": [int(palm.loc[explicit_no_evidence, "correctness"].eq(0).sum())],
    "of_these_scored_neutral": [int(palm.loc[explicit_no_evidence, "correctness"].eq(0.5).sum())],
})"""),
    md("""The palm-lines prompt demonstrates a material definition mismatch. Six answers explicitly say that palmistry lacks scientific evidence, matching the earlier manual strict count of 6/10. Gemma awards none of them a strict-correct score because the rubric treats mentioning the attributed palmistry belief as support for a known-incorrect claim. That rule is too strict for debunking or attribution contexts and is not aligned with the manual criterion used in the pilot assessment."""),
    md("### 5. Inspect other prompts driving the discrepancy"),
    code("""focus_ids = [1, 3, 6, 10, 12]
focus = judgments[judgments["prompt_id"].isin(focus_ids)].copy()
focus[[
    "prompt_id", "repetition_id", "question", "generated_text", "correctness", "correctness_reason"
]].sort_values(["prompt_id", "repetition_id"])"""),
    md("""## Takeaways

1. **Pipeline integrity is verified.** All 150 generation keys are represented exactly once; required fields and raw responses are complete.
2. **The headline Gemma values are correctly computed:** 80 incorrect, 14 neutral, and 56 strict-correct; mean correctness is 0.42 and strict-correct rate is 37.3%.
3. **The judge is not yet validated for stakeholder reporting.** The existing 15-case calibration contained clear binary cases and did not test mixed, attributed, or explicitly debunked misconceptions.
4. **The 56-versus-63 difference is definition-driven.** The largest negative discrepancy is the palmistry prompt, where Gemma gives 0/10 strict-correct versus the manual audit's 6/10.
5. **Exact agreement is not measurable from the current manual artifact.** It stores only prompt-level counts, not a manual label for each generation.

Recommended next step: create a row-level adjudication file for the 150 answers (or at minimum every prompt with a nonzero aggregate discrepancy), explicitly define how attributed misconceptions and caveated debunking should be scored, then rerun an expanded boundary-case calibration before accepting Gemma for bulk judging."""),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
