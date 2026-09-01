#!/usr/bin/env python3
"""Validate and summarize human or external-judge labels for generated answers.

This script intentionally does not call a proprietary judge API. It exports a
provider-neutral rubric table and then validates imported labels, keeping API
credentials out of the experiment code and raw results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["experiment_id", "prompt_id", "temperature", "top_p", "top_k", "repetition_id"]
LABEL_COLUMNS = ["correctness", "informativeness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/generations_modes_antirepeat.csv")
    parser.add_argument("--judgments", default=None, help="已完成评分的 CSV；省略时只导出评分模板。")
    parser.add_argument("--template-output", default="outputs/generated_truthfulness_judging_template.csv")
    parser.add_argument("--scored-output", default="outputs/generations_with_truthfulness.csv")
    parser.add_argument("--summary-output", default="outputs/generated_truthfulness_summary.csv")
    parser.add_argument("--sample-per-cell", type=int, default=0, help="0=导出全部；正数=每个实验单元抽样。")
    parser.add_argument("--seed", type=int, default=443)
    return parser.parse_args()


def validate_generation_keys(df: pd.DataFrame) -> None:
    missing = set(KEY_COLUMNS + ["question", "generated_text"]) - set(df.columns)
    if missing:
        raise ValueError(f"Generation file is missing columns: {sorted(missing)}")
    duplicates = int(df.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        raise ValueError(f"Generation file contains {duplicates} duplicate keys")


def build_template(df: pd.DataFrame, sample_per_cell: int, seed: int) -> pd.DataFrame:
    if sample_per_cell > 0:
        groups = ["prompt_id", "temperature", "top_p", "top_k"]
        df = (
            df.groupby(groups, group_keys=False)
            .sample(n=sample_per_cell, random_state=seed)
            .sort_values(KEY_COLUMNS)
        )
    columns = KEY_COLUMNS + [
        "question",
        "best_answer",
        "correct_answers",
        "best_incorrect_answer",
        "incorrect_answers",
        "generated_text",
    ]
    template = df[columns].copy()
    template["correctness"] = pd.NA
    template["informativeness"] = pd.NA
    template["judge_id"] = ""
    template["judge_notes"] = ""
    return template


def validate_judgments(judgments: pd.DataFrame, expected_keys: pd.DataFrame) -> None:
    missing = set(KEY_COLUMNS + LABEL_COLUMNS + ["judge_id"]) - set(judgments.columns)
    if missing:
        raise ValueError(f"Judgment file is missing columns: {sorted(missing)}")
    duplicates = int(judgments.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        raise ValueError(f"Judgment file contains {duplicates} duplicate keys")
    correctness = pd.to_numeric(judgments["correctness"], errors="coerce")
    invalid_correctness = correctness.isna() | ~correctness.isin([0, 0.5, 1])
    if invalid_correctness.any():
        raise ValueError(
            "correctness must contain only 0, 0.5, or 1; "
            f"{int(invalid_correctness.sum())} invalid rows"
        )
    informativeness = pd.to_numeric(judgments["informativeness"], errors="coerce")
    invalid_informativeness = informativeness.isna() | ~informativeness.between(0, 1)
    if invalid_informativeness.any():
        raise ValueError(
            "informativeness must be complete scores in [0, 1]; "
            f"{int(invalid_informativeness.sum())} invalid rows"
        )
    if judgments["judge_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every judgment must record a non-empty judge_id")
    expected = set(expected_keys[KEY_COLUMNS].itertuples(index=False, name=None))
    observed = set(judgments[KEY_COLUMNS].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            f"Judgment keys do not match the exported template: missing={len(expected-observed)}, "
            f"unexpected={len(observed-expected)}"
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    generations = pd.read_csv(root / args.input)
    validate_generation_keys(generations)
    template = build_template(generations, args.sample_per_cell, args.seed)
    template_path = root / args.template_output
    template_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.judgments:
        template.to_csv(template_path, index=False)
        print(f"Saved provider-neutral judging template to {template_path}")
        return

    judgments = pd.read_csv(root / args.judgments)
    if "correctness" not in judgments and "truthful" in judgments:
        judgments["correctness"] = judgments["truthful"]
    if "informativeness" not in judgments and "informative" in judgments:
        judgments["informativeness"] = judgments["informative"]
    validate_judgments(judgments, template)
    labels = judgments[KEY_COLUMNS + LABEL_COLUMNS + ["judge_id", "judge_notes"]].copy()
    for column in LABEL_COLUMNS:
        labels[column] = labels[column].astype(float)
    labels["quality_product"] = labels["correctness"] * labels["informativeness"]
    labels["truthful"] = labels["correctness"]
    labels["informative"] = labels["informativeness"]
    scored = generations.merge(labels, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    summary = (
        scored.groupby(["temperature", "top_p", "top_k"])[
            ["correctness", "informativeness", "quality_product"]
        ]
        .agg(["count", "mean", "std", "sem"])
        .reset_index()
    )
    summary.columns = ["_".join(str(x) for x in col if x != "") for col in summary.columns]
    scored_path = root / args.scored_output
    summary_path = root / args.summary_output
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved scored generations to {scored_path}")
    print(f"Saved truthfulness summary to {summary_path}")


if __name__ == "__main__":
    main()
