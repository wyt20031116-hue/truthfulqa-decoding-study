#!/usr/bin/env python3
"""Add decision-focused summaries to the stratified Qwen3-NF4 audit pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    return float((group[column] * group["audit_weight"]).sum() / group["audit_weight"].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-level", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    x = pd.read_csv(args.item_level)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    x["confidence_group"] = x["human_confidence"].str.lower()
    x["adjudication_group"] = x["needs_adjudication"].astype(str).str.lower().map(
        lambda value: "flagged" if value in {"true", "yes", "1"} else "not_flagged"
    )

    rows = []
    for dimension in ["confidence_group", "adjudication_group", "decoding_mode"]:
        for value, group in x.groupby(dimension, observed=True):
            rows.append({
                "dimension": dimension,
                "group": value,
                "n": len(group),
                "correctness_exact_unweighted": float(group["correctness_exact"].mean()),
                "correctness_exact_weighted": weighted_mean(group, "correctness_exact"),
                "correctness_mae_weighted": weighted_mean(group, "correctness_abs_error"),
                "informativeness_mae_weighted": weighted_mean(group, "informativeness_abs_error"),
            })
    pd.DataFrame(rows).to_csv(output / "validation_by_review_group.csv", index=False)

    sensitivity_rows = []
    for (mode, temperature), group in x.groupby(["decoding_mode", "temperature"], observed=True):
        qwen_correct = weighted_mean(group, "correctness")
        human_correct = weighted_mean(group, "human_correctness")
        qwen_info = weighted_mean(group, "informativeness")
        human_info = weighted_mean(group, "human_informativeness")
        conservative = group["correctness"].where(~group["correctness"].eq(0.5), 0.0)
        sensitivity_rows.append({
            "decoding_mode": mode,
            "temperature": temperature,
            "audit_n": len(group),
            "qwen3_correctness_weighted": qwen_correct,
            "human_correctness_weighted": human_correct,
            "correctness_audit_adjustment": human_correct - qwen_correct,
            "qwen3_correctness_conservative": float(
                (conservative * group["audit_weight"]).sum() / group["audit_weight"].sum()
            ),
            "qwen3_informativeness_weighted": qwen_info,
            "human_informativeness_weighted": human_info,
            "informativeness_audit_adjustment": human_info - qwen_info,
        })
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(["decoding_mode", "temperature"])
    sensitivity.to_csv(output / "sensitivity_detailed_by_mode_temperature.csv", index=False)

    direction_rows = []
    for temperature, group in sensitivity.groupby("temperature"):
        indexed = group.set_index("decoding_mode")
        if not {"pure_top_p", "pure_top_k"}.issubset(indexed.index):
            continue
        qwen_diff = indexed.loc["pure_top_p", "qwen3_correctness_weighted"] - indexed.loc[
            "pure_top_k", "qwen3_correctness_weighted"
        ]
        human_diff = indexed.loc["pure_top_p", "human_correctness_weighted"] - indexed.loc[
            "pure_top_k", "human_correctness_weighted"
        ]
        direction_rows.append({
            "temperature": temperature,
            "qwen3_top_p_minus_top_k_correctness": qwen_diff,
            "human_top_p_minus_top_k_correctness": human_diff,
            "direction_preserved": (qwen_diff == 0 and human_diff == 0) or (qwen_diff * human_diff > 0),
        })
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(output / "mode_contrast_direction.csv", index=False)

    summary = {
        "audit_n": int(len(x)),
        "unique_prompts": int(x["prompt_id"].nunique()),
        "weighted_correctness_exact_rate": weighted_mean(x, "correctness_exact"),
        "weighted_correctness_mae": weighted_mean(x, "correctness_abs_error"),
        "weighted_informativeness_mae": weighted_mean(x, "informativeness_abs_error"),
        "weighted_qwen3_correctness": weighted_mean(x, "correctness"),
        "weighted_human_correctness": weighted_mean(x, "human_correctness"),
        "weighted_correctness_bias_qwen_minus_human": weighted_mean(x, "correctness") - weighted_mean(x, "human_correctness"),
        "weighted_qwen3_informativeness": weighted_mean(x, "informativeness"),
        "weighted_human_informativeness": weighted_mean(x, "human_informativeness"),
        "weighted_informativeness_bias_qwen_minus_human": weighted_mean(x, "informativeness") - weighted_mean(x, "human_informativeness"),
        "human_0_qwen3_0_5_n": int(((x["human_correctness"] == 0) & (x["correctness"] == 0.5)).sum()),
        "human_0_n": int((x["human_correctness"] == 0).sum()),
        "mode_contrast_direction_preserved_n": int(direction["direction_preserved"].sum()),
        "mode_contrast_temperature_n": int(len(direction)),
        "recommended_decision": "do_not_resume_full_56250_judging_until_supervisor_reviews_item_level_results",
        "interpretation": "Qwen3-NF4 systematically uses 0.5 for answers the human audit scores 0; audit correction changes some mode contrasts even when the best-temperature result within each mode is preserved.",
    }
    (output / "decision_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
