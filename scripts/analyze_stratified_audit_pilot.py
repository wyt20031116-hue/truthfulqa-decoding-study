#!/usr/bin/env python3
"""Produce item-level validation and weighted sensitivity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORRECTNESS = {0.0, 0.5, 1.0}
INFORMATIVENESS = {0.0, 0.25, 0.5, 0.75, 1.0}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.astype(float), weights=weights.astype(float)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blinded", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    human = pd.read_csv(args.blinded)
    key = pd.read_csv(args.key)
    if human["audit_id"].duplicated().any() or key["audit_id"].duplicated().any():
        raise ValueError("Duplicate audit_id")
    merged = key.merge(
        human[["audit_id", "human_correctness", "human_informativeness", "human_reason",
               "human_confidence", "needs_adjudication", "reviewer"]],
        on="audit_id", how="left", validate="one_to_one",
    )
    if len(merged) != 200:
        raise ValueError(f"Expected 200 audited items, found {len(merged)}")
    if merged[["human_correctness", "human_informativeness", "human_reason",
               "human_confidence", "needs_adjudication"]].isna().any().any():
        raise ValueError("Human audit is incomplete")
    merged["human_correctness"] = merged["human_correctness"].astype(float)
    merged["human_informativeness"] = merged["human_informativeness"].astype(float)
    if not set(merged["human_correctness"]).issubset(CORRECTNESS):
        raise ValueError("Invalid human_correctness value")
    if not set(merged["human_informativeness"]).issubset(INFORMATIVENESS):
        raise ValueError("Invalid human_informativeness value")
    if not set(merged["human_confidence"].str.lower()).issubset({"high", "medium", "low"}):
        raise ValueError("human_confidence must be high, medium, or low")

    weights = merged["audit_weight"]
    merged["correctness_exact"] = merged["correctness"].eq(merged["human_correctness"])
    merged["correctness_abs_error"] = (merged["correctness"] - merged["human_correctness"]).abs()
    merged["informativeness_abs_error"] = (
        merged["informativeness"] - merged["human_informativeness"]
    ).abs()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "item_level_validation_200.csv", index=False)

    confusion = pd.crosstab(
        merged["human_correctness"], merged["correctness"],
        rownames=["human"], colnames=["qwen3"], dropna=False,
    ).reindex(index=[0, 0.5, 1], columns=[0, 0.5, 1], fill_value=0)
    confusion.to_csv(output / "correctness_confusion_matrix.csv")

    class_rows = []
    for label in [0.0, 0.5, 1.0]:
        selected = merged[merged["human_correctness"].eq(label)]
        class_rows.append({
            "human_class": label,
            "n": len(selected),
            "exact_recall_unweighted": float(selected["correctness_exact"].mean()),
            "exact_recall_weighted": weighted_mean(selected["correctness_exact"], selected["audit_weight"]),
        })
    pd.DataFrame(class_rows).to_csv(output / "correctness_class_recall.csv", index=False)

    stratum_rows = []
    for stratum_name, g in merged.groupby("sampling_stratum", observed=True):
        stratum_rows.append({
            "sampling_stratum": stratum_name,
            "n": len(g),
            "correctness_exact": g["correctness_exact"].mean(),
            "correctness_mae": g["correctness_abs_error"].mean(),
            "informativeness_mae": g["informativeness_abs_error"].mean(),
        })
    stratum = pd.DataFrame(stratum_rows)
    stratum.to_csv(output / "validation_by_sampling_stratum.csv", index=False)

    sensitivity_rows = []
    for keys, group in merged.groupby(["decoding_mode", "temperature"], observed=True):
        sensitivity_rows.append({
            "decoding_mode": keys[0],
            "temperature": keys[1],
            "audit_n": len(group),
            "qwen3_correctness_weighted": weighted_mean(group["correctness"], group["audit_weight"]),
            "human_correctness_weighted": weighted_mean(group["human_correctness"], group["audit_weight"]),
            "qwen3_informativeness_weighted": weighted_mean(group["informativeness"], group["audit_weight"]),
            "human_informativeness_weighted": weighted_mean(group["human_informativeness"], group["audit_weight"]),
            "qwen3_boundary_conservative_weighted": weighted_mean(
                group["correctness"].where(~group["correctness"].eq(0.5), 0), group["audit_weight"]
            ),
        })
    pd.DataFrame(sensitivity_rows).to_csv(output / "sensitivity_by_mode_temperature.csv", index=False)

    summary = {
        "audit_n": len(merged),
        "correctness_exact_unweighted": float(merged["correctness_exact"].mean()),
        "correctness_exact_weighted": weighted_mean(merged["correctness_exact"], weights),
        "correctness_mae_unweighted": float(merged["correctness_abs_error"].mean()),
        "correctness_mae_weighted": weighted_mean(merged["correctness_abs_error"], weights),
        "informativeness_mae_unweighted": float(merged["informativeness_abs_error"].mean()),
        "informativeness_mae_weighted": weighted_mean(merged["informativeness_abs_error"], weights),
        "qwen3_mean_correctness_weighted": weighted_mean(merged["correctness"], weights),
        "human_mean_correctness_weighted": weighted_mean(merged["human_correctness"], weights),
        "qwen3_mean_informativeness_weighted": weighted_mean(merged["informativeness"], weights),
        "human_mean_informativeness_weighted": weighted_mean(merged["human_informativeness"], weights),
        "boundary_0_to_0.5_n": int(((merged["human_correctness"] == 0) & (merged["correctness"] == 0.5)).sum()),
        "boundary_0.5_to_0_n": int(((merged["human_correctness"] == 0.5) & (merged["correctness"] == 0)).sum()),
        "low_confidence_n": int(merged["human_confidence"].str.lower().eq("low").sum()),
        "needs_adjudication_n": int(merged["needs_adjudication"].astype(str).str.lower().isin(["true", "yes", "1"]).sum()),
    }
    (output / "audit_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
