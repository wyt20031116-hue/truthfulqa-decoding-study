#!/usr/bin/env python3
"""Evaluate v2 on development regression sets and a fresh untouched holdout."""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def split_metrics(frame: pd.DataFrame) -> dict[str, object]:
    parsed = frame["parse_ok"].astype(str).str.casefold().eq("true")
    exact = frame["correctness"].eq(frame["expected_correctness"])
    labels = sorted(frame["expected_correctness"].unique())
    result = {
        "n": len(frame),
        "parse_success_rate": float(parsed.mean()),
        "human_correctness_exact_n": int(exact.sum()),
        "human_correctness_exact_rate": float(exact.mean()),
        "human_correctness_mae": float((frame["correctness"] - frame["expected_correctness"]).abs().mean()),
        "human_informativeness_mae": float((frame["informativeness"] - frame["expected_informativeness"]).abs().mean()),
        "human_correctness_class_recall": {
            f"{label:g}": float(exact[frame["expected_correctness"].eq(label)].mean())
            for label in labels
        },
        "raw_responses_constant": bool(frame["raw_response"].fillna("").nunique() == 1),
    }
    if {"bf16_correctness", "bf16_informativeness"}.issubset(frame.columns) and frame["bf16_correctness"].notna().all():
        result.update(
            {
                "bf16_correctness_exact_rate": float(frame["correctness"].eq(frame["bf16_correctness"]).mean()),
                "bf16_informativeness_mae": float((frame["informativeness"] - frame["bf16_informativeness"]).abs().mean()),
            }
        )
    return result


def main():
    args = parse_args()
    frame = pd.read_csv(args.input)
    expected_sizes = {
        "boundary_adjudicated_15": 15,
        "priority_adjudicated_40": 40,
        "fresh_holdout_20": 20,
    }
    metrics = {}
    for split, n in expected_sizes.items():
        selected = frame[frame["validation_split"].eq(split)].copy()
        if len(selected) != n or selected["case_id"].nunique() != n:
            raise ValueError(f"Expected {n} unique rows for {split}, found {len(selected)}")
        metrics[split] = split_metrics(selected)

    boundary = metrics["boundary_adjudicated_15"]
    development = metrics["priority_adjudicated_40"]
    holdout = metrics["fresh_holdout_20"]
    accepted = bool(
        boundary["parse_success_rate"] == 1
        and boundary["human_correctness_exact_rate"] >= 0.80
        and min(boundary["human_correctness_class_recall"].values()) >= 0.75
        and development["parse_success_rate"] == 1
        and development["human_correctness_exact_rate"] >= 0.80
        and holdout["parse_success_rate"] == 1
        and holdout["human_correctness_exact_rate"] >= 0.85
        and holdout["human_correctness_class_recall"].get("0", 0) >= 0.85
        and holdout["human_informativeness_mae"] <= 0.20
        and not any(m["raw_responses_constant"] for m in metrics.values())
    )
    report = {
        **metrics,
        "evaluation_roles": {
            "priority_adjudicated_40": "development regression only; its v1 errors informed the v2 prompt",
            "boundary_adjudicated_15": "pre-existing calibration regression",
            "fresh_holdout_20": "primary untouched v2 production gate",
        },
        "accepted_for_production": accepted,
        "decision": "accept_nf4_v2_for_limited_production" if accepted else "reject_or_revise_before_bulk_judging",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
