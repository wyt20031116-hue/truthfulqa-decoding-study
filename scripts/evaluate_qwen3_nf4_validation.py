#!/usr/bin/env python3
"""Evaluate NF4 outputs against frozen human labels and the prior BF16 judge."""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    parsed = frame["parse_ok"].astype(str).str.casefold().eq("true")
    exact = frame["correctness"].eq(frame["expected_correctness"])
    bf16_exact = frame["correctness"].eq(frame["bf16_correctness"])
    result = {
        "n": len(frame),
        "parse_success_rate": float(parsed.mean()),
        "human_correctness_exact_n": int(exact.sum()),
        "human_correctness_exact_rate": float(exact.mean()),
        "human_correctness_mae": float((frame["correctness"] - frame["expected_correctness"]).abs().mean()),
        "human_informativeness_mae": float((frame["informativeness"] - frame["expected_informativeness"]).abs().mean()),
        "bf16_correctness_exact_n": int(bf16_exact.sum()),
        "bf16_correctness_exact_rate": float(bf16_exact.mean()),
        "bf16_informativeness_mae": float((frame["informativeness"] - frame["bf16_informativeness"]).abs().mean()),
        "raw_responses_constant": bool(frame["raw_response"].fillna("").nunique() == 1),
    }
    labels = sorted(frame["expected_correctness"].unique())
    result["human_correctness_class_recall"] = {
        f"{label:g}": float(exact[frame["expected_correctness"].eq(label)].mean())
        for label in labels
    }
    return result


def main():
    args = parse_args()
    boundary = pd.read_csv(args.boundary)
    priority = pd.read_csv(args.priority)
    if len(boundary) != 15 or boundary["case_id"].nunique() != 15:
        raise ValueError("Boundary output must contain 15 unique cases")
    if len(priority) != 40 or priority["case_id"].nunique() != 40:
        raise ValueError("Priority output must contain 40 unique cases")
    boundary_metrics = metrics(boundary)
    priority_metrics = metrics(priority)
    accepted = bool(
        boundary_metrics["parse_success_rate"] == 1
        and boundary_metrics["human_correctness_exact_rate"] >= 0.80
        and min(boundary_metrics["human_correctness_class_recall"].values()) >= 0.75
        and priority_metrics["parse_success_rate"] == 1
        and priority_metrics["human_correctness_exact_rate"] >= 0.80
        and priority_metrics["human_informativeness_mae"] <= 0.20
        and not boundary_metrics["raw_responses_constant"]
        and not priority_metrics["raw_responses_constant"]
    )
    report = {
        "boundary_adjudicated_15": boundary_metrics,
        "priority_adjudicated_40": priority_metrics,
        "acceptance_thresholds": {
            "parse_success_rate": 1.0,
            "boundary_correctness_exact_rate_min": 0.80,
            "boundary_class_recall_min": 0.75,
            "priority_correctness_exact_rate_min": 0.80,
            "priority_informativeness_mae_max": 0.20,
        },
        "accepted_for_production": accepted,
        "decision": "accept_nf4_for_production_benchmark" if accepted else "reject_or_revise_nf4_prompt_before_bulk_judging",
        "note": "The priority set is deliberately difficult and is evaluated separately from the balanced 15-case boundary set.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
