#!/usr/bin/env python3
"""Validate a judge on 15 human-labelled 0/0.5/1 boundary cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-class-recall", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {
        "case_id",
        "expected_correctness",
        "correctness",
        "raw_response",
        "parse_ok",
        "judge_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    if len(frame) != 15 or frame["case_id"].nunique() != 15:
        raise ValueError("Calibration must contain exactly 15 unique cases")

    parsed = frame["parse_ok"].astype(str).str.casefold().eq("true")
    exact = frame["correctness"].eq(frame["expected_correctness"])
    labels = sorted(frame["expected_correctness"].dropna().unique())
    if not set(labels).issubset({0, 0.5, 1}):
        raise ValueError(f"Unexpected correctness labels: {labels}")
    recalls = {}
    for label in labels:
        selected = frame["expected_correctness"].eq(label)
        recalls[f"{label:g}"] = float(exact[selected].mean())
    score_counts = {
        str(key): int(value)
        for key, value in frame["correctness"].value_counts(dropna=False).items()
    }
    raw_constant = frame["raw_response"].fillna("").nunique() == 1
    score_constant = frame["correctness"].nunique(dropna=False) == 1
    metrics = {
        "judge_id": str(frame["judge_id"].iloc[0]),
        "n_cases": len(frame),
        "parse_success_rate": float(parsed.mean()),
        "exact_accuracy": float(exact.mean()),
        "class_recall": recalls,
        "score_counts": score_counts,
        "raw_responses_constant": bool(raw_constant),
        "scores_constant": bool(score_constant),
    }
    metrics["accepted"] = bool(
        metrics["parse_success_rate"] == 1
        and metrics["exact_accuracy"] >= args.min_accuracy
        and min(recalls.values()) >= args.min_class_recall
        and not raw_constant
        and not score_constant
    )
    metrics["decision"] = (
        "accept_for_pilot" if metrics["accepted"] else "reject_and_try_next_model"
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
