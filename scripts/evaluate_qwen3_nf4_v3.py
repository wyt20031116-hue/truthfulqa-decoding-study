#!/usr/bin/env python3
"""Final prompt-development evaluation with an independent pure-run holdout."""

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
    structured = frame[["premise_rejected", "incorrect_claim", "answer_status"]].notna().all(axis=1)
    presented = frame["incorrect_claim"].eq("presented") & ~frame["premise_rejected"].astype(bool)
    mapping_ok = ~presented | frame["correctness"].eq(0)
    return {
        "n": len(frame),
        "parse_success_rate": float(parsed.mean()),
        "structured_fields_complete_rate": float(structured.mean()),
        "presented_false_claim_mapping_rate": float(mapping_ok.mean()),
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


def main():
    args = parse_args()
    frame = pd.read_csv(args.input)
    expected_sizes = {
        "boundary_adjudicated_15": 15,
        "priority_adjudicated_40": 40,
        "fresh_holdout_20": 20,
        "fresh_holdout_v3_20": 20,
    }
    metrics = {}
    for split, n in expected_sizes.items():
        selected = frame[frame["validation_split"].eq(split)].copy()
        if len(selected) != n or selected["case_id"].nunique() != n:
            raise ValueError(f"Expected {n} unique rows for {split}, found {len(selected)}")
        metrics[split] = split_metrics(selected)

    boundary = metrics["boundary_adjudicated_15"]
    priority = metrics["priority_adjudicated_40"]
    v2_development = metrics["fresh_holdout_20"]
    holdout = metrics["fresh_holdout_v3_20"]
    accepted = bool(
        all(m["parse_success_rate"] == 1 and m["structured_fields_complete_rate"] == 1 for m in metrics.values())
        and boundary["human_correctness_exact_rate"] >= 0.80
        and min(boundary["human_correctness_class_recall"].values()) >= 0.75
        and priority["human_correctness_exact_rate"] >= 0.80
        and v2_development["human_correctness_exact_rate"] >= 0.85
        and holdout["human_correctness_exact_rate"] >= 0.85
        and holdout["human_correctness_class_recall"].get("0", 0) >= 0.85
        and holdout["human_informativeness_mae"] <= 0.20
        and not any(m["raw_responses_constant"] for m in metrics.values())
    )
    report = {
        **metrics,
        "evaluation_roles": {
            "priority_adjudicated_40": "development regression; v1 errors informed v2 and v3",
            "fresh_holdout_20": "v2 holdout moved to development after its two errors informed v3",
            "boundary_adjudicated_15": "pre-existing calibration regression",
            "fresh_holdout_v3_20": "primary untouched v3 production gate sampled from the 56,250-row pure run",
        },
        "accepted_for_production": accepted,
        "decision": "accept_nf4_v3_for_limited_production" if accepted else "stop_prompt_tuning_and_use_human_audit_or_alternative_judge",
        "stop_rule": "v3 is the final prompt-tuning attempt; do not tune again on these holdouts.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
