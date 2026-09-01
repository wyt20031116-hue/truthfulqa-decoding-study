#!/usr/bin/env python3
"""Build a paired, blinded human audit of representative Pareto settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    ("pure_top_p", 0.7, 0.8, 0),
    ("pure_top_k", 1.0, 1.0, 25),
    ("pure_top_k", 1.5, 1.0, 30),
    ("pure_top_p", 1.5, 0.95, 0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/qwen3_nf4_pure_56250/judgments.csv",
    )
    parser.add_argument("--output-dir", default="outputs/frontier_targeted_audit_200")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(source, low_memory=False)

    selected_parts = []
    for mode, temp, p, k in TARGETS:
        mask = (
            x["decoding_mode"].eq(mode)
            & np.isclose(x["temperature"], temp)
            & np.isclose(x["top_p"], p)
            & x["top_k"].eq(k)
        )
        part = x.loc[mask].copy()
        if len(part) != 50 * 25:
            raise ValueError(f"Target {(mode, temp, p, k)} has {len(part)} rows")
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)

    # Draw one repetition index per prompt and use it for all four settings.
    rng = np.random.default_rng(args.seed)
    chosen_rep = {int(pid): int(rng.integers(0, 25)) for pid in sorted(x["prompt_id"].unique())}
    audit = selected[
        selected.apply(lambda r: int(r["repetition_id"]) == chosen_rep[int(r["prompt_id"])], axis=1)
    ].copy()
    if len(audit) != 200 or audit["prompt_id"].nunique() != 50:
        raise ValueError("Expected four settings for each of 50 prompts")

    audit["audit_id"] = [f"F{n:03d}" for n in range(1, len(audit) + 1)]
    # Randomize presentation; audit_id remains the join key.
    audit = audit.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    blind_columns = [
        "audit_id", "question", "correct_answers", "incorrect_answers", "generated_text",
    ]
    blind = audit[blind_columns].copy()
    blind["human_correctness"] = ""
    blind["human_informativeness"] = ""
    blind["human_reason"] = ""
    blind["human_confidence"] = ""
    blind["needs_adjudication"] = ""
    blind["reviewer"] = ""
    blind["review_status"] = ""
    blind["reviewer_notes"] = ""
    blind.to_csv(out / "audit_blinded_200.csv", index=False)

    key_columns = [
        "audit_id", "source_row_sha256", "prompt_id", "repetition_id", "random_seed",
        "decoding_mode", "temperature", "top_p", "top_k", "question", "generated_text",
        "correctness", "informativeness", "correctness_reason", "informativeness_reason",
        "model_generation_latency_seconds",
    ]
    audit[key_columns].to_csv(out / "audit_key_200.csv", index=False)

    counts = (
        audit.groupby(["decoding_mode", "temperature", "top_p", "top_k"], dropna=False)
        .size().rename("n").reset_index().to_dict("records")
    )
    manifest = {
        "input": str(source),
        "seed": args.seed,
        "audit_n": len(audit),
        "unique_prompts": int(audit["prompt_id"].nunique()),
        "design": "paired: one common repetition index per prompt across four settings",
        "target_counts": counts,
        "blinding": "decoding conditions and all Qwen3-NF4 scores/reasons excluded",
        "required_human_fields": {
            "correctness": [0, 0.5, 1],
            "informativeness": [0, 0.25, 0.5, 0.75, 1],
        },
    }
    (out / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
