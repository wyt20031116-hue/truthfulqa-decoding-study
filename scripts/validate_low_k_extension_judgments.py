#!/usr/bin/env python3
"""Fail closed unless the 5,000-row low-k follow-up is completely judged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(row: pd.Series) -> str:
    columns = [
        "case_id", "experiment_id", "prompt_id", "repetition_id", "random_seed",
        "decoding_mode", "temperature", "top_p", "top_k", "question",
        "correct_answers", "incorrect_answers", "generated_text",
    ]
    payload = "\n".join(str(row.get(column, "")) for column in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input, low_memory=False)
    judged = pd.read_csv(args.judgments, low_memory=False)
    expected_hashes = set(source.apply(source_id, axis=1))
    actual_hashes = set(judged["source_row_sha256"])
    cell_counts = judged.groupby(["temperature", "top_k"]).size()

    checks = {
        "source_rows": len(source) == 5_000,
        "judgment_rows": len(judged) == 5_000,
        "unique_source_hashes": judged["source_row_sha256"].nunique() == 5_000,
        "source_rows_match_exactly": actual_hashes == expected_hashes,
        "all_parsed": bool(judged["parse_ok"].eq(True).all()),
        "correctness_domain": bool(judged["correctness"].isin([0, 0.5, 1]).all()),
        "informativeness_domain": bool(
            judged["informativeness"].isin([0, 0.25, 0.5, 0.75, 1]).all()
        ),
        "pure_top_k_only": bool(judged["decoding_mode"].eq("pure_top_k").all()),
        "top_p_disabled": bool(judged["top_p"].eq(1.0).all()),
        "k_values": set(judged["top_k"]) == {1, 2, 5, 10},
        "temperatures": set(judged["temperature"]) == {0.1, 0.3, 0.7, 1.0, 1.5},
        "balanced_cells": len(cell_counts) == 20 and bool(cell_counts.eq(250).all()),
        "prompts": judged["prompt_id"].nunique() == 50,
        "repetitions": set(judged["repetition_id"]) == set(range(5)),
        "raw_responses_nonconstant": judged["raw_response"].nunique() > 1,
    }
    report = {
        "design": "post_hoc_low_k_follow_up",
        "provisional_judge": True,
        "reporting_requirement": (
            "Use finalized human-audit sensitivity analysis and disclose the "
            "Qwen3-NF4 0-versus-0.5 boundary limitation."
        ),
        "source_sha256": sha256(args.input),
        "judgments_sha256": sha256(args.judgments),
        "n_rows": len(judged),
        "parse_failure_n": int((~judged["parse_ok"].eq(True)).sum()),
        "checks": checks,
        "valid": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit("Low-k judgment validation failed; analysis is not authorized.")


if __name__ == "__main__":
    main()
