#!/usr/bin/env python3
"""Fail closed unless the full pure-p/pure-k judgment artifact is complete."""

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
        "case_id",
        "experiment_id",
        "prompt_id",
        "repetition_id",
        "random_seed",
        "decoding_mode",
        "temperature",
        "top_p",
        "top_k",
        "question",
        "correct_answers",
        "incorrect_answers",
        "generated_text",
    ]
    payload = "\n".join(str(row.get(column, "")) for column in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.input)
    judged_path = Path(args.judgments)
    source = pd.read_csv(source_path)
    judged = pd.read_csv(judged_path)
    expected_hashes = set(source.apply(source_id, axis=1))
    actual_hashes = set(judged["source_row_sha256"])

    expected_modes = {"pure_top_p": 25_000, "pure_top_k": 31_250}
    actual_modes = judged["decoding_mode"].value_counts().to_dict()
    checks = {
        "source_rows": len(source) == 56_250,
        "judgment_rows": len(judged) == 56_250,
        "unique_source_hashes": judged["source_row_sha256"].nunique() == 56_250,
        "source_rows_match_exactly": actual_hashes == expected_hashes,
        "all_parsed": bool(judged["parse_ok"].eq(True).all()),
        "correctness_domain": bool(judged["correctness"].isin([0, 0.5, 1]).all()),
        "informativeness_domain": bool(judged["informativeness"].isin([0, 0.25, 0.5, 0.75, 1]).all()),
        "mode_counts": actual_modes == expected_modes,
        "prompts": judged["prompt_id"].nunique() == 50,
        "repetitions": judged["repetition_id"].nunique() == 25,
        "settings": judged[["decoding_mode", "temperature", "top_p", "top_k"]].drop_duplicates().shape[0] == 45,
        "raw_responses_nonconstant": judged["raw_response"].nunique() > 1,
    }
    report = {
        "provisional_judge": True,
        "reporting_requirement": "Disclose that Qwen3-32B NF4 did not pass every hard human-audit gate.",
        "source_sha256": sha256(source_path),
        "judgments_sha256": sha256(judged_path),
        "n_rows": len(judged),
        "mode_counts": actual_modes,
        "parse_failure_n": int((~judged["parse_ok"].eq(True)).sum()),
        "checks": checks,
        "valid": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit("Full-run validation failed; analysis was not authorized.")


if __name__ == "__main__":
    main()
