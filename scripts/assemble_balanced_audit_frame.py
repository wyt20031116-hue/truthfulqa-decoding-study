#!/usr/bin/env python3
"""Assemble and validate the 1,000-row, 50-prompt judged sampling frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True)
    parser.add_argument("--screen-judgments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    existing = pd.read_csv(args.existing)
    screen = pd.read_csv(args.screen_judgments)
    combined = pd.concat([existing, screen], ignore_index=True)
    prompt_counts = combined.groupby("prompt_id").size()
    checks = {
        "rows_1000": len(combined) == 1000,
        "prompts_50": combined["prompt_id"].nunique() == 50,
        "twenty_per_prompt": bool(prompt_counts.eq(20).all()),
        "mode_balance": combined["decoding_mode"].value_counts().to_dict()
        == {"pure_top_p": 500, "pure_top_k": 500},
        "temperature_balance": bool(combined["temperature"].value_counts().eq(200).all()),
        "unique_source_hashes": combined["source_row_sha256"].nunique() == 1000,
        "all_parsed": bool(combined["parse_ok"].astype(str).str.lower().eq("true").all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Balanced frame validation failed: {checks}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    manifest = {
        "n": len(combined),
        "prompt_counts": prompt_counts.sort_index().to_dict(),
        "mode_counts": combined["decoding_mode"].value_counts().to_dict(),
        "temperature_counts": combined["temperature"].value_counts().sort_index().to_dict(),
        "checks": checks,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
