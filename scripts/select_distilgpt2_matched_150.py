#!/usr/bin/env python3
"""Select a 15-question x 10-repeat DistilGPT2 comparison set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    matched = frame.loc[
        frame["temperature"].eq(0.7)
        & frame["top_p"].eq(0.9)
        & frame["top_k"].eq(10)
    ].copy()
    if len(matched) != 150:
        raise RuntimeError(f"Expected 150 matched DistilGPT2 rows, found {len(matched)}")
    if matched["question"].nunique() != 15:
        raise RuntimeError("Expected 15 DistilGPT2 questions")
    if not matched.groupby("question").size().eq(10).all():
        raise RuntimeError("Expected 10 DistilGPT2 repetitions per question")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output, index=False)
    print(f"Saved 150 matched DistilGPT2 rows to {output}")


if __name__ == "__main__":
    main()
