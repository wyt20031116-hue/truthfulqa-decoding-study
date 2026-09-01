#!/usr/bin/env python3
"""Build a 50-prompt balanced screen while reusing complete partial judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_PER_PROMPT = 45 * 25
SEED = 20260802


def select_twenty(frame: pd.DataFrame, prompt_id: object, seed: int) -> pd.DataFrame:
    prompt = frame[frame["prompt_id"].eq(prompt_id)].copy()
    pieces = []
    for offset, ((mode, temperature), cell) in enumerate(
        prompt.groupby(["decoding_mode", "temperature"], observed=True, sort=True)
    ):
        if len(cell) < 2:
            raise ValueError(f"Prompt {prompt_id}, {mode}, T={temperature} has only {len(cell)} rows")
        pieces.append(cell.sample(n=2, random_state=seed + offset))
    selected = pd.concat(pieces, ignore_index=True)
    if len(selected) != 20:
        raise ValueError(f"Prompt {prompt_id} produced {len(selected)} rows instead of 20")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--partial-judgments", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    generations = pd.read_csv(args.generations)
    partial = pd.read_csv(args.partial_judgments)
    if len(generations) != 56_250 or generations["prompt_id"].nunique() != 50:
        raise ValueError("Expected complete 56,250-row, 50-prompt generation file")
    counts = partial.groupby("prompt_id").size()
    reusable_prompts = sorted(counts[counts.eq(EXPECTED_PER_PROMPT)].index.tolist())
    all_prompts = sorted(generations["prompt_id"].unique().tolist())
    screen_prompts = [prompt for prompt in all_prompts if prompt not in reusable_prompts]

    existing = pd.concat(
        [select_twenty(partial, prompt, SEED + int(prompt) * 100) for prompt in reusable_prompts],
        ignore_index=True,
    )
    screen = pd.concat(
        [select_twenty(generations, prompt, SEED + int(prompt) * 100) for prompt in screen_prompts],
        ignore_index=True,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    existing.to_csv(output / "existing_judged_balanced.csv", index=False)
    screen.to_csv(output / "screen_to_judge.csv", index=False)
    manifest = {
        "seed": SEED,
        "reusable_complete_prompts": reusable_prompts,
        "screen_prompts": screen_prompts,
        "existing_judged_n": len(existing),
        "screen_to_judge_n": len(screen),
        "target_balanced_frame_n": len(existing) + len(screen),
        "selection": "2 rows per prompt x decoding_mode x temperature cell; 20 rows per prompt.",
    }
    (output / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
