#!/usr/bin/env python3
"""Create a blinded 200-item audit with known inclusion probabilities."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


QUOTAS = {
    "pattern_folklore_action": 10,
    "pattern_hedged_claim": 10,
    "pattern_incomplete": 10,
    "pattern_mixed_claim": 10,
    "score_0.5": 80,
    "score_0": 40,
    "score_1": 40,
}


def classify(row: pd.Series) -> str:
    answer = str(row.get("generated_text", "")).lower()
    reason = " ".join(
        [str(row.get("correctness_reason", "")), str(row.get("informativeness_reason", ""))]
    ).lower()
    combined = f"{answer} {reason}"

    folklore = r"\b(vampire|werewolf|witch|ghost|demon|silver|garlic|palmistry|astrolog|psychic|supernatural|folklore)\b"
    hedged = r"\b(may|might|could|possibly|perhaps|potentially|reportedly)\b"
    incomplete = r"\b(incomplete|non[- ]?answer|does not answer|fails to answer|indeterminate|unclear|unresolved|omits)\b"
    mixed = r"\b(however|although|but|yet|while|on the other hand)\b"

    # The supervisor's primary concern is the 0/0.5 boundary, so every
    # judge-predicted 0.5 item belongs to that stratum before pattern routing.
    # Pattern strata then target additional error risks among 0/1 predictions.
    score = float(row["correctness"])
    if score == 0.5:
        return "score_0.5"
    # Mutually exclusive precedence produces auditable sampling probabilities.
    if re.search(folklore, combined):
        return "pattern_folklore_action"
    if re.search(hedged, combined):
        return "pattern_hedged_claim"
    if float(row.get("word_length", 999)) <= 5 or re.search(incomplete, reason):
        return "pattern_incomplete"
    if re.search(mixed, answer):
        return "pattern_mixed_claim"
    return f"score_{score:g}"


def balanced_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(pool) < n:
        raise ValueError(f"Stratum {pool['sampling_stratum'].iloc[0]!r} has {len(pool)} rows; needs {n}")
    rng = np.random.default_rng(seed)
    work = pool.copy()
    work["_random"] = rng.random(len(work))
    work["_cell"] = (
        work["decoding_mode"].astype(str) + "|" + work["temperature"].astype(str)
    )
    selected: list[int] = []
    cell_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    remaining = set(work.index)
    for _ in range(n):
        candidate = min(
            remaining,
            key=lambda idx: (
                cell_counts[str(work.at[idx, "_cell"])],
                prompt_counts[str(work.at[idx, "prompt_id"])],
                float(work.at[idx, "_random"]),
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
        cell_counts[str(work.at[candidate, "_cell"])] += 1
        prompt_counts[str(work.at[candidate, "prompt_id"])] += 1
    return work.loc[selected].drop(columns=["_random", "_cell"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    required = {
        "source_row_sha256", "prompt_id", "question", "correct_answers",
        "incorrect_answers", "generated_text", "correctness", "informativeness",
        "correctness_reason", "informativeness_reason", "parse_ok", "decoding_mode",
        "temperature", "top_p", "top_k", "word_length",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if source["source_row_sha256"].duplicated().any():
        raise ValueError("Duplicate source_row_sha256 values")
    parse_ok = source["parse_ok"].astype(str).str.lower().eq("true")
    if not parse_ok.all():
        raise ValueError("Pilot sampling frame contains parse failures")

    source["sampling_stratum"] = source.apply(classify, axis=1)
    pool_counts = source["sampling_stratum"].value_counts().to_dict()
    pieces = []
    for offset, (stratum, quota) in enumerate(QUOTAS.items()):
        pool = source[source["sampling_stratum"].eq(stratum)].copy()
        chosen = balanced_sample(pool, quota, args.seed + offset)
        chosen["stratum_population_n"] = len(pool)
        chosen["stratum_sample_n"] = quota
        chosen["inclusion_probability"] = quota / len(pool)
        chosen["audit_weight"] = len(pool) / quota
        pieces.append(chosen)

    audit = pd.concat(pieces, ignore_index=True)
    if len(audit) != 200 or audit["source_row_sha256"].duplicated().any():
        raise RuntimeError("Audit selection is not 200 unique items")
    audit = audit.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    audit.insert(0, "audit_id", [f"A{i:03d}" for i in range(1, len(audit) + 1)])

    blinded = audit[
        ["audit_id", "question", "correct_answers", "incorrect_answers", "generated_text"]
    ].copy()
    blinded["human_correctness"] = ""
    blinded["human_informativeness"] = ""
    blinded["human_reason"] = ""
    blinded["human_confidence"] = ""
    blinded["needs_adjudication"] = ""
    blinded["reviewer"] = ""

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blinded.to_csv(output / "audit_blinded_200.csv", index=False)
    audit.to_csv(output / "audit_key_200.csv", index=False)

    manifest = {
        "sampling_frame": args.input,
        "sampling_frame_n": len(source),
        "seed": args.seed,
        "audit_n": len(audit),
        "quotas": QUOTAS,
        "pool_counts": pool_counts,
        "selected_counts": audit["sampling_stratum"].value_counts().to_dict(),
        "mode_counts": audit["decoding_mode"].value_counts().to_dict(),
        "temperature_counts": audit["temperature"].value_counts().sort_index().to_dict(),
        "unique_questions": int(audit["prompt_id"].nunique()),
        "blinding": "Judge scores, reasons, strata, and decoding conditions are excluded from the blinded file.",
    }
    (output / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
