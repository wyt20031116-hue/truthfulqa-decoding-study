#!/usr/bin/env python3
"""Compare finalized human labels with Qwen3-NF4 on the frontier audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [0.0, 0.5, 1.0]


def cluster_bootstrap_exact(
    frame: pd.DataFrame, *, clusters: str, seed: int = 20260804, draws: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = {key: group for key, group in frame.groupby(clusters, sort=True)}
    keys = np.array(list(groups))
    estimates = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        values = np.concatenate(
            [groups[key]["correctness_exact"].to_numpy(dtype=float) for key in sampled]
        )
        estimates[draw] = values.mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human = pd.read_excel(args.workbook, sheet_name="Audit Review")
    key = pd.read_csv(args.key, low_memory=False)

    # Grain and integrity gates before computing agreement.
    assert len(human) == 200 and len(key) == 200
    assert human["audit_id"].notna().all() and key["audit_id"].notna().all()
    assert human["audit_id"].is_unique and key["audit_id"].is_unique
    assert set(human["audit_id"]) == set(key["audit_id"])
    assert human["review_status"].eq("Human reviewed").all()
    for column, allowed in [
        ("human_correctness", LABELS),
        ("human_informativeness", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ]:
        numeric = pd.to_numeric(human[column], errors="coerce")
        assert numeric.notna().all()
        assert numeric.isin(allowed).all()
        human[column] = numeric
    assert key["correctness"].isin(LABELS).all()
    assert key["informativeness"].isin([0.0, 0.25, 0.5, 0.75, 1.0]).all()

    merged = human.merge(key, on="audit_id", suffixes=("_human_file", "_judge_file"), validate="one_to_one")
    for column in ["question", "generated_text"]:
        left = merged[f"{column}_human_file"].fillna("").astype(str).str.strip()
        right = merged[f"{column}_judge_file"].fillna("").astype(str).str.strip()
        assert left.eq(right).all(), f"Text mismatch in {column}"

    merged["human_correctness"] = merged["human_correctness"].astype(float)
    merged["judge_correctness"] = merged["correctness"].astype(float)
    merged["correctness_exact"] = merged["human_correctness"].eq(merged["judge_correctness"])
    merged["correctness_absolute_error"] = (
        merged["judge_correctness"] - merged["human_correctness"]
    ).abs()
    merged["judge_minus_human_correctness"] = (
        merged["judge_correctness"] - merged["human_correctness"]
    )
    merged["informativeness_absolute_error"] = (
        merged["informativeness"].astype(float) - merged["human_informativeness"].astype(float)
    ).abs()

    confusion = pd.crosstab(
        merged["human_correctness"],
        merged["judge_correctness"],
        rownames=["human"],
        colnames=["qwen3_nf4"],
        dropna=False,
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)
    confusion.to_csv(args.output_dir / "correctness_confusion_matrix.csv")

    by_human = []
    for label in LABELS:
        subset = merged[merged["human_correctness"].eq(label)]
        by_human.append(
            {
                "human_correctness": label,
                "n": len(subset),
                "exact_n": int(subset["correctness_exact"].sum()),
                "exact_rate": float(subset["correctness_exact"].mean()),
                "judge_mean": float(subset["judge_correctness"].mean()),
            }
        )
    pd.DataFrame(by_human).to_csv(args.output_dir / "correctness_by_human_class.csv", index=False)

    setting_columns = ["decoding_mode", "temperature", "top_p", "top_k"]
    by_setting = (
        merged.groupby(setting_columns, dropna=False)
        .agg(
            n=("audit_id", "size"),
            exact_rate=("correctness_exact", "mean"),
            correctness_mae=("correctness_absolute_error", "mean"),
            judge_mean_correctness=("judge_correctness", "mean"),
            human_mean_correctness=("human_correctness", "mean"),
            informativeness_mae=("informativeness_absolute_error", "mean"),
        )
        .reset_index()
    )
    by_setting.to_csv(args.output_dir / "agreement_by_setting.csv", index=False)

    ci_low, ci_high = cluster_bootstrap_exact(merged, clusters="prompt_id")
    h0 = merged[merged["human_correctness"].eq(0.0)]
    h05 = merged[merged["human_correctness"].eq(0.5)]
    summary = {
        "scope": "paired 200-row targeted audit of four selected decoding settings",
        "n": len(merged),
        "unique_prompts": int(merged["prompt_id"].nunique()),
        "all_rows_human_reviewed": True,
        "one_to_one_join_and_text_match": True,
        "correctness_exact_n": int(merged["correctness_exact"].sum()),
        "correctness_exact_rate": float(merged["correctness_exact"].mean()),
        "correctness_error_rate": float(1 - merged["correctness_exact"].mean()),
        "correctness_exact_cluster_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "correctness_mae": float(merged["correctness_absolute_error"].mean()),
        "judge_mean_correctness": float(merged["judge_correctness"].mean()),
        "human_mean_correctness": float(merged["human_correctness"].mean()),
        "judge_minus_human_mean_correctness": float(
            merged["judge_correctness"].mean() - merged["human_correctness"].mean()
        ),
        "informativeness_mae": float(merged["informativeness_absolute_error"].mean()),
        "boundary_0_to_0.5_n": int(
            (merged["human_correctness"].eq(0.0) & merged["judge_correctness"].eq(0.5)).sum()
        ),
        "boundary_0_to_0.5_rate_given_human_0": float(
            h0["judge_correctness"].eq(0.5).mean()
        ),
        "boundary_0.5_to_0_n": int(
            (merged["human_correctness"].eq(0.5) & merged["judge_correctness"].eq(0.0)).sum()
        ),
        "boundary_0.5_to_0_rate_given_human_0.5": float(
            h05["judge_correctness"].eq(0.0).mean()
        ),
        "over_score_n": int(merged["judge_minus_human_correctness"].gt(0).sum()),
        "under_score_n": int(merged["judge_minus_human_correctness"].lt(0).sum()),
        "interpretation_limit": (
            "Descriptive error estimate for the four deliberately selected frontier settings; "
            "not an unweighted population error rate for all 56,250 generations."
        ),
    }
    with open(args.output_dir / "agreement_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    columns = [
        "audit_id", "source_row_sha256", "prompt_id", "decoding_mode", "temperature",
        "top_p", "top_k", "human_correctness", "judge_correctness",
        "correctness_exact", "correctness_absolute_error",
        "human_informativeness", "informativeness", "informativeness_absolute_error",
        "human_confidence", "needs_adjudication", "review_status",
    ]
    merged[columns].to_csv(args.output_dir / "item_level_agreement.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
