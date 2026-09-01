#!/usr/bin/env python3
"""Integrate the 200-row stratified human audit with full pure-p/k results.

The audit is used to estimate a mode-by-temperature additive judge bias.  The
result is a sensitivity analysis, not a replacement for row-level human labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/pure_pk_main_25rep_revised/tables/marginal_effect_summary.csv"
AUDIT = ROOT / "data/human_audit_item_level_200.csv"
OUT = ROOT / "analysis/audit_adjusted_sensitivity"


def weighted_mean_se(values: pd.Series, weights: pd.Series) -> tuple[float, float, float]:
    v = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    mean = float(np.average(v, weights=w))
    sw = w.sum()
    sw2 = np.square(w).sum()
    n_eff = float(sw * sw / sw2)
    denom = sw - sw2 / sw
    if len(v) < 2 or denom <= 0 or n_eff <= 1:
        return mean, float("nan"), n_eff
    var = float(np.sum(w * np.square(v - mean)) / denom)
    return mean, float(np.sqrt(var / n_eff)), n_eff


def audit_adjustments(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, temp), g in audit.groupby(["decoding_mode", "temperature"], sort=True):
        for metric, judge_col, human_col in [
            ("correctness", "correctness", "human_correctness"),
            ("informativeness", "informativeness", "human_informativeness"),
        ]:
            delta = g[human_col] - g[judge_col]
            adj, se, n_eff = weighted_mean_se(delta, g["audit_weight"])
            rows.append(
                {
                    "decoding_mode": mode,
                    "temperature": float(temp),
                    "metric": metric,
                    "audit_n": len(g),
                    "audit_effective_n": n_eff,
                    "audit_adjustment": adj,
                    "audit_adjustment_se": se,
                    "audit_adjustment_ci95_low": adj - 1.96 * se,
                    "audit_adjustment_ci95_high": adj + 1.96 * se,
                }
            )
    return pd.DataFrame(rows)


def build_estimates(full: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    base = full[
        (full["dimension"] == "temperature")
        & full["metric"].isin(["correctness", "strict_accuracy", "informativeness"])
    ].copy()
    base = base.rename(
        columns={
            "value": "temperature",
            "mean": "full_judge_mean",
            "ci95_low": "full_judge_ci95_low",
            "ci95_high": "full_judge_ci95_high",
        }
    )
    strict = base[base["metric"] == "strict_accuracy"][
        ["decoding_mode", "temperature", "full_judge_mean"]
    ].rename(columns={"full_judge_mean": "conservative_correctness_05_to_0"})
    estimates = base[base["metric"].isin(["correctness", "informativeness"])].merge(
        adjustments, on=["decoding_mode", "temperature", "metric"], validate="one_to_one"
    )
    estimates = estimates.merge(strict, on=["decoding_mode", "temperature"], how="left")
    main_se = (
        estimates["full_judge_ci95_high"] - estimates["full_judge_ci95_low"]
    ) / (2 * 1.96)
    estimates["audit_adjusted_mean"] = (
        estimates["full_judge_mean"] + estimates["audit_adjustment"]
    ).clip(0, 1)
    estimates["audit_adjusted_se_approx"] = np.sqrt(
        np.square(main_se) + np.square(estimates["audit_adjustment_se"])
    )
    estimates["audit_adjusted_ci95_low"] = (
        estimates["audit_adjusted_mean"] - 1.96 * estimates["audit_adjusted_se_approx"]
    ).clip(0, 1)
    estimates["audit_adjusted_ci95_high"] = (
        estimates["audit_adjusted_mean"] + 1.96 * estimates["audit_adjusted_se_approx"]
    ).clip(0, 1)
    estimates.loc[estimates["metric"] != "correctness", "conservative_correctness_05_to_0"] = np.nan
    return estimates.sort_values(["decoding_mode", "metric", "temperature"])


def build_contrasts(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, metric), g in estimates.groupby(["decoding_mode", "metric"]):
        g = g.set_index("temperature")
        ref = g.loc[0.1]
        for temp, row in g.drop(index=0.1).iterrows():
            raw = row["full_judge_mean"] - ref["full_judge_mean"]
            adjusted = row["audit_adjusted_mean"] - ref["audit_adjusted_mean"]
            rows.append(
                {
                    "decoding_mode": mode,
                    "metric": metric,
                    "temperature": temp,
                    "reference_temperature": 0.1,
                    "raw_difference": raw,
                    "audit_adjusted_difference": adjusted,
                    "direction_preserved": bool(np.sign(raw) == np.sign(adjusted)),
                }
            )
    return pd.DataFrame(rows)


def plot_estimates(estimates: pd.DataFrame, mode: str, metric: str) -> None:
    g = estimates[(estimates["decoding_mode"] == mode) & (estimates["metric"] == metric)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(g["temperature"], g["full_judge_mean"], "o-", label="Qwen3-NF4 raw")
    ax.plot(g["temperature"], g["audit_adjusted_mean"], "s--", label="Audit-adjusted")
    if metric == "correctness":
        ax.plot(
            g["temperature"],
            g["conservative_correctness_05_to_0"],
            "^:",
            label="Conservative (0.5 recoded to 0)",
        )
    ax.set(xlabel="Temperature", ylabel=metric.capitalize(), ylim=(0, 1))
    label = "pure top-p" if mode == "pure_top_p" else "pure top-k"
    ax.set_title(f"{metric.capitalize()} sensitivity: {label}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = OUT / "figures" / f"{mode}_{metric}_audit_sensitivity"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    full = pd.read_csv(FULL)
    audit = pd.read_csv(AUDIT, low_memory=False)
    required = {
        "decoding_mode", "temperature", "correctness", "informativeness",
        "human_correctness", "human_informativeness", "audit_weight",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Audit is missing columns: {sorted(missing)}")
    if len(audit) != 200:
        raise ValueError(f"Expected 200 audit rows, found {len(audit)}")
    adjustments = audit_adjustments(audit)
    estimates = build_estimates(full, adjustments)
    contrasts = build_contrasts(estimates)
    adjustments.to_csv(OUT / "audit_adjustments_by_mode_temperature.csv", index=False)
    estimates.to_csv(OUT / "full_results_with_audit_sensitivity.csv", index=False)
    contrasts.to_csv(OUT / "temperature_direction_sensitivity.csv", index=False)
    for mode in ["pure_top_p", "pure_top_k"]:
        for metric in ["correctness", "informativeness"]:
            plot_estimates(estimates, mode, metric)
    manifest = {
        "full_results": str(FULL),
        "human_audit": str(AUDIT),
        "audit_rows": int(len(audit)),
        "audit_cells": int(adjustments[["decoding_mode", "temperature"]].drop_duplicates().shape[0]),
        "method": "Add the inverse-probability-weighted mean human-minus-judge discrepancy within each mode-temperature cell to the full-run Qwen3-NF4 mean.",
        "uncertainty": "Approximate 95% CI combines the across-prompt full-run SE and the weighted audit-adjustment SE in quadrature; bounded metrics are clipped to [0,1].",
        "conservative_correctness": "Full-run strict accuracy, equivalent to recoding every Qwen3 correctness=0.5 judgment as 0.",
        "limitations": [
            "Only 20 audited items are available in each mode-temperature cell.",
            "The audit was deliberately stratified toward 0.5 and known error patterns; weights correct the sampling design but do not eliminate model or reviewer error.",
            "Audit-adjusted estimates are sensitivity estimates, not fully human-labelled population means.",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "rows": len(estimates), "contrasts": len(contrasts)}, indent=2))


if __name__ == "__main__":
    main()
