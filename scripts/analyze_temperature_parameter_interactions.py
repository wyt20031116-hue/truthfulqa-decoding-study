#!/usr/bin/env python3
"""Formal temperature-by-top-p/top-k interaction analysis.

Uses prompt-setting means, prompt fixed effects, and CR1 standard errors
clustered by prompt. Temperature and truncation parameters are categorical.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f, t


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/pure_pk_main_25rep_revised/tables/prompt_setting_means.csv"
OUT = ROOT / "analysis/temperature_parameter_interactions"

METRICS = [
    "correctness",
    "strict_accuracy",
    "informativeness",
    "within_answer_distinct_1",
    "within_answer_distinct_2",
    "trigram_repetition_rate",
    "model_generation_latency_seconds",
    "tokens_per_second",
    "word_length",
    "stopped_early_on_eos",
    "unique_answer_rate",
    "pairwise_token_jaccard_distance",
    "corpus_distinct_1",
]

DIRECT_METRICS = {
    "within_answer_distinct_1",
    "within_answer_distinct_2",
    "trigram_repetition_rate",
    "model_generation_latency_seconds",
    "tokens_per_second",
    "word_length",
    "stopped_early_on_eos",
    "unique_answer_rate",
    "pairwise_token_jaccard_distance",
    "corpus_distinct_1",
    "corpus_distinct_2",
    "semantic_cosine_distance",
    "semantic_near_duplicate_rate",
}


def design_matrix(df: pd.DataFrame, parameter: str, interaction: bool) -> tuple[np.ndarray, list[str]]:
    temps = sorted(df["temperature"].unique())
    params = sorted(df[parameter].unique())
    prompts = sorted(df["prompt_id"].unique())
    cols = [np.ones(len(df))]
    names = ["intercept"]
    for q in prompts[1:]:
        cols.append((df["prompt_id"].to_numpy() == q).astype(float))
        names.append(f"prompt[{q}]")
    for temp in temps[1:]:
        cols.append((df["temperature"].to_numpy() == temp).astype(float))
        names.append(f"temperature[{temp}]")
    for value in params[1:]:
        cols.append((df[parameter].to_numpy() == value).astype(float))
        names.append(f"{parameter}[{value}]")
    if interaction:
        for temp in temps[1:]:
            for value in params[1:]:
                cols.append(
                    (
                        (df["temperature"].to_numpy() == temp)
                        & (df[parameter].to_numpy() == value)
                    ).astype(float)
                )
                names.append(f"temperature[{temp}]:{parameter}[{value}]")
    return np.column_stack(cols), names


def ols_cluster(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique = np.unique(clusters)
    for cluster in unique:
        mask = clusters == cluster
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(y), x.shape[1], len(unique)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    return beta, covariance, residual, float(residual @ residual)


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1)
    return pd.Series(out, index=pvalues.index)


def joint_test(df: pd.DataFrame, metric: str, parameter: str) -> dict:
    y = df[metric].to_numpy(float)
    clusters = df["prompt_id"].to_numpy()
    x_full, names = design_matrix(df, parameter, interaction=True)
    x_reduced, _ = design_matrix(df, parameter, interaction=False)
    beta, cov, _, sse_full = ols_cluster(y, x_full, clusters)
    _, _, _, sse_reduced = ols_cluster(y, x_reduced, clusters)
    idx = [i for i, name in enumerate(names) if ":" in name]
    b = beta[idx]
    v = cov[np.ix_(idx, idx)]
    rank = int(np.linalg.matrix_rank(v))
    if rank == 0:
        stat = 0.0
        pvalue = 1.0
    else:
        v_inv = np.linalg.pinv(v)
        stat = float(b.T @ v_inv @ b)
        pvalue = float(f.sf(stat / rank, rank, len(np.unique(clusters)) - 1))
    partial_r2 = max(0.0, float((sse_reduced - sse_full) / sse_reduced)) if sse_reduced else 0.0
    return {
        "metric": metric,
        "metric_source": "direct_generation" if metric in DIRECT_METRICS else "provisional_qwen3_nf4",
        "interaction": f"temperature_x_{parameter}",
        "n_prompt_settings": len(df),
        "n_prompts": df["prompt_id"].nunique(),
        "wald_chi2": stat,
        "wald_f": stat / rank if rank else 0.0,
        "df": rank,
        "denominator_df": len(np.unique(clusters)) - 1,
        "p_value": pvalue,
        "partial_r2": partial_r2,
    }


def extreme_temperature_did(df: pd.DataFrame, metric: str, parameter: str) -> list[dict]:
    low, high = 0.1, 1.5
    values = sorted(df[parameter].unique())
    baseline = values[0]
    pivot = df.pivot_table(index="prompt_id", columns=["temperature", parameter], values=metric)
    rows = []
    for value in values[1:]:
        did = (
            (pivot[(high, value)] - pivot[(low, value)])
            - (pivot[(high, baseline)] - pivot[(low, baseline)])
        )
        mean = float(did.mean())
        se = float(did.std(ddof=1) / np.sqrt(len(did)))
        crit = float(t.ppf(0.975, len(did) - 1))
        rows.append(
            {
                "metric": metric,
                "metric_source": "direct_generation" if metric in DIRECT_METRICS else "provisional_qwen3_nf4",
                "parameter": parameter,
                "parameter_value": value,
                "baseline_parameter_value": baseline,
                "temperature_low": low,
                "temperature_high": high,
                "difference_in_differences": mean,
                "se_across_prompts": se,
                "ci95_low": mean - crit * se,
                "ci95_high": mean + crit * se,
                "n_prompts": len(did),
            }
        )
    return rows


def plot_did(did: pd.DataFrame, parameter: str) -> None:
    labels = {
        "unique_answer_rate": "Unique-answer rate",
        "pairwise_token_jaccard_distance": "Pairwise token distance",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric in zip(axes, labels):
        g = did[(did["parameter"] == parameter) & (did["metric"] == metric)]
        x = g["parameter_value"].astype(float).to_numpy()
        y = g["difference_in_differences"].to_numpy()
        lo = y - g["ci95_low"].to_numpy()
        hi = g["ci95_high"].to_numpy() - y
        ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o-", color="#1f5a94", capsize=4)
        ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
        ax.set_title(labels[metric])
        ax.set_xlabel(parameter.replace("_", "-"))
        ax.set_ylabel("Interaction contrast\n(high-low temperature amplification)")
        ax.grid(alpha=0.22)
    fig.suptitle(f"Temperature 1.5 vs 0.1 interaction with {parameter.replace('_', '-')}")
    fig.tight_layout()
    stem = OUT / "figures" / f"temperature_x_{parameter}_extreme_did"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    data = pd.read_csv(INPUT)
    expected = {"pure_top_p": 1000, "pure_top_k": 1250}
    if data.groupby("decoding_mode").size().to_dict() != expected:
        raise ValueError("Unexpected prompt-setting design")
    tests = []
    did_rows = []
    for mode, parameter in [("pure_top_p", "top_p"), ("pure_top_k", "top_k")]:
        subset = data[data["decoding_mode"] == mode].copy()
        for metric in METRICS:
            tests.append(joint_test(subset, metric, parameter))
            did_rows.extend(extreme_temperature_did(subset, metric, parameter))
    tests = pd.DataFrame(tests)
    tests["p_value_bh_within_family"] = tests.groupby("interaction")["p_value"].transform(bh_adjust)
    tests["significant_bh_0_05"] = tests["p_value_bh_within_family"] < 0.05
    did = pd.DataFrame(did_rows)
    tests.to_csv(OUT / "interaction_joint_tests.csv", index=False)
    did.to_csv(OUT / "extreme_temperature_difference_in_differences.csv", index=False)
    for parameter in ["top_p", "top_k"]:
        plot_did(did, parameter)
    manifest = {
        "input": str(INPUT),
        "analysis_grain": "50 prompt means per decoding setting; each mean averages 25 repetitions",
        "model": "OLS with categorical temperature, categorical top-p or top-k, their full interaction, and prompt fixed effects",
        "uncertainty": "CR1 sandwich covariance clustered by prompt for joint Wald tests; t intervals across 50 prompt-level difference-in-differences",
        "multiple_testing": "Benjamini-Hochberg correction across 14 metrics within each interaction family",
        "interpretation": "Direct-generation metrics are primary. Correctness and informativeness remain provisional Qwen3-NF4 outcomes.",
        "excluded_metric": "corpus_distinct_2 was excluded because it is undefined for 283 prompt-settings with insufficient bigrams; corpus_distinct_1, exact uniqueness, and pairwise token distance remain available.",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(tests[["interaction", "metric", "p_value_bh_within_family", "partial_r2", "significant_bh_0_05"]].to_string(index=False))


if __name__ == "__main__":
    main()
