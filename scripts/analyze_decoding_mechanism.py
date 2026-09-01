#!/usr/bin/env python3
"""Analyze the token-distribution mechanism experiment.

Five repetitions are first summarized within each question-setting cell.  All
confidence intervals, correlations, and regressions then treat the 50 questions
as clusters; token steps are never treated as independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MECHANISMS = [
    "mean_token_entropy_nats",
    "mean_effective_support_size",
    "mean_retained_support_size",
    "mean_max_token_probability",
]
DIVERSITY = ["unique_answer_rate", "pairwise_token_jaccard_distance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="runs/mechanism_50q_21settings_5rep/generations_with_mechanism.csv",
    )
    parser.add_argument(
        "--token-steps",
        default="runs/mechanism_50q_21settings_5rep/token_steps.csv",
    )
    parser.add_argument("--output-dir", default="analysis/decoding_mechanism")
    return parser.parse_args()


def normalize_tokens(text: object) -> set[str]:
    return set(re.findall(r"\b\w+\b", str(text).lower()))


def pairwise_jaccard_distance(texts: pd.Series) -> float:
    token_sets = [normalize_tokens(text) for text in texts]
    distances = []
    for left, right in combinations(token_sets, 2):
        union = left | right
        distances.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    return float(np.mean(distances)) if distances else 0.0


def normal_summary(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "se": se,
                    "ci95_low": mean - 1.96 * se,
                    "ci95_high": mean + 1.96 * se,
                    "n_questions": len(values),
                }
            )
    return pd.DataFrame(rows)


def design_matrix(
    frame: pd.DataFrame,
    categorical: list[str],
    continuous: list[str] | None = None,
    interaction: tuple[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a treatment-coded matrix with an intercept."""
    continuous = continuous or []
    pieces = [np.ones((len(frame), 1), dtype=float)]
    names = ["Intercept"]
    dummy_blocks: dict[str, tuple[np.ndarray, list[str]]] = {}
    for column in categorical:
        levels = sorted(frame[column].dropna().unique())
        matrix = np.column_stack(
            [frame[column].eq(level).to_numpy(float) for level in levels[1:]]
        ) if len(levels) > 1 else np.empty((len(frame), 0))
        block_names = [f"{column}[{level:g}]" for level in levels[1:]]
        pieces.append(matrix)
        names.extend(block_names)
        dummy_blocks[column] = (matrix, block_names)
    for column in continuous:
        pieces.append(frame[[column]].to_numpy(float))
        names.append(column)
    if interaction is not None:
        left_matrix, left_names = dummy_blocks[interaction[0]]
        right_matrix, right_names = dummy_blocks[interaction[1]]
        interaction_columns = []
        for left_index, left_name in enumerate(left_names):
            for right_index, right_name in enumerate(right_names):
                interaction_columns.append(
                    (left_matrix[:, left_index] * right_matrix[:, right_index])[:, None]
                )
                names.append(f"{left_name}:{right_name}")
        if interaction_columns:
            pieces.append(np.hstack(interaction_columns))
    return np.hstack(pieces), names


def cluster_ols(
    y: np.ndarray, X: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS with a question-clustered CR1 sandwich covariance matrix."""
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        selected = clusters == cluster
        score = X[selected].T @ residual[selected]
        meat += np.outer(score, score)
    n, k = X.shape
    g = len(unique_clusters)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    # Two-sided normal approximation; 50 question clusters are available.
    p = np.array([math.erfc(abs(value) / math.sqrt(2)) for value in z])
    return beta, se, p, beta - 1.96 * se, beta + 1.96 * se


def clustered_models(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, parameter in (("pure_top_p", "top_p"), ("pure_top_k", "top_k")):
        arm = cells.loc[cells.decoding_mode.eq(mode)].copy()
        arm[parameter] = arm[parameter].astype(float)
        for outcome in MECHANISMS + DIVERSITY:
            X, terms = design_matrix(
                arm,
                categorical=["temperature", parameter],
                interaction=("temperature", parameter),
            )
            estimates, ses, p_values, lows, highs = cluster_ols(
                arm[outcome].to_numpy(float), X, arm.prompt_id.to_numpy()
            )
            formula = f"{outcome} ~ categorical temperature * categorical {parameter}"
            for index, term in enumerate(terms):
                rows.append(
                    {
                        "decoding_mode": mode,
                        "outcome": outcome,
                        "formula": formula,
                        "term": term,
                        "estimate": estimates[index],
                        "cluster_se": ses[index],
                        "p_value": p_values[index],
                        "ci95_low": lows[index],
                        "ci95_high": highs[index],
                        "n_cells": len(arm),
                        "n_question_clusters": int(arm.prompt_id.nunique()),
                    }
                )
    return pd.DataFrame(rows)


def pathway_models(
    cells: pd.DataFrame, *, include_question_fixed_effects: bool
) -> pd.DataFrame:
    """Estimate conditional associations, not causal mediation effects.

    The primary specification includes question fixed effects. Clustering by
    question is retained because fixed effects remove between-question level
    differences but do not make observations within a question independent.
    """
    rows = []
    for mode, parameter in (("pure_top_p", "top_p"), ("pure_top_k", "top_k")):
        arm = cells.loc[cells.decoding_mode.eq(mode)].copy()
        arm[parameter] = arm[parameter].astype(float)
        for outcome in DIVERSITY:
            for mechanism in MECHANISMS:
                standardized_mechanism = f"z_{mechanism}"
                arm[standardized_mechanism] = (
                    arm[mechanism] - arm[mechanism].mean()
                ) / arm[mechanism].std(ddof=1)
                standardized_outcome = f"z_{outcome}"
                arm[standardized_outcome] = (
                    arm[outcome] - arm[outcome].mean()
                ) / arm[outcome].std(ddof=1)
                categorical = ["temperature", parameter]
                if include_question_fixed_effects:
                    categorical = ["prompt_id", *categorical]
                X, terms = design_matrix(
                    arm,
                    categorical=categorical,
                    continuous=[standardized_mechanism],
                )
                estimates, ses, p_values, lows, highs = cluster_ols(
                    arm[standardized_outcome].to_numpy(float),
                    X,
                    arm.prompt_id.to_numpy(),
                )
                index = terms.index(standardized_mechanism)
                formula = (
                    f"standardized {outcome} ~ standardized {mechanism} + "
                    f"categorical temperature + categorical {parameter}"
                )
                if include_question_fixed_effects:
                    formula += " + question fixed effects"
                rows.append(
                    {
                        "decoding_mode": mode,
                        "diversity_outcome": outcome,
                        "mechanism": mechanism,
                        "formula": formula,
                        "estimate": estimates[index],
                        "cluster_se": ses[index],
                        "p_value": p_values[index],
                        "ci95_low": lows[index],
                        "ci95_high": highs[index],
                        "n_cells": len(arm),
                        "n_question_clusters": int(arm.prompt_id.nunique()),
                        "question_fixed_effects": include_question_fixed_effects,
                        "interpretation": (
                            "standardized within-question conditional association; "
                            "not a causal mediation estimate"
                            if include_question_fixed_effects
                            else "standardized conditional association without question fixed effects; "
                            "not a causal mediation estimate"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, mode: str, parameter: str, output: Path) -> None:
    data = summary.loc[
        summary.metric.eq(metric) & summary.decoding_mode.eq(mode)
    ].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for value, group in data.groupby(parameter, sort=True):
        group = group.sort_values("temperature")
        ax.errorbar(
            group.temperature,
            group["mean"],
            yerr=1.96 * group["se"],
            marker="o",
            linewidth=1.8,
            capsize=2,
            label=f"{parameter}={value:g}",
        )
    ax.set_xlabel("Temperature")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} — {mode.replace('_', ' ')}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    answers = pd.read_csv(args.input, low_memory=False)
    steps = pd.read_csv(args.token_steps, low_memory=False)
    keys = ["prompt_id", "decoding_mode", "temperature", "top_p", "top_k"]
    if len(answers) != 5250 or answers.generation_key.nunique() != 5250:
        raise ValueError("Expected 5,250 unique answer rows")
    if len(steps) != int(answers.new_tokens.sum()):
        raise ValueError("Token-step rows do not match generated-token total")

    cell_rows = []
    for key_values, group in answers.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key_values))
        normalized = group.generated_text.fillna("").astype(str).str.strip().str.lower()
        cell_rows.append(
            {
                **base,
                "repetitions": len(group),
                "unique_answer_rate": normalized.nunique() / len(group),
                "pairwise_token_jaccard_distance": pairwise_jaccard_distance(
                    group.generated_text
                ),
                "mean_word_length": group.word_length.mean(),
                "mean_latency_seconds": group.model_generation_latency_seconds.mean(),
                **{metric: group[metric].mean() for metric in MECHANISMS},
            }
        )
    cells = pd.DataFrame(cell_rows)
    if len(cells) != 1050 or not cells.repetitions.eq(5).all():
        raise ValueError("Expected 1,050 balanced question-setting cells")
    cells.to_csv(tables / "question_setting_cells.csv", index=False)

    metrics = MECHANISMS + DIVERSITY + ["mean_word_length", "mean_latency_seconds"]
    summary = normal_summary(
        cells,
        ["decoding_mode", "temperature", "top_p", "top_k"],
        metrics,
    )
    summary.to_csv(tables / "by_setting_summary.csv", index=False)
    clustered_models(cells).to_csv(tables / "clustered_factor_models.csv", index=False)
    pathway_no_fe = pathway_models(cells, include_question_fixed_effects=False)
    pathway_fe = pathway_models(cells, include_question_fixed_effects=True)
    pathway_fe.to_csv(
        tables / "mechanism_diversity_associations_question_fe.csv", index=False
    )
    pathway_no_fe.to_csv(
        tables / "mechanism_diversity_associations_without_question_fe.csv",
        index=False,
    )
    comparison_keys = ["decoding_mode", "diversity_outcome", "mechanism"]
    pathway_comparison = pathway_fe.merge(
        pathway_no_fe,
        on=comparison_keys,
        suffixes=("_question_fe", "_without_question_fe"),
        validate="one_to_one",
    )
    pathway_comparison["estimate_change"] = (
        pathway_comparison["estimate_question_fe"]
        - pathway_comparison["estimate_without_question_fe"]
    )
    pathway_comparison.to_csv(
        tables / "mechanism_diversity_associations_fe_comparison.csv", index=False
    )

    step_summary = (
        steps.groupby(["decoding_mode", "temperature", "top_p", "top_k"], as_index=False)
        .agg(
            token_steps=("step_id", "size"),
            mean_entropy_nats=("token_entropy_nats", "mean"),
            mean_effective_support=("effective_support_size", "mean"),
            median_retained_support=("retained_support_size", "median"),
            maximum_retained_support=("retained_support_size", "max"),
            mean_max_probability=("max_token_probability", "mean"),
        )
    )
    step_summary.to_csv(tables / "token_step_descriptive_summary.csv", index=False)

    k1 = steps.loc[steps.decoding_mode.eq("pure_top_k") & steps.top_k.eq(1)]
    k1_by_temperature = (
        k1.groupby("temperature", as_index=False)
        .agg(
            token_steps=("step_id", "size"),
            tied_steps=("retained_support_size", lambda x: int((x > 1).sum())),
            maximum_support=("retained_support_size", "max"),
        )
    )
    k1_by_temperature["tie_rate"] = (
        k1_by_temperature.tied_steps / k1_by_temperature.token_steps
    )
    k1_by_temperature.to_csv(tables / "k1_ties_by_temperature.csv", index=False)

    for metric in MECHANISMS + DIVERSITY:
        plot_metric(summary, metric, "pure_top_p", "top_p", figures / f"{metric}_pure_top_p.png")
        plot_metric(summary, metric, "pure_top_k", "top_k", figures / f"{metric}_pure_top_k.png")

    key_means = summary.loc[
        summary.metric.isin(MECHANISMS + DIVERSITY),
        ["decoding_mode", "temperature", "top_p", "top_k", "metric", "mean"],
    ]
    manifest = {
        "input": args.input,
        "token_steps": args.token_steps,
        "answer_rows": len(answers),
        "token_rows": len(steps),
        "question_setting_cells": len(cells),
        "questions": int(cells.prompt_id.nunique()),
        "uncertainty_unit": "question",
        "aggregation": "five repetitions within question-setting, then 50 questions",
        "model_scope": (
            "primary mechanism-diversity associations include question fixed effects "
            "and CR1 standard errors clustered by question; estimates are not causal mediation"
        ),
        "outputs": {
            "tables": len(list(tables.glob("*.csv"))),
            "figures": len(list(figures.glob("*.png"))),
        },
        "k1_tied_steps": int((k1.retained_support_size > 1).sum()),
        "k1_token_steps": len(k1),
        "k1_tie_rate": float((k1.retained_support_size > 1).mean()),
        "valid": bool(
            len(cells) == 1050
            and cells.prompt_id.nunique() == 50
            and not key_means["mean"].isna().any()
        ),
    }
    with (output / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    if not manifest["valid"]:
        raise RuntimeError("Mechanism analysis validation failed")


if __name__ == "__main__":
    main()
