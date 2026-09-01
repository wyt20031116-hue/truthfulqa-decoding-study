#!/usr/bin/env python3
"""Revised pure top-p/top-k analysis with between-generation diversity metrics."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_MODE_ROWS = {"pure_top_p": 25_000, "pure_top_k": 31_250}
PROMPT_GROUP = ["decoding_mode", "temperature", "top_p", "top_k", "prompt_id"]
SETTING_GROUP = ["decoding_mode", "temperature", "top_p", "top_k"]
AVERAGED_METRICS = {
    "correctness": ("Mean correctness", "score (0–1)"),
    "strict_accuracy": ("Strict accuracy", "fraction scored 1"),
    "informativeness": ("Mean informativeness", "score (0–1)"),
    "within_answer_distinct_1": ("Within-answer Distinct-1", "mean distinct-1"),
    "within_answer_distinct_2": ("Within-answer Distinct-2", "mean distinct-2"),
    "trigram_repetition_rate": ("Trigram repetition", "mean repetition rate (lower is better)"),
    "model_generation_latency_seconds": ("Generation latency", "seconds per answer"),
    "tokens_per_second": ("Generation throughput", "tokens per second"),
    "word_length": ("Answer length", "words"),
    "stopped_early_on_eos": ("Natural stopping rate", "fraction stopped on EOS"),
}
BETWEEN_GENERATION_METRICS = {
    "unique_answer_rate": ("Unique-answer rate", "unique fraction across 25 repetitions"),
    "pairwise_token_jaccard_distance": ("Between-answer lexical distance", "mean pairwise Jaccard distance"),
    "corpus_distinct_1": ("Across-repetition Distinct-1", "pooled distinct-1 across 25 answers"),
    "corpus_distinct_2": ("Across-repetition Distinct-2", "pooled distinct-2 across 25 answers"),
}
METRICS = {**AVERAGED_METRICS, **BETWEEN_GENERATION_METRICS}
BOUNDED_METRICS = {
    "correctness", "strict_accuracy", "informativeness", "within_answer_distinct_1",
    "within_answer_distinct_2", "trigram_repetition_rate", "stopped_early_on_eos",
    "unique_answer_rate", "pairwise_token_jaccard_distance", "corpus_distinct_1",
    "corpus_distinct_2",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(str(text).casefold())


def prompt_setting_row(keys: tuple[object, ...], group: pd.DataFrame) -> dict[str, object]:
    row = dict(zip(PROMPT_GROUP, keys))
    for metric in AVERAGED_METRICS:
        row[metric] = float(group[metric].astype(float).mean())

    answers = [str(value).strip() for value in group["generated_text"]]
    normalized = [" ".join(value.casefold().split()) for value in answers]
    token_lists = [tokens(value) for value in answers]
    token_sets = [set(value) for value in token_lists]
    distances = []
    for left, right in itertools.combinations(token_sets, 2):
        union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    unigrams = [token for answer in token_lists for token in answer]
    bigrams = [pair for answer in token_lists for pair in zip(answer, answer[1:])]
    row.update({
        "unique_answer_rate": len(set(normalized)) / len(normalized),
        "pairwise_token_jaccard_distance": float(np.mean(distances)),
        "corpus_distinct_1": len(set(unigrams)) / len(unigrams) if unigrams else np.nan,
        "corpus_distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else np.nan,
    })
    return row


def ci_summary(prompt_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in prompt_level.groupby(SETTING_GROUP, observed=True):
        base = dict(zip(SETTING_GROUP, keys))
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            low, high = mean - 1.96 * sem, mean + 1.96 * sem
            if metric in BOUNDED_METRICS:
                low, high = max(0.0, low), min(1.0, high)
            rows.append({**base, "metric": metric, "mean": mean, "ci95_low": low,
                         "ci95_high": high, "n_prompts": len(values)})
    return pd.DataFrame(rows)


def marginal_summary(prompt_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    parameter_by_mode = {"pure_top_p": "top_p", "pure_top_k": "top_k"}
    for mode, mode_data in prompt_level.groupby("decoding_mode", observed=True):
        parameter = parameter_by_mode[mode]
        for dimension in ["temperature", parameter]:
            # Average settings within each prompt first, preserving the prompt as
            # the independent unit for uncertainty across the 50 questions.
            prompt_margin = mode_data.groupby([dimension, "prompt_id"], observed=True)[list(METRICS)].mean().reset_index()
            for value, group in prompt_margin.groupby(dimension, observed=True):
                for metric in METRICS:
                    values = group[metric].dropna().astype(float)
                    mean = float(values.mean())
                    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                    low, high = mean - 1.96 * sem, mean + 1.96 * sem
                    if metric in BOUNDED_METRICS:
                        low, high = max(0.0, low), min(1.0, high)
                    rows.append({
                        "decoding_mode": mode,
                        "dimension": dimension,
                        "value": value,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_prompts": len(values),
                    })
    return pd.DataFrame(rows)


def paired_contrast_summary(prompt_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    parameter_by_mode = {"pure_top_p": "top_p", "pure_top_k": "top_k"}
    for mode, mode_data in prompt_level.groupby("decoding_mode", observed=True):
        parameter = parameter_by_mode[mode]
        for dimension in ["temperature", parameter]:
            prompt_margin = mode_data.groupby([dimension, "prompt_id"], observed=True)[list(METRICS)].mean().reset_index()
            reference = float(prompt_margin[dimension].min())
            for value in sorted(prompt_margin[dimension].unique()):
                if float(value) == reference:
                    continue
                current = prompt_margin[prompt_margin[dimension] == value].set_index("prompt_id")
                baseline = prompt_margin[prompt_margin[dimension] == reference].set_index("prompt_id")
                common = current.index.intersection(baseline.index)
                for metric in METRICS:
                    differences = current.loc[common, metric].astype(float) - baseline.loc[common, metric].astype(float)
                    mean = float(differences.mean())
                    sem = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
                    low, high = mean - 1.96 * sem, mean + 1.96 * sem
                    rows.append({
                        "decoding_mode": mode,
                        "dimension": dimension,
                        "reference_value": reference,
                        "comparison_value": value,
                        "metric": metric,
                        "mean_paired_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "n_prompts": len(differences),
                    })
    return pd.DataFrame(rows)


def save_curve(summary: pd.DataFrame, mode: str, metric: str, x: str, hue: str, output: Path) -> None:
    data = summary[(summary["decoding_mode"] == mode) & (summary["metric"] == metric)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for color, hue_value in zip(PALETTE, sorted(data[hue].unique())):
        line = data[data[hue] == hue_value].sort_values(x)
        ax.plot(line[x], line["mean"], marker="o", linewidth=2, markersize=5,
                color=color, label=f"{hue}={hue_value:g}")
    title, ylabel = METRICS[metric]
    family = "pure top-p" if mode == "pure_top_p" else "pure top-k"
    fig.suptitle(f"{title} across {x.replace('_', ' ')} — {family}", x=0.125, y=0.98,
                 ha="left", fontsize=13, weight="bold")
    ax.set_title("Means of 50 prompt-level averages; 95% CIs are retained in the summary table",
                 loc="left", fontsize=9, color="#555555", pad=10)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title=None, ncol=2 if len(data[hue].unique()) > 4 else 1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    required = set(AVERAGED_METRICS) - {"strict_accuracy", "within_answer_distinct_1", "within_answer_distinct_2"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing analysis columns: {sorted(missing)}")
    if len(frame) != 56_250 or frame["decoding_mode"].value_counts().to_dict() != EXPECTED_MODE_ROWS:
        raise ValueError("Expected the complete 56,250-row pure top-p/top-k run")
    if not frame["parse_ok"].eq(True).all():
        raise ValueError("Judge parse failures remain")

    frame["strict_accuracy"] = frame["correctness"].eq(1).astype(float)
    frame["stopped_early_on_eos"] = frame["stopped_early_on_eos"].astype(float)
    frame["within_answer_distinct_1"] = frame["distinct_1"].astype(float)
    frame["within_answer_distinct_2"] = frame["distinct_2"].astype(float)

    prompt_rows = [prompt_setting_row(keys, group) for keys, group in frame.groupby(PROMPT_GROUP, observed=True)]
    prompt_level = pd.DataFrame(prompt_rows)
    setting_summary = ci_summary(prompt_level)
    marginal = marginal_summary(prompt_level)
    paired_contrasts = paired_contrast_summary(prompt_level)

    output = Path(args.output_dir)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    prompt_level.to_csv(tables / "prompt_setting_means.csv", index=False)
    setting_summary.to_csv(tables / "setting_summary_with_prompt_ci.csv", index=False)
    marginal.to_csv(tables / "marginal_effect_summary.csv", index=False)
    paired_contrasts.to_csv(tables / "paired_contrast_summary.csv", index=False)

    extrema_rows = []
    for mode, group in setting_summary.groupby("decoding_mode", observed=True):
        for metric, metric_group in group.groupby("metric", observed=True):
            for direction, index in [("minimum", metric_group["mean"].idxmin()),
                                     ("maximum", metric_group["mean"].idxmax())]:
                row = metric_group.loc[index]
                extrema_rows.append({
                    "decoding_mode": mode,
                    "metric": metric,
                    "extreme": direction,
                    "temperature": row["temperature"],
                    "top_p": row["top_p"],
                    "top_k": row["top_k"],
                    "mean": row["mean"],
                    "ci95_low": row["ci95_low"],
                    "ci95_high": row["ci95_high"],
                })
    pd.DataFrame(extrema_rows).to_csv(tables / "setting_extrema.csv", index=False)

    designs = {
        "pure_top_p": ("top_p", output / "figures" / "pure_top_p"),
        "pure_top_k": ("top_k", output / "figures" / "pure_top_k"),
    }
    for mode, (parameter, folder) in designs.items():
        for metric in METRICS:
            save_curve(setting_summary, mode, metric, "temperature", parameter,
                       folder / f"temperature_effect_{metric}")
            save_curve(setting_summary, mode, metric, parameter, "temperature",
                       folder / f"{parameter}_effect_{metric}")

    manifest = {
        "input": args.input,
        "n_rows": len(frame),
        "n_prompts": int(frame["prompt_id"].nunique()),
        "n_repetitions": int(frame["repetition_id"].nunique()),
        "settings": {mode: int(group[SETTING_GROUP].drop_duplicates().shape[0])
                     for mode, group in frame.groupby("decoding_mode")},
        "aggregation": "Average 25 repetitions within each prompt-setting, then average across 50 prompts.",
        "uncertainty": "95% normal CIs across 50 prompt means are saved in tables and clipped to [0,1] for bounded metrics; main curves omit overlapping ribbons.",
        "diversity": {
            "within_answer_distinct_1_2": "Lexical non-repetition inside individual answers.",
            "unique_answer_rate": "Exact normalized answer uniqueness among 25 repetitions.",
            "pairwise_token_jaccard_distance": "Mean lexical-set distance over all answer pairs among 25 repetitions.",
            "corpus_distinct_1_2": "Distinct n-grams pooled across the 25 answers without crossing answer boundaries.",
        },
        "curve_rule": "Pure top-p and pure top-k remain separate; temperature plots use one curve per p or k.",
        "latency_definition": "Qwen2.5 model_generation_latency_seconds; judge runtime is excluded.",
        "judge_limitation": "Correctness and informativeness are provisional Qwen3-NF4 scores and require the stratified human-audit sensitivity analysis.",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
