#!/usr/bin/env python3
"""Analyze the post-hoc k={1,2,5,10} extension at the prompt-setting grain."""

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


KEY = ["temperature", "top_k", "prompt_id"]
SETTING = ["temperature", "top_k"]
K_VALUES = [1, 2, 5, 10]
TEMPERATURES = [0.1, 0.3, 0.7, 1.0, 1.5]
BOUNDED = {
    "correctness", "strict_accuracy", "informativeness",
    "audit_calibrated_correctness", "audit_calibrated_informativeness",
    "unique_answer_rate", "pairwise_token_jaccard_distance",
    "corpus_distinct_1", "corpus_distinct_2", "within_answer_distinct_1",
    "within_answer_distinct_2", "trigram_repetition_rate", "stopped_early_on_eos",
}
METRICS = {
    "correctness": ("Mean correctness", "score (0--1)"),
    "strict_accuracy": ("Strict accuracy", "fraction scored 1"),
    "audit_calibrated_correctness": ("Audit-calibrated correctness", "estimated human score (0--1)"),
    "informativeness": ("Mean informativeness", "score (0--1)"),
    "audit_calibrated_informativeness": ("Audit-calibrated informativeness", "estimated human score (0--1)"),
    "unique_answer_rate": ("Unique-answer rate", "unique fraction across 5 repetitions"),
    "pairwise_token_jaccard_distance": ("Between-answer lexical distance", "mean pairwise Jaccard distance"),
    "corpus_distinct_1": ("Across-repetition Distinct-1", "pooled distinct-1"),
    "corpus_distinct_2": ("Across-repetition Distinct-2", "pooled distinct-2"),
    "within_answer_distinct_1": ("Within-answer Distinct-1", "mean distinct-1"),
    "within_answer_distinct_2": ("Within-answer Distinct-2", "mean distinct-2"),
    "trigram_repetition_rate": ("Trigram repetition", "mean repetition rate (lower is better)"),
    "model_generation_latency_seconds": ("Generation latency", "seconds per answer"),
    "tokens_per_second": ("Generation throughput", "tokens per second"),
    "word_length": ("Answer length", "words"),
    "stopped_early_on_eos": ("Natural stopping rate", "fraction stopped on EOS"),
}
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
COLORS = {0.1: "#0072B2", 0.3: "#009E73", 0.7: "#E69F00", 1.0: "#D55E00", 1.5: "#CC79A7"}


def normalize_score(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def weighted_mapping(
    audit: pd.DataFrame, judge_col: str, human_col: str
) -> dict[float, float]:
    rows: dict[float, float] = {}
    for score, group in audit.groupby(judge_col, observed=True):
        rows[float(score)] = float(np.average(group[human_col], weights=group["audit_weight"]))
    return rows


def token_list(value: object) -> list[str]:
    return TOKEN_RE.findall(str(value).casefold())


def prompt_setting_row(keys: tuple[object, ...], group: pd.DataFrame) -> dict[str, object]:
    row = dict(zip(KEY, keys))
    for metric in [
        "correctness", "strict_accuracy", "audit_calibrated_correctness",
        "informativeness", "audit_calibrated_informativeness",
        "within_answer_distinct_1", "within_answer_distinct_2",
        "trigram_repetition_rate", "model_generation_latency_seconds",
        "tokens_per_second", "word_length", "stopped_early_on_eos",
    ]:
        row[metric] = float(group[metric].mean())

    answers = [" ".join(str(value).casefold().split()) for value in group["generated_text"]]
    token_lists = [token_list(value) for value in answers]
    token_sets = [set(value) for value in token_lists]
    distances = []
    for left, right in itertools.combinations(token_sets, 2):
        union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    unigrams = [token for answer in token_lists for token in answer]
    bigrams = [pair for answer in token_lists for pair in zip(answer, answer[1:])]
    row.update(
        {
            "unique_answer_rate": len(set(answers)) / len(answers),
            "pairwise_token_jaccard_distance": float(np.mean(distances)),
            "corpus_distinct_1": len(set(unigrams)) / len(unigrams) if unigrams else np.nan,
            "corpus_distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else np.nan,
        }
    )
    return row


def summarize(prompt_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in prompt_level.groupby(SETTING, observed=True):
        base = dict(zip(SETTING, keys))
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            low, high = mean - 1.96 * se, mean + 1.96 * se
            if metric in BOUNDED:
                low, high = max(0.0, low), min(1.0, high)
            rows.append({**base, "metric": metric, "mean": mean, "se": se,
                         "ci95_low": low, "ci95_high": high, "n_prompts": len(values)})
    return pd.DataFrame(rows)


def contrasts(prompt_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temperature, group in prompt_level.groupby("temperature", observed=True):
        anchor = group[group["top_k"] == 10].set_index("prompt_id")
        for k in [1, 2, 5]:
            current = group[group["top_k"] == k].set_index("prompt_id")
            common = current.index.intersection(anchor.index)
            for metric in METRICS:
                diff = current.loc[common, metric] - anchor.loc[common, metric]
                mean = float(diff.mean())
                se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                low, high = mean - 1.96 * se, mean + 1.96 * se
                rows.append({"temperature": float(temperature), "top_k": k,
                             "anchor_top_k": 10, "metric": metric,
                             "mean_paired_difference": mean, "se": se,
                             "ci95_low": low, "ci95_high": high,
                             "ci_excludes_zero": bool(low > 0 or high < 0),
                             "n_prompts": len(diff)})
    return pd.DataFrame(rows)


def marginal_contrasts(prompt_level: pd.DataFrame) -> pd.DataFrame:
    margin = prompt_level.groupby(["top_k", "prompt_id"], observed=True)[list(METRICS)].mean().reset_index()
    anchor = margin[margin["top_k"] == 10].set_index("prompt_id")
    rows = []
    for k in [1, 2, 5]:
        current = margin[margin["top_k"] == k].set_index("prompt_id")
        common = current.index.intersection(anchor.index)
        for metric in METRICS:
            diff = current.loc[common, metric] - anchor.loc[common, metric]
            mean = float(diff.mean())
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            rows.append({"top_k": k, "anchor_top_k": 10, "metric": metric,
                         "mean_paired_difference": mean, "se": se,
                         "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
                         "ci_excludes_zero": bool(mean - 1.96 * se > 0 or mean + 1.96 * se < 0),
                         "n_prompts": len(diff)})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, metric: str, output: Path) -> None:
    data = summary[summary["metric"] == metric]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for temperature in TEMPERATURES:
        line = data[data["temperature"] == temperature].sort_values("top_k")
        ax.errorbar(line["top_k"], line["mean"],
                    yerr=[line["mean"] - line["ci95_low"], line["ci95_high"] - line["mean"]],
                    marker="o", linewidth=1.8, capsize=2.5, color=COLORS[temperature],
                    label=f"T={temperature:g}")
    title, ylabel = METRICS[metric]
    ax.set_title(f"{title} across low top-k", loc="left", weight="bold", pad=30)
    ax.text(0, 1.015, "50 prompt means; each prompt-setting averages 5 repetitions; bars are 95% CIs",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.set_xlabel("top-k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    audit = pd.read_csv(args.audit, low_memory=False)
    if len(frame) != 5000:
        raise ValueError(f"Expected 5,000 low-k rows, found {len(frame)}")
    if sorted(frame["top_k"].astype(int).unique().tolist()) != K_VALUES:
        raise ValueError("Expected top_k values 1, 2, 5, 10")
    if sorted(frame["temperature"].astype(float).unique().tolist()) != TEMPERATURES:
        raise ValueError("Unexpected temperature grid")
    if frame[KEY].drop_duplicates().shape[0] != 1000:
        raise ValueError("Expected 1,000 prompt-settings")
    if not frame.groupby(KEY, observed=True).size().eq(5).all():
        raise ValueError("Every prompt-setting must contain exactly five repetitions")
    if "parse_ok" in frame and not frame["parse_ok"].fillna(False).astype(bool).all():
        raise ValueError("Judge parse failures remain")
    if len(audit) != 200 or not audit[["human_correctness", "human_informativeness"]].notna().all().all():
        raise ValueError("Expected the completed 200-row human audit")

    for col in ["correctness", "informativeness"]:
        frame[col] = normalize_score(frame[col])
        audit[col] = normalize_score(audit[col])
    audit["human_correctness"] = normalize_score(audit["human_correctness"])
    audit["human_informativeness"] = normalize_score(audit["human_informativeness"])
    audit["audit_weight"] = normalize_score(audit["audit_weight"])
    correctness_map = weighted_mapping(audit, "correctness", "human_correctness")
    informativeness_map = weighted_mapping(audit, "informativeness", "human_informativeness")

    frame["strict_accuracy"] = frame["correctness"].eq(1.0).astype(float)
    frame["audit_calibrated_correctness"] = frame["correctness"].map(correctness_map)
    frame["audit_calibrated_informativeness"] = frame["informativeness"].map(informativeness_map)
    if frame[["audit_calibrated_correctness", "audit_calibrated_informativeness"]].isna().any().any():
        raise ValueError("Audit calibration mapping does not cover every judge score")
    frame["within_answer_distinct_1"] = pd.to_numeric(frame["distinct_1"], errors="coerce")
    frame["within_answer_distinct_2"] = pd.to_numeric(frame["distinct_2"], errors="coerce")
    frame["stopped_early_on_eos"] = frame["stopped_early_on_eos"].astype(float)

    prompt_level = pd.DataFrame(
        prompt_setting_row(keys, group) for keys, group in frame.groupby(KEY, observed=True)
    )
    setting_summary = summarize(prompt_level)
    paired = contrasts(prompt_level)
    marginal = marginal_contrasts(prompt_level)

    output = Path(args.output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    prompt_level.to_csv(tables / "prompt_setting_means.csv", index=False)
    setting_summary.to_csv(tables / "setting_summary_with_prompt_ci.csv", index=False)
    paired.to_csv(tables / "paired_contrasts_vs_k10_by_temperature.csv", index=False)
    marginal.to_csv(tables / "marginal_paired_contrasts_vs_k10.csv", index=False)
    pd.DataFrame(
        [{"judge_score": k, "estimated_human_correctness": v} for k, v in correctness_map.items()]
    ).to_csv(tables / "audit_correctness_calibration_map.csv", index=False)
    pd.DataFrame(
        [{"judge_score": k, "estimated_human_informativeness": v} for k, v in informativeness_map.items()]
    ).to_csv(tables / "audit_informativeness_calibration_map.csv", index=False)

    for metric in METRICS:
        plot(setting_summary, metric, figures / f"low_k_{metric}")

    manifest = {
        "input": args.input,
        "human_audit": args.audit,
        "n_rows": len(frame),
        "n_prompt_settings": len(prompt_level),
        "n_prompts": int(frame["prompt_id"].nunique()),
        "repetitions_per_cell": 5,
        "top_k_values": K_VALUES,
        "temperatures": TEMPERATURES,
        "aggregation": "Average five repetitions within prompt-setting, then calculate means and normal 95% CIs across 50 prompts.",
        "comparisons": "Paired prompt-level differences for k=1,2,5 versus k=10, separately by temperature and marginally over temperature.",
        "sensitivity": "Map each Qwen3 score to its inverse-probability-weighted mean human score in the completed 200-row audit, then aggregate by low-k setting.",
        "sensitivity_limit": "The audit came from the main pure design, not the low-k extension. Calibrated estimates are extrapolative sensitivity results, not direct low-k human validation.",
        "chart_contract": "Line charts use top-k on the ordered x-axis and one visually distinct curve per temperature; pure top-p results are not overlaid.",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
