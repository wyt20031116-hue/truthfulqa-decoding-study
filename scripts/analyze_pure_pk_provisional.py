#!/usr/bin/env python3
"""Analyze pure top-p and pure top-k runs without conflating their curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_MODE_ROWS = {"pure_top_p": 25_000, "pure_top_k": 31_250}
PROMPT_GROUP = ["decoding_mode", "temperature", "top_p", "top_k", "prompt_id"]
SETTING_GROUP = ["decoding_mode", "temperature", "top_p", "top_k"]
METRICS = {
    "correctness": ("Mean correctness", "score (0–1)"),
    "strict_accuracy": ("Strict accuracy", "fraction scored 1"),
    "informativeness": ("Mean informativeness", "score (0–1)"),
    "distinct_1": ("Distinct-1 diversity", "mean within-answer distinct-1"),
    "distinct_2": ("Distinct-2 diversity", "mean within-answer distinct-2"),
    "trigram_repetition_rate": ("Trigram repetition", "mean repetition rate (lower is better)"),
    "model_generation_latency_seconds": ("Generation latency", "seconds per answer"),
    "tokens_per_second": ("Generation throughput", "tokens per second"),
    "word_length": ("Answer length", "words"),
    "stopped_early_on_eos": ("Natural stopping rate", "fraction stopped on EOS"),
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def ci_summary(prompt_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in prompt_level.groupby(SETTING_GROUP, observed=True):
        base = dict(zip(SETTING_GROUP, keys))
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append({**base, "metric": metric, "mean": mean, "ci95_low": mean - 1.96 * sem,
                         "ci95_high": mean + 1.96 * sem, "n_prompts": len(values)})
    return pd.DataFrame(rows)


def save_curve(summary: pd.DataFrame, mode: str, metric: str, x: str, hue: str, output: Path) -> None:
    data = summary[(summary["decoding_mode"] == mode) & (summary["metric"] == metric)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for color, hue_value in zip(PALETTE, sorted(data[hue].unique())):
        line = data[data[hue] == hue_value].sort_values(x)
        ax.plot(line[x], line["mean"], marker="o", linewidth=2, markersize=5,
                color=color, label=f"{hue}={hue_value:g}")
        ax.fill_between(line[x].to_numpy(float), line["ci95_low"].to_numpy(float),
                        line["ci95_high"].to_numpy(float), color=color, alpha=0.13)
    title, ylabel = METRICS[metric]
    family = "pure top-p" if mode == "pure_top_p" else "pure top-k"
    fig.suptitle(f"{title} across {x.replace('_', ' ')} — {family}", x=0.125, y=0.98,
                 ha="left", fontsize=13, weight="bold")
    ax.set_title("Means of 50 prompt-level averages; ribbons are 95% CIs across prompts",
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
    parser.add_argument("--input", required=True, help="Completed 56,250-row judgment CSV")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    missing = set(METRICS) - set(frame)
    if missing - {"strict_accuracy"}:
        raise ValueError(f"Missing analysis columns: {sorted(missing - {'strict_accuracy'})}")
    if len(frame) != 56_250 or frame["decoding_mode"].value_counts().to_dict() != EXPECTED_MODE_ROWS:
        raise ValueError("Expected the complete 56,250-row pure top-p/top-k run")
    if not frame["parse_ok"].eq(True).all():
        raise ValueError("Judge parse failures remain; refusing to plot incomplete scores")
    frame["strict_accuracy"] = frame["correctness"].eq(1).astype(float)
    frame["stopped_early_on_eos"] = frame["stopped_early_on_eos"].astype(float)

    output = Path(args.output_dir)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    prompt_level = frame.groupby(PROMPT_GROUP, observed=True)[list(METRICS)].mean().reset_index()
    setting_summary = ci_summary(prompt_level)
    prompt_level.to_csv(tables / "prompt_setting_means.csv", index=False)
    setting_summary.to_csv(tables / "setting_summary_with_prompt_ci.csv", index=False)

    # Never combine pure-p and pure-k curves in one panel.  Each family gets:
    # (1) temperature on x with one curve per p or k; and
    # (2) its own decoding parameter on x with one curve per temperature.
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
        "aggregation": "Average 25 repetitions within each prompt-setting, then average across 50 prompts; 95% normal CIs across prompt means.",
        "curve_rule": "Pure top-p and pure top-k are always separate; temperature plots use one curve per p or k.",
        "latency_definition": "Qwen2.5 model_generation_latency_seconds; judge runtime is excluded.",
        "judge_limitation": "Correctness and informativeness use provisional Qwen3-32B NF4 v1 compact scores; hard-set validation limitations must be disclosed.",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
