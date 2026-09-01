#!/usr/bin/env python3
"""Estimate semantic-diversity/latency Pareto frontiers with prompt bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["decoding_mode", "temperature", "top_p", "top_k"]


def pareto_mask(diversity: np.ndarray, latency: np.ndarray) -> np.ndarray:
    """Return nondominated points: maximize diversity and minimize latency."""
    n = len(diversity)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominates_i = (
            (diversity >= diversity[i])
            & (latency <= latency[i])
            & ((diversity > diversity[i]) | (latency < latency[i]))
        )
        if dominates_i.any():
            keep[i] = False
    return keep


def label(row: pd.Series) -> str:
    if row["decoding_mode"] == "pure_top_p":
        return f"T={row['temperature']:g}, p={row['top_p']:g}"
    return f"T={row['temperature']:g}, k={int(row['top_k'])}"


def analyze(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    prompts = np.sort(frame["prompt_id"].unique())
    settings = frame[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    setting_index = pd.MultiIndex.from_frame(settings)

    point = (
        frame.groupby(KEYS, dropna=False)[
            ["semantic_cosine_distance", "model_generation_latency_seconds",
             "correctness", "informativeness", "unique_answer_rate"]
        ]
        .mean()
        .reindex(setting_index)
        .reset_index()
    )
    point["pareto_point_estimate"] = pareto_mask(
        point["semantic_cosine_distance"].to_numpy(),
        point["model_generation_latency_seconds"].to_numpy(),
    )

    rng = np.random.default_rng(seed)
    counts = np.zeros(len(point), dtype=int)
    indexed = frame.set_index("prompt_id")
    for _ in range(n_boot):
        draw = rng.choice(prompts, size=len(prompts), replace=True)
        sampled = pd.concat([indexed.loc[[p]] for p in draw], ignore_index=True)
        means = (
            sampled.groupby(KEYS, dropna=False)[
                ["semantic_cosine_distance", "model_generation_latency_seconds"]
            ]
            .mean()
            .reindex(setting_index)
        )
        counts += pareto_mask(
            means["semantic_cosine_distance"].to_numpy(),
            means["model_generation_latency_seconds"].to_numpy(),
        )
    point["pareto_bootstrap_probability"] = counts / n_boot
    point["setting_label"] = point.apply(label, axis=1)
    return point


def plot_frontier(results: pd.DataFrame, output: Path) -> None:
    colors = {"pure_top_p": "#D97706", "pure_top_k": "#1F5A94"}
    markers = {"pure_top_p": "o", "pure_top_k": "s"}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for mode, g in results.groupby("decoding_mode"):
        sizes = 35 + 180 * g["pareto_bootstrap_probability"]
        ax.scatter(
            g["model_generation_latency_seconds"], g["semantic_cosine_distance"],
            s=sizes, alpha=0.72, color=colors[mode], marker=markers[mode],
            edgecolor="white", linewidth=0.6, label=mode.replace("_", " "),
        )
    front = results[results["pareto_point_estimate"]].sort_values(
        "model_generation_latency_seconds"
    )
    ax.plot(
        front["model_generation_latency_seconds"], front["semantic_cosine_distance"],
        color="#222222", linewidth=1.3, linestyle="--", label="point-estimate frontier",
    )
    annotation_offsets = {
        "T=0.7, p=0.8": (7, 7),
        "T=1, k=25": (-52, 8),
        "T=1.5, k=30": (7, -17),
        "T=1.5, p=0.9": (7, 5),
        "T=1.5, p=0.95": (7, 5),
    }
    annotated = results[results["setting_label"].isin(annotation_offsets)]
    for _, row in annotated.iterrows():
        ax.annotate(
            row["setting_label"],
            (row["model_generation_latency_seconds"], row["semantic_cosine_distance"]),
            xytext=annotation_offsets[row["setting_label"]],
            textcoords="offset points", fontsize=7,
        )
    ax.set_xlabel("Mean generator latency (seconds per answer; lower is better)")
    ax.set_ylabel("Mean pairwise semantic cosine distance (higher is better)")
    ax.set_title("Semantic-diversity--latency frontier\nMarker size = prompt-bootstrap frontier probability")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="analysis/semantic_diversity/prompt_setting_metrics_with_semantic.csv"
    )
    parser.add_argument("--output-dir", default="analysis/pareto_frontier")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, low_memory=False)
    required = set(KEYS + [
        "prompt_id", "semantic_cosine_distance", "model_generation_latency_seconds",
        "correctness", "informativeness", "unique_answer_rate",
    ])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(frame) != 2250 or frame["prompt_id"].nunique() != 50:
        raise ValueError("Expected 2,250 prompt-settings from 50 prompts")

    results = analyze(frame, args.bootstrap, args.seed)
    results.to_csv(out / "pareto_settings.csv", index=False)
    plot_frontier(results, out / "semantic_diversity_latency_frontier.png")

    stable = results.sort_values(
        ["pareto_bootstrap_probability", "semantic_cosine_distance"], ascending=False
    )
    stable[stable["pareto_bootstrap_probability"] >= 0.50].to_csv(
        out / "stable_frontier_settings.csv", index=False
    )
    manifest = {
        "input": str(source),
        "n_prompt_settings": int(len(frame)),
        "n_settings": int(len(results)),
        "n_prompts": int(frame["prompt_id"].nunique()),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "prompt",
        "objectives": {
            "maximize": "semantic_cosine_distance",
            "minimize": "model_generation_latency_seconds",
        },
        "quality_metrics_role": "descriptive only because Qwen3-NF4 quality scores are provisional",
        "point_frontier_n": int(results["pareto_point_estimate"].sum()),
        "stable_probability_ge_0.5_n": int((results["pareto_bootstrap_probability"] >= 0.5).sum()),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nTop frontier probabilities:")
    print(stable[["setting_label", "semantic_cosine_distance",
                  "model_generation_latency_seconds", "pareto_bootstrap_probability",
                  "correctness", "informativeness"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
