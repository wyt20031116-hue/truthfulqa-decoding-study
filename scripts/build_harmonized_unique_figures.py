#!/usr/bin/env python3
"""Build five-repetition unique-answer figures including the low-k extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEMPERATURES = [0.1, 0.3, 0.7, 1.0, 1.5]
K_VALUES = [1, 2, 5, 10, 15, 20, 25, 30]
P_VALUES = [0.6, 0.8, 0.9, 0.95]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A3E9D", "#4D4D4D"]


def unique_prompt_means(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    rows = []
    for (temperature, value, prompt_id), group in frame.groupby(
        ["temperature", parameter, "prompt_id"], observed=True
    ):
        answers = [" ".join(str(x).casefold().split()) for x in group["generated_text"]]
        if len(answers) != 5:
            raise ValueError("Every harmonized prompt-setting must contain five repetitions")
        rows.append(
            {
                "temperature": float(temperature),
                parameter: float(value),
                "prompt_id": prompt_id,
                "unique_answer_rate": len(set(answers)) / 5,
            }
        )
    return pd.DataFrame(rows)


def summarize(prompt: pd.DataFrame, parameter: str) -> pd.DataFrame:
    rows = []
    for keys, group in prompt.groupby(["temperature", parameter], observed=True):
        values = group["unique_answer_rate"].astype(float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(
            {
                "temperature": float(keys[0]),
                parameter: float(keys[1]),
                "mean": mean,
                "ci95_low": max(0.0, mean - 1.96 * se),
                "ci95_high": min(1.0, mean + 1.96 * se),
                "n_prompts": len(values),
            }
        )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, x: str, hue: str, path: Path, family: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    hue_values = sorted(summary[hue].unique())
    for color, value in zip(PALETTE, hue_values):
        line = summary[summary[hue] == value].sort_values(x)
        ax.plot(line[x], line["mean"], marker="o", linewidth=2, markersize=5,
                color=color, label=f"{hue}={value:g}")
    ax.set_title(f"Unique-answer rate across {x.replace('_', ' ')} -- {family}",
                 loc="left", weight="bold", pad=30)
    ax.text(0, 1.015,
            "Harmonized repetitions 0--4 for every setting; means across 50 prompts",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel("unique fraction across 5 repetitions")
    if x == "top_k":
        ax.set_xticks(K_VALUES)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2 if len(hue_values) > 4 else 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-judgments", required=True)
    parser.add_argument("--low-k-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-assets", required=True)
    args = parser.parse_args()

    main = pd.read_csv(
        args.main_judgments,
        low_memory=False,
        usecols=["decoding_mode", "temperature", "top_p", "top_k", "prompt_id", "repetition_id", "generated_text"],
    )
    main = main[main["repetition_id"].isin(range(5))].copy()
    top_k_main = main[(main["decoding_mode"] == "pure_top_k") & (main["top_k"].isin([10, 15, 20, 25, 30]))]
    top_p_main = main[main["decoding_mode"] == "pure_top_p"]
    k_prompt = unique_prompt_means(top_k_main, "top_k")
    p_prompt = unique_prompt_means(top_p_main, "top_p")
    k_summary = summarize(k_prompt, "top_k")
    p_summary = summarize(p_prompt, "top_p")

    low = pd.read_csv(args.low_k_summary)
    low = low[(low["metric"] == "unique_answer_rate") & (low["top_k"].isin([1, 2, 5]))]
    low = low[["temperature", "top_k", "mean", "ci95_low", "ci95_high", "n_prompts"]]
    combined_k = pd.concat([low, k_summary], ignore_index=True).sort_values(["temperature", "top_k"])
    if combined_k[["temperature", "top_k"]].drop_duplicates().shape[0] != 40:
        raise ValueError("Expected 40 temperature-by-k settings")
    if p_summary[["temperature", "top_p"]].drop_duplicates().shape[0] != 20:
        raise ValueError("Expected 20 temperature-by-p settings")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined_k.to_csv(output / "top_k_unique_harmonized_5rep.csv", index=False)
    p_summary.to_csv(output / "top_p_unique_harmonized_5rep.csv", index=False)

    plot(combined_k, "temperature", "top_k", output / "top_k_temperature_unique", "pure top-k")
    plot(p_summary, "temperature", "top_p", output / "top_p_temperature_unique", "pure top-p")
    plot(combined_k, "top_k", "temperature", output / "top_k_effect_unique", "pure top-k")
    plot(p_summary, "top_p", "temperature", output / "top_p_effect_unique", "pure top-p")

    assets = Path(args.report_assets)
    assets.mkdir(parents=True, exist_ok=True)
    for name in ["top_k_temperature_unique", "top_p_temperature_unique", "top_k_effect_unique", "top_p_effect_unique"]:
        (assets / f"{name}.png").write_bytes((output / f"{name}.png").read_bytes())
    print("Built harmonized five-repetition figures with k=1,2,5,10,15,20,25,30")


if __name__ == "__main__":
    main()
