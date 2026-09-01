"""Create the two-panel main-study temperature figure.

The input table contains one row per question and decoding setting. Each row
already averages the 25 repeated generations for that question-setting cell.
The plotted mean and 95% interval are therefore computed across 50 question
means, keeping the question as the unit of uncertainty.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/pure_pk_main_25rep_revised/tables/prompt_setting_means.csv"
OUT_PNG = ROOT / "reports/assets/main_temperature_unique_two_panel.png"
OUT_PDF = ROOT / "reports/assets/main_temperature_unique_two_panel.pdf"
OUT_DATA = ROOT / "analysis/pure_pk_main_25rep_revised/tables/main_temperature_unique_two_panel.csv"


def summarize(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["temperature", parameter], sort=True):
        values = group["unique_answer_rate"].to_numpy(dtype=float)
        mean = values.mean()
        se = values.std(ddof=1) / np.sqrt(len(values))
        rows.append(
            {
                "temperature": keys[0],
                parameter: keys[1],
                "mean": mean,
                "se": se,
                "ci95_low": mean - 1.96 * se,
                "ci95_high": mean + 1.96 * se,
                "n_questions": len(values),
            }
        )
    return pd.DataFrame(rows)


def draw_panel(ax, summary: pd.DataFrame, parameter: str, levels, title: str):
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    markers = ["o", "s", "^", "D", "P"]
    linestyles = ["-", "--", "-.", ":", "-"]
    for level, color, marker, linestyle in zip(levels, colors, markers, linestyles):
        part = summary.loc[summary[parameter] == level].sort_values("temperature")
        yerr = np.vstack(
            [part["mean"] - part["ci95_low"], part["ci95_high"] - part["mean"]]
        )
        label = rf"${parameter[-1]}={level:g}$"
        ax.errorbar(
            part["temperature"],
            part["mean"],
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=4.5,
            capsize=2.5,
            elinewidth=0.9,
            label=label,
        )
    ax.set_title(title, fontsize=11, weight="semibold")
    ax.set_xlabel("Temperature")
    ax.set_xticks([0.1, 0.3, 0.7, 1.0, 1.5])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")


def main():
    data = pd.read_csv(INPUT)
    k = data.loc[
        (data["decoding_mode"] == "pure_top_k")
        & data["top_k"].isin([10, 15, 20, 25, 30])
    ].copy()
    p = data.loc[
        (data["decoding_mode"] == "pure_top_p")
        & data["top_p"].isin([0.6, 0.8, 0.9, 0.95])
    ].copy()

    k_summary = summarize(k, "top_k")
    p_summary = summarize(p, "top_p")
    combined = pd.concat(
        [k_summary.assign(decoding_mode="pure_top_k"),
         p_summary.assign(decoding_mode="pure_top_p")],
        ignore_index=True,
        sort=False,
    )
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_DATA, index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    draw_panel(axes[0], k_summary, "top_k", [10, 15, 20, 25, 30], "Pure top-$k$")
    draw_panel(axes[1], p_summary, "top_p", [0.6, 0.8, 0.9, 0.95], "Pure top-$p$")
    axes[0].set_ylabel("Unique-answer rate")
    axes[0].set_ylim(0, 0.9)
    fig.suptitle("Unique-answer rate across temperature", fontsize=13, weight="semibold")
    fig.text(
        0.5,
        0.01,
        "Points average 25 repetitions within each question; bars are 95% intervals across 50 question means.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=2.5)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    expected = 25 + 20
    if len(combined) != expected or not (combined["n_questions"] == 50).all():
        raise RuntimeError("Unexpected number of cells or question means")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_DATA}")


if __name__ == "__main__":
    main()
