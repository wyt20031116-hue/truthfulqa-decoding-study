#!/usr/bin/env python3
"""Compute between-generation semantic diversity for the 56,250-answer run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/main_50q_45pure_settings_25rep/generations.csv"
LEXICAL = ROOT / "results/pure_pk_main_25rep_revised/tables/prompt_setting_means.csv"
OUT = ROOT / "analysis/semantic_diversity"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = ROOT / "models/fastembed"
NEAR_DUPLICATE_SIMILARITY = 0.95

sys.path.insert(0, str(ROOT / "scripts"))
from analyze_temperature_parameter_interactions import (  # noqa: E402
    bh_adjust,
    extreme_temperature_did,
    joint_test,
)


KEY_COLUMNS = [
    "experiment_id", "prompt_id", "repetition_id", "random_seed",
    "decoding_mode", "temperature", "top_p", "top_k", "generated_text",
]


def source_hash(row: pd.Series) -> str:
    payload = "\x1f".join(str(row[c]) for c in KEY_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_or_load(data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_path = OUT / "embeddings_float32.npy"
    key_path = OUT / "embedding_row_keys.csv"
    keys = data.apply(source_hash, axis=1)
    if embedding_path.exists() and key_path.exists():
        saved = pd.read_csv(key_path)
        embeddings = np.load(embedding_path, mmap_mode="r")
        if len(saved) != len(data) or embeddings.shape[0] != len(data):
            raise ValueError("Embedding cache row count mismatch")
        if not np.array_equal(saved["semantic_source_sha256"].to_numpy(), keys.to_numpy()):
            raise ValueError("Embedding cache keys do not match the current source")
        return np.asarray(embeddings), saved
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE))
    texts = data["generated_text"].fillna("").astype(str).tolist()
    embeddings = np.vstack(list(model.embed(texts, batch_size=256))).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)
    np.save(embedding_path, embeddings)
    saved = data[["prompt_id", "repetition_id", "decoding_mode", "temperature", "top_p", "top_k"]].copy()
    saved.insert(0, "semantic_source_sha256", keys)
    saved.to_csv(key_path, index=False)
    return embeddings, saved


def group_metrics(data: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    group_cols = ["decoding_mode", "temperature", "top_p", "top_k", "prompt_id"]
    for key, indices in data.groupby(group_cols, sort=True).indices.items():
        idx = np.asarray(indices)
        e = embeddings[idx]
        similarity = np.clip(e @ e.T, -1, 1)
        upper = similarity[np.triu_indices(len(e), k=1)]
        row = dict(zip(group_cols, key))
        row.update(
            n_answers=len(e),
            n_pairs=len(upper),
            semantic_cosine_distance=float(np.mean(1 - upper)),
            semantic_near_duplicate_rate=float(np.mean(upper >= NEAR_DUPLICATE_SIMILARITY)),
            semantic_similarity_mean=float(np.mean(upper)),
            semantic_similarity_sd=float(np.std(upper, ddof=1)),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def marginal_summary(prompt_settings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["semantic_cosine_distance", "semantic_near_duplicate_rate"]
    for mode, parameter in [("pure_top_p", "top_p"), ("pure_top_k", "top_k")]:
        subset = prompt_settings[prompt_settings["decoding_mode"] == mode]
        for dimension in ["temperature", parameter]:
            per_prompt = subset.groupby(["prompt_id", dimension], as_index=False)[metrics].mean()
            for value, g in per_prompt.groupby(dimension):
                for metric in metrics:
                    mean, lo, hi = mean_ci(g[metric])
                    rows.append(
                        {
                            "decoding_mode": mode,
                            "dimension": dimension,
                            "value": value,
                            "metric": metric,
                            "mean": mean,
                            "ci95_low": max(0.0, lo),
                            "ci95_high": min(1.0, hi),
                            "n_prompts": len(g),
                        }
                    )
    return pd.DataFrame(rows)


def plot_curves(prompt_settings: pd.DataFrame, mode: str, parameter: str) -> None:
    subset = prompt_settings[prompt_settings["decoding_mode"] == mode]
    means = subset.groupby(["temperature", parameter], as_index=False)["semantic_cosine_distance"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for value, g in means.groupby(parameter):
        ax.plot(g["temperature"], g["semantic_cosine_distance"], marker="o", label=f"{parameter.replace('_', '-')}={value:g}")
    ax.set(
        xlabel="Temperature",
        ylabel="Mean pairwise embedding cosine distance",
        title=f"Semantic diversity across temperature -- {mode.replace('_', ' ')}",
    )
    temperatures = sorted(means["temperature"].unique())
    ax.set_xticks(temperatures, [f"{value:g}" for value in temperatures])
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    stem = OUT / "figures" / f"{mode}_temperature_semantic_cosine_distance"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparison = [
        "unique_answer_rate",
        "pairwise_token_jaccard_distance",
        "corpus_distinct_1",
        "within_answer_distinct_1",
        "word_length",
    ]
    for mode in ["all", "pure_top_p", "pure_top_k"]:
        g = merged if mode == "all" else merged[merged["decoding_mode"] == mode]
        for metric in comparison:
            rho, pvalue = spearmanr(g["semantic_cosine_distance"], g[metric], nan_policy="omit")
            rows.append({"decoding_mode": mode, "comparison_metric": metric, "spearman_rho": rho, "p_value": pvalue, "n": g[["semantic_cosine_distance", metric]].dropna().shape[0]})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    data = pd.read_csv(INPUT, low_memory=False)
    if len(data) != 56250:
        raise ValueError(f"Expected 56,250 rows, found {len(data)}")
    group_cols = ["decoding_mode", "temperature", "top_p", "top_k", "prompt_id"]
    sizes = data.groupby(group_cols).size()
    if len(sizes) != 2250 or not sizes.eq(25).all():
        raise ValueError("Expected 2,250 prompt-settings with 25 answers each")
    embeddings, keys = embed_or_load(data)
    if embeddings.shape != (56250, 384):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    prompt_settings = group_metrics(data, embeddings)
    lexical = pd.read_csv(LEXICAL)
    merged = lexical.merge(prompt_settings, on=group_cols, validate="one_to_one")
    summary = marginal_summary(merged)
    corr = correlations(merged)
    tests = []
    did_rows = []
    for mode, parameter in [("pure_top_p", "top_p"), ("pure_top_k", "top_k")]:
        subset = merged[merged["decoding_mode"] == mode]
        for metric in ["semantic_cosine_distance", "semantic_near_duplicate_rate"]:
            tests.append(joint_test(subset, metric, parameter))
            did_rows.extend(extreme_temperature_did(subset, metric, parameter))
    tests = pd.DataFrame(tests)
    tests["p_value_bh_within_family"] = tests.groupby("interaction")["p_value"].transform(bh_adjust)
    did = pd.DataFrame(did_rows)
    prompt_settings.to_csv(OUT / "prompt_setting_semantic_diversity.csv", index=False)
    merged.to_csv(OUT / "prompt_setting_metrics_with_semantic.csv", index=False)
    summary.to_csv(OUT / "semantic_marginal_summary.csv", index=False)
    corr.to_csv(OUT / "semantic_lexical_correlations.csv", index=False)
    tests.to_csv(OUT / "semantic_interaction_joint_tests.csv", index=False)
    did.to_csv(OUT / "semantic_extreme_temperature_difference_in_differences.csv", index=False)
    plot_curves(merged, "pure_top_p", "top_p")
    plot_curves(merged, "pure_top_k", "top_k")
    manifest = {
        "input": str(INPUT),
        "input_rows": len(data),
        "prompt_settings": len(prompt_settings),
        "answers_per_prompt_setting": 25,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": embeddings.shape[1],
        "embedding_cache": str(OUT / "embeddings_float32.npy"),
        "metric": "Mean of 1 - cosine similarity over all 300 answer pairs within each prompt-setting",
        "near_duplicate_threshold": NEAR_DUPLICATE_SIMILARITY,
        "judge_independent": True,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(data), "prompt_settings": len(prompt_settings), "embedding_shape": list(embeddings.shape)}, indent=2))
    print("\nInteraction tests")
    print(tests[["interaction", "metric", "wald_f", "df", "p_value", "p_value_bh_within_family", "partial_r2"]].to_string(index=False))
    print("\nCorrelations")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
