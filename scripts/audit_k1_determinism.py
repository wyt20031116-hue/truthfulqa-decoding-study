#!/usr/bin/env python3
"""Forensic audit of the post-hoc ``top_k=1`` extension.

The audit has two parts:

1. Recompute exact-text variation from the saved 3,750-row low-k run.
2. Replay ``top_k=1`` on the four prompts that varied, while saving generated
   token IDs and the number of finite candidates left by the logits warpers at
   every generation step.

Hugging Face's top-k warper removes scores that are *strictly* below the kth
largest score. Consequently, k=1 can retain more than one token when the
largest (BF16) scores are exactly tied. This script measures that mechanism
directly and also runs greedy decoding as a deterministic comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = (
    "Answer the question directly and concisely. Give only the answer, "
    "without discussing these instructions."
)
DEFAULT_ANOMALOUS_PROMPTS = (1, 7, 26, 47)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations",
        default="runs/low_k_extension_50q_k1_2_5_5rep/generations.csv",
    )
    parser.add_argument(
        "--questions",
        default="runs/main_50q_45pure_settings_25rep/sampled_questions.csv",
    )
    parser.add_argument("--output-dir", default="analysis/k1_forensic")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--main-seed", type=int, default=443)
    parser.add_argument(
        "--prompt-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_ANOMALOUS_PROMPTS),
    )
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def qwen_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalized_text(text: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def audit_saved_rows(path: Path, output_dir: Path) -> dict[str, object]:
    data = pd.read_csv(path, low_memory=False)
    required = {
        "prompt_id",
        "temperature",
        "top_p",
        "top_k",
        "repetition_id",
        "question",
        "generated_text",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Saved generations are missing columns: {missing}")

    keys = ["prompt_id", "temperature", "top_p", "top_k", "repetition_id"]
    duplicate_keys = int(data.duplicated(keys).sum())
    k1 = data.loc[data["top_k"].eq(1) & data["top_p"].eq(1.0)].copy()
    k1["normalized_generated_text"] = k1["generated_text"].map(normalized_text)

    cells = (
        k1.groupby(["prompt_id", "temperature"], as_index=False)
        .agg(
            question=("question", "first"),
            repetitions=("repetition_id", "nunique"),
            exact_unique_n=("generated_text", "nunique"),
            normalized_unique_n=("normalized_generated_text", "nunique"),
        )
        .sort_values(["prompt_id", "temperature"])
    )
    cells["varied_exactly"] = cells["exact_unique_n"].gt(1)
    cells["formatting_only_variation"] = cells["exact_unique_n"].gt(
        cells["normalized_unique_n"]
    )
    cells.to_csv(output_dir / "saved_k1_cells.csv", index=False)

    varying_keys = cells.loc[cells["varied_exactly"], ["prompt_id", "temperature"]]
    variants = k1.merge(varying_keys, on=["prompt_id", "temperature"], how="inner")
    keep = [
        column
        for column in [
            "prompt_id",
            "source_row_id",
            "question",
            "temperature",
            "top_p",
            "top_k",
            "repetition_id",
            "random_seed",
            "new_tokens",
            "ended_with_eos",
            "generated_text",
            "normalized_generated_text",
        ]
        if column in variants.columns
    ]
    variants[keep].sort_values(
        ["prompt_id", "temperature", "repetition_id"]
    ).to_csv(output_dir / "saved_k1_variant_rows.csv", index=False)

    across_temperature = (
        k1.groupby("prompt_id", as_index=False)
        .agg(
            question=("question", "first"),
            rows=("generated_text", "size"),
            exact_unique_n=("generated_text", "nunique"),
            normalized_unique_n=("normalized_generated_text", "nunique"),
        )
        .sort_values("prompt_id")
    )
    across_temperature.to_csv(
        output_dir / "saved_k1_across_temperature.csv", index=False
    )

    return {
        "input": str(path),
        "all_low_k_rows": int(len(data)),
        "duplicate_experiment_keys": duplicate_keys,
        "k1_rows": int(len(k1)),
        "k1_cells": int(len(cells)),
        "k1_cells_with_exact_variation": int(cells["varied_exactly"].sum()),
        "k1_prompts_with_exact_variation": sorted(
            cells.loc[cells["varied_exactly"], "prompt_id"].unique().astype(int).tolist()
        ),
        "formatting_only_cells": int(cells["formatting_only_variation"].sum()),
    }


def score_diagnostics(scores: tuple[torch.Tensor, ...]) -> dict[str, object]:
    support_sizes: list[int] = []
    maximum_tie_sizes: list[int] = []
    for score in scores:
        row = score[0]
        finite = torch.isfinite(row)
        support_sizes.append(int(finite.sum().item()))
        finite_scores = row[finite]
        maximum_tie_sizes.append(
            int((finite_scores == finite_scores.max()).sum().item())
            if finite_scores.numel()
            else 0
        )
    multi_support_steps = [i for i, size in enumerate(support_sizes) if size > 1]
    return {
        "support_sizes": support_sizes,
        "maximum_tie_sizes": maximum_tie_sizes,
        "multi_support_steps": multi_support_steps,
        "first_multi_support_step": multi_support_steps[0] if multi_support_steps else None,
        "max_support_size": max(support_sizes, default=0),
    }


def replay(
    questions_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    repetition_penalty: float,
    main_seed: int,
) -> dict[str, object]:
    questions = pd.read_csv(questions_path)
    selected = questions.loc[questions["prompt_id"].isin(prompt_ids)].copy()
    found = set(selected["prompt_id"].astype(int))
    if found != set(prompt_ids):
        raise ValueError(f"Missing requested prompt IDs: {sorted(set(prompt_ids) - found)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    commit = getattr(model.config, "_commit_hash", None)

    rows: list[dict[str, object]] = []
    temperatures = (0.1, 0.3, 0.7, 1.0, 1.5)
    with torch.inference_mode():
        for _, question_row in selected.sort_values("prompt_id").iterrows():
            prompt_id = int(question_row["prompt_id"])
            prompt = qwen_chat_prompt(str(question_row["Question"]))
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens = int(encoded.input_ids.shape[1])
            input_hash = hashlib.sha256(
                encoded.input_ids.detach().cpu().numpy().tobytes()
            ).hexdigest()

            for temperature in temperatures:
                for repetition_id in range(5):
                    seed = stable_seed(
                        main_seed,
                        prompt_id,
                        temperature,
                        1.0,
                        1,
                        repetition_id,
                    )
                    torch.manual_seed(seed)
                    if device == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        top_p=1.0,
                        top_k=1,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        forced_eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    token_ids = generated.sequences[0, prompt_tokens:].detach().cpu().tolist()
                    diagnostics = score_diagnostics(generated.scores)
                    rows.append(
                        {
                            "mode": "sample_top_k_1",
                            "prompt_id": prompt_id,
                            "question": question_row["Question"],
                            "temperature": temperature,
                            "repetition_id": repetition_id,
                            "random_seed": seed,
                            "input_ids_sha256": input_hash,
                            "generated_token_ids": json.dumps(token_ids),
                            "generated_text": tokenizer.decode(
                                token_ids, skip_special_tokens=True
                            ).strip(),
                            **{
                                key: json.dumps(value) if isinstance(value, list) else value
                                for key, value in diagnostics.items()
                            },
                        }
                    )

            greedy = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            token_ids = greedy.sequences[0, prompt_tokens:].detach().cpu().tolist()
            diagnostics = score_diagnostics(greedy.scores)
            rows.append(
                {
                    "mode": "greedy",
                    "prompt_id": prompt_id,
                    "question": question_row["Question"],
                    "temperature": np.nan,
                    "repetition_id": 0,
                    "random_seed": np.nan,
                    "input_ids_sha256": input_hash,
                    "generated_token_ids": json.dumps(token_ids),
                    "generated_text": tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ).strip(),
                    **{
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in diagnostics.items()
                    },
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "replay_rows.csv", index=False)
    sampled = frame.loc[frame["mode"].eq("sample_top_k_1")]
    prompt_summary = (
        sampled.groupby("prompt_id", as_index=False)
        .agg(
            question=("question", "first"),
            replay_rows=("generated_text", "size"),
            replay_unique_texts=("generated_text", "nunique"),
            rows_with_multi_token_support=(
                "max_support_size",
                lambda values: int((values > 1).sum()),
            ),
            maximum_observed_support=("max_support_size", "max"),
        )
        .sort_values("prompt_id")
    )
    prompt_summary.to_csv(output_dir / "replay_prompt_summary.csv", index=False)

    return {
        "model_name": model_name,
        "resolved_model_commit": commit,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "model_training_flag": bool(model.training),
        "prompt_ids": prompt_ids,
        "replay_rows": int(len(frame)),
        "sampled_rows": int(len(sampled)),
        "greedy_rows": int(frame["mode"].eq("greedy").sum()),
        "sampled_rows_with_multi_token_support": int(
            sampled["max_support_size"].gt(1).sum()
        ),
        "sampled_prompts_with_multiple_outputs": sorted(
            prompt_summary.loc[
                prompt_summary["replay_unique_texts"].gt(1), "prompt_id"
            ].astype(int).tolist()
        ),
        "top_k_implementation_note": (
            "Transformers removes scores strictly below the kth score; exact ties "
            "at the maximum can therefore leave support_size > 1 when top_k=1."
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_summary = audit_saved_rows(root / args.generations, output_dir)
    replay_summary = replay(
        root / args.questions,
        output_dir,
        args.model_name,
        args.device,
        args.prompt_ids,
        args.max_new_tokens,
        args.repetition_penalty,
        args.main_seed,
    )
    report = {
        "saved_run": saved_summary,
        "replay": replay_summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
