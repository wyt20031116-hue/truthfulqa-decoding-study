#!/usr/bin/env python3
"""Run and validate a true greedy control for the low-k follow-up.

This control uses the same 50 questions, chat prompt, model, BF16 dtype,
repetition penalty, and length limit as the low-k extension. The only decoding
change is ``do_sample=False``. Five repeated calls per question are retained so
that determinism is tested rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default="runs/main_50q_45pure_settings_25rep/sampled_questions.csv",
    )
    parser.add_argument(
        "--low-k-generations",
        default="runs/low_k_extension_50q_k1_2_5_5rep/generations.csv",
    )
    parser.add_argument("--output-dir", default="runs/greedy_control_50q_5rep")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def qwen_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    questions_path = root / args.questions
    low_k_path = root / args.low_k_generations
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generations.csv"

    questions = pd.read_csv(questions_path, low_memory=False)
    required_question_columns = {"prompt_id", "source_row_id", "Question"}
    missing = sorted(required_question_columns - set(questions.columns))
    if missing:
        raise ValueError(f"Question frame is missing columns: {missing}")
    if len(questions) != 50 or questions["prompt_id"].nunique() != 50:
        raise ValueError("Greedy control requires the exact 50-question frame")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    commit = getattr(model.config, "_commit_hash", None)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for _, question_row in questions.sort_values("prompt_id").iterrows():
            prompt_id = int(question_row["prompt_id"])
            prompt = qwen_chat_prompt(str(question_row["Question"]))
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            prompt_tokens = int(encoded.input_ids.shape[1])
            input_hash = hashlib.sha256(
                encoded.input_ids.detach().cpu().numpy().tobytes()
            ).hexdigest()

            for repetition_id in range(args.repetitions):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_eos_token_id=tokenizer.eos_token_id,
                )
                if args.device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                token_ids = generated[0, prompt_tokens:].detach().cpu().tolist()
                text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                ended_with_eos = bool(
                    token_ids and token_ids[-1] == tokenizer.eos_token_id
                )
                rows.append(
                    {
                        "model_name": args.model_name,
                        "resolved_model_commit": commit,
                        "prompt_id": prompt_id,
                        "source_row_id": int(question_row["source_row_id"]),
                        "question": question_row["Question"],
                        "decoding_mode": "greedy",
                        "do_sample": False,
                        "temperature": np.nan,
                        "top_p": np.nan,
                        "top_k": np.nan,
                        "repetition_id": repetition_id,
                        "repetition_penalty": args.repetition_penalty,
                        "max_new_tokens": args.max_new_tokens,
                        "input_ids_sha256": input_hash,
                        "generated_token_ids": json.dumps(token_ids),
                        "generated_text": text,
                        "new_tokens": len(token_ids),
                        "ended_with_eos": ended_with_eos,
                        "stopped_early_on_eos": bool(
                            ended_with_eos and len(token_ids) < args.max_new_tokens
                        ),
                        "word_length": len(text.split()),
                        "model_generation_latency_seconds": latency,
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    per_prompt = (
        result.groupby("prompt_id", as_index=False)
        .agg(
            question=("question", "first"),
            repetitions=("repetition_id", "nunique"),
            unique_token_sequences=("generated_token_ids", "nunique"),
            unique_texts=("generated_text", "nunique"),
            mean_latency_seconds=("model_generation_latency_seconds", "mean"),
        )
        .sort_values("prompt_id")
    )
    per_prompt.to_csv(output_dir / "determinism_by_prompt.csv", index=False)

    low_k = pd.read_csv(low_k_path, low_memory=False)
    k1 = low_k.loc[low_k["top_k"].eq(1) & low_k["top_p"].eq(1.0)].copy()
    greedy_one = result.loc[
        result["repetition_id"].eq(0), ["prompt_id", "generated_text"]
    ].rename(columns={"generated_text": "greedy_text"})
    comparison = k1.merge(greedy_one, on="prompt_id", how="left", validate="many_to_one")
    comparison["matches_greedy"] = comparison["generated_text"].eq(
        comparison["greedy_text"]
    )
    comparison.to_csv(output_dir / "k1_row_comparison.csv", index=False)

    comparison_by_prompt = (
        comparison.groupby("prompt_id", as_index=False)
        .agg(
            question=("question", "first"),
            k1_rows=("generated_text", "size"),
            k1_unique_texts=("generated_text", "nunique"),
            greedy_text=("greedy_text", "first"),
            k1_rows_matching_greedy=("matches_greedy", "sum"),
            greedy_is_observed_k1_variant=("matches_greedy", "any"),
        )
        .sort_values("prompt_id")
    )
    comparison_by_prompt["k1_greedy_match_rate"] = (
        comparison_by_prompt["k1_rows_matching_greedy"]
        / comparison_by_prompt["k1_rows"]
    )
    comparison_by_prompt.to_csv(
        output_dir / "k1_vs_greedy_by_prompt.csv", index=False
    )

    validation = {
        "design": "true_greedy_control",
        "questions_sha256": file_sha256(questions_path),
        "low_k_generations_sha256": file_sha256(low_k_path),
        "greedy_generations_sha256": file_sha256(output_path),
        "model_name": args.model_name,
        "resolved_model_commit": commit,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if args.device == "cuda"
            else platform.processor()
        ),
        "model_training_flag": bool(model.training),
        "n_rows": int(len(result)),
        "n_questions": int(result["prompt_id"].nunique()),
        "repetitions_per_question": args.repetitions,
        "prompts_with_multiple_token_sequences": int(
            per_prompt["unique_token_sequences"].gt(1).sum()
        ),
        "prompts_with_multiple_texts": int(per_prompt["unique_texts"].gt(1).sum()),
        "all_greedy_repetitions_identical": bool(
            per_prompt["unique_token_sequences"].eq(1).all()
        ),
        "k1_rows": int(len(comparison)),
        "k1_rows_matching_greedy": int(comparison["matches_greedy"].sum()),
        "k1_greedy_match_rate": float(comparison["matches_greedy"].mean()),
        "prompts_where_greedy_was_not_observed_in_k1": sorted(
            comparison_by_prompt.loc[
                ~comparison_by_prompt["greedy_is_observed_k1_variant"], "prompt_id"
            ].astype(int).tolist()
        ),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if not validation["all_greedy_repetitions_identical"]:
        raise RuntimeError("True greedy decoding was not token-identical across repetitions")


if __name__ == "__main__":
    main()
