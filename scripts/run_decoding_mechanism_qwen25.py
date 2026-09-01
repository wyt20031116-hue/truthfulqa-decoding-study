#!/usr/bin/env python3
"""Measure how decoding controls alter the token distribution.

The experiment uses the existing 50-question frame.  For every generated token,
``model.generate(..., output_scores=True)`` returns the processed logits after
temperature scaling and the active top-p or top-k truncation.  We normalize the
finite entries and record entropy, effective support size, retained support size,
and maximum token probability.  Pure top-p and pure top-k remain separate arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
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
    parser.add_argument("--output-dir", default="runs/mechanism_50q_21settings_5rep")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_text(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def settings() -> list[dict[str, float | int | str]]:
    temperatures = (0.3, 0.7, 1.5)
    rows: list[dict[str, float | int | str]] = []
    for temperature in temperatures:
        for top_p in (0.6, 0.9, 0.95):
            rows.append(
                {
                    "decoding_mode": "pure_top_p",
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 0,
                }
            )
        for top_k in (1, 5, 10, 30):
            rows.append(
                {
                    "decoding_mode": "pure_top_k",
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": top_k,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    questions_path = root / args.questions
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(questions_path, low_memory=False).sort_values("prompt_id")
    if args.limit_questions is not None:
        questions = questions.head(args.limit_questions)
    required = {"prompt_id", "source_row_id", "Question"}
    missing = sorted(required - set(questions.columns))
    if missing:
        raise ValueError(f"Question frame is missing columns: {missing}")
    if questions["prompt_id"].duplicated().any():
        raise ValueError("Question frame contains duplicate prompt IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    commit = getattr(model.config, "_commit_hash", None)

    answer_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    design = settings()
    total = len(questions) * len(design) * args.repetitions

    with torch.inference_mode():
        progress = tqdm(total=total, desc="Mechanism generations")
        for _, question_row in questions.iterrows():
            prompt_id = int(question_row["prompt_id"])
            encoded = tokenizer(
                prompt_text(str(question_row["Question"])), return_tensors="pt"
            ).to(args.device)
            prompt_tokens = int(encoded.input_ids.shape[1])

            for setting_id, setting in enumerate(design):
                for repetition_id in range(args.repetitions):
                    row_seed = args.seed + prompt_id * 100_000 + setting_id * 1_000 + repetition_id
                    random.seed(row_seed)
                    np.random.seed(row_seed % (2**32 - 1))
                    torch.manual_seed(row_seed)
                    if args.device == "cuda":
                        torch.cuda.manual_seed_all(row_seed)
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=float(setting["temperature"]),
                        top_p=float(setting["top_p"]),
                        top_k=int(setting["top_k"]),
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        forced_eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - start

                    token_ids = generated.sequences[0, prompt_tokens:].detach().cpu().tolist()
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    generation_key = hashlib.sha256(
                        f"{prompt_id}|{setting_id}|{repetition_id}|{row_seed}".encode()
                    ).hexdigest()

                    entropies: list[float] = []
                    effective_supports: list[float] = []
                    retained_supports: list[int] = []
                    max_probabilities: list[float] = []
                    chosen_probabilities: list[float] = []
                    for step_id, (processed_logits, token_id) in enumerate(
                        zip(generated.scores, token_ids)
                    ):
                        logits = processed_logits[0].float()
                        finite = torch.isfinite(logits)
                        retained_support = int(finite.sum().item())
                        if retained_support < 1:
                            raise RuntimeError("A generation step has empty sampling support")
                        retained_logits = logits[finite]
                        probabilities = torch.softmax(retained_logits, dim=-1)
                        positive = probabilities > 0
                        entropy = float(
                            -(probabilities[positive] * probabilities[positive].log()).sum().item()
                        )
                        effective_support = float(math.exp(entropy))
                        max_probability = float(probabilities.max().item())
                        chosen_probability = float(torch.softmax(logits, dim=-1)[token_id].item())

                        entropies.append(entropy)
                        effective_supports.append(effective_support)
                        retained_supports.append(retained_support)
                        max_probabilities.append(max_probability)
                        chosen_probabilities.append(chosen_probability)
                        step_rows.append(
                            {
                                "generation_key": generation_key,
                                "prompt_id": prompt_id,
                                "setting_id": setting_id,
                                "repetition_id": repetition_id,
                                **setting,
                                "step_id": step_id,
                                "chosen_token_id": token_id,
                                "chosen_token": tokenizer.decode([token_id]),
                                "token_entropy_nats": entropy,
                                "effective_support_size": effective_support,
                                "retained_support_size": retained_support,
                                "max_token_probability": max_probability,
                                "chosen_token_probability": chosen_probability,
                                "chosen_is_eos": token_id == tokenizer.eos_token_id,
                            }
                        )

                    answer_rows.append(
                        {
                            "generation_key": generation_key,
                            "model_name": args.model_name,
                            "resolved_model_commit": commit,
                            "prompt_id": prompt_id,
                            "source_row_id": int(question_row["source_row_id"]),
                            "question": question_row["Question"],
                            "setting_id": setting_id,
                            **setting,
                            "repetition_id": repetition_id,
                            "seed": row_seed,
                            "generated_text": text,
                            "generated_token_ids": json.dumps(token_ids),
                            "new_tokens": len(token_ids),
                            "word_length": len(text.split()),
                            "stopped_early_on_eos": bool(
                                token_ids
                                and token_ids[-1] == tokenizer.eos_token_id
                                and len(token_ids) < args.max_new_tokens
                            ),
                            "model_generation_latency_seconds": latency,
                            "mean_token_entropy_nats": float(np.mean(entropies)),
                            "mean_effective_support_size": float(np.mean(effective_supports)),
                            "mean_retained_support_size": float(np.mean(retained_supports)),
                            "mean_max_token_probability": float(np.mean(max_probabilities)),
                            "mean_chosen_token_probability": float(np.mean(chosen_probabilities)),
                        }
                    )
                    progress.update(1)
        progress.close()

    answers = pd.DataFrame(answer_rows)
    steps = pd.DataFrame(step_rows)
    answers_path = output_dir / "generations_with_mechanism.csv"
    steps_path = output_dir / "token_steps.csv"
    answers.to_csv(answers_path, index=False)
    steps.to_csv(steps_path, index=False)

    expected_rows = len(questions) * 21 * args.repetitions
    k1_steps = steps.loc[
        steps.decoding_mode.eq("pure_top_k") & steps.top_k.eq(1)
    ].copy()
    k1_tied_steps = int(k1_steps["retained_support_size"].gt(1).sum())
    checks = {
        "expected_answer_rows": len(answers) == expected_rows,
        "unique_generation_keys": answers["generation_key"].nunique() == len(answers),
        "exact_step_count": len(steps) == int(answers["new_tokens"].sum()),
        "all_supports_nonempty": bool(steps["retained_support_size"].ge(1).all()),
        "all_probabilities_valid": bool(
            steps["max_token_probability"].between(0, 1).all()
            and steps["chosen_token_probability"].between(0, 1).all()
        ),
        "pure_top_p_disables_top_k": bool(
            answers.loc[answers.decoding_mode.eq("pure_top_p"), "top_k"].eq(0).all()
        ),
        "pure_top_k_disables_top_p": bool(
            answers.loc[answers.decoding_mode.eq("pure_top_k"), "top_p"].eq(1.0).all()
        ),
        # Hugging Face keeps all tokens tied at the kth logit threshold.  Thus
        # top-k=1 can legitimately have support >1 at exact ties; the forensic
        # control established that this is the source of its residual variation.
        "k1_support_never_empty": bool(k1_steps["retained_support_size"].ge(1).all()),
    }
    manifest = {
        "design": "token_distribution_mechanism_pilot",
        "questions_path": args.questions,
        "questions_sha256": sha256(questions_path),
        "n_questions": len(questions),
        "temperatures": [0.3, 0.7, 1.5],
        "pure_top_p_values": [0.6, 0.9, 0.95],
        "pure_top_k_values": [1, 5, 10, 30],
        "n_settings": 21,
        "repetitions": args.repetitions,
        "n_answer_rows": len(answers),
        "n_token_rows": len(steps),
        "k1_token_steps": int(len(k1_steps)),
        "k1_tied_token_steps": k1_tied_steps,
        "k1_tied_token_step_rate": (
            k1_tied_steps / len(k1_steps) if len(k1_steps) else None
        ),
        "k1_max_retained_support_size": int(
            k1_steps["retained_support_size"].max()
        ),
        "top_k_tie_rule": (
            "TopKLogitsWarper retains tokens equal to the kth-logit threshold, "
            "so top-k=1 may retain multiple exactly tied tokens."
        ),
        "score_definition": (
            "Processed generation logits after temperature scaling, repetition "
            "penalty, and top-p or top-k truncation; finite logits are renormalized."
        ),
        "model_name": args.model_name,
        "resolved_model_commit": commit,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "checks": checks,
        "valid": all(checks.values()),
    }
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    if not manifest["valid"]:
        raise RuntimeError("Mechanism experiment validation failed")


if __name__ == "__main__":
    main()
