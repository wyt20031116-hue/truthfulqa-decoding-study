#!/usr/bin/env python3
"""Compute Qwen2.5 reference-candidate binary, MC1, and MC2 scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from qwen25_reference_accuracy_common import (
    MODEL_NAME,
    MODEL_REVISION,
    answer_logits_and_targets,
    choose_device,
    load_frozen_questions,
    load_qwen,
    qwen_chat_prompt,
    sha256,
    split_answers,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-input",
        default="runs/main_50q_45pure_settings_25rep/sampled_questions.csv",
    )
    parser.add_argument("--output", default="outputs/qwen25_reference_accuracy/by_question.csv")
    parser.add_argument("--summary-output", default="outputs/qwen25_reference_accuracy/summary.csv")
    parser.add_argument("--metadata-output", default="outputs/qwen25_reference_accuracy/metadata.json")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def score_answer(model, tokenizer, prompt, answer, device):
    logits, targets, _, _ = answer_logits_and_targets(
        model, tokenizer, prompt, answer, device
    )
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    total = float(token_log_probs.sum().item())
    return total, total / int(targets.numel()), int(targets.numel())


def score_question(row, model, tokenizer, device):
    prompt = qwen_chat_prompt(row["Question"])
    correct = list(dict.fromkeys(split_answers(row["Correct Answers"])))
    incorrect = list(dict.fromkeys(split_answers(row["Incorrect Answers"])))
    candidates = list(dict.fromkeys(correct + incorrect + [
        row["Best Answer"], row["Best Incorrect Answer"]
    ]))
    scores = {
        answer: score_answer(model, tokenizer, prompt, answer, device)
        for answer in candidates
    }
    total = {answer: value[0] for answer, value in scores.items()}
    mean = {answer: value[1] for answer, value in scores.items()}
    best_correct = total[row["Best Answer"]]
    best_incorrect = total[row["Best Incorrect Answer"]]
    max_incorrect = max(total[answer] for answer in incorrect)
    correct_mass = logsumexp([total[answer] for answer in correct])
    incorrect_mass = logsumexp([total[answer] for answer in incorrect])
    denominator = logsumexp([correct_mass, incorrect_mass])
    return {
        "prompt_id": int(row["prompt_id"]),
        "source_row_id": int(row["source_row_id"]),
        "category": row["Category"],
        "question": row["Question"],
        "binary_correct": int(best_correct > best_incorrect),
        "binary_logprob_margin": best_correct - best_incorrect,
        "mc1_correct": int(best_correct > max_incorrect),
        "mc1_logprob_margin": best_correct - max_incorrect,
        "mc2_score": math.exp(correct_mass - denominator),
        "best_answer_logprob": best_correct,
        "best_incorrect_logprob": best_incorrect,
        "best_answer_mean_logprob": mean[row["Best Answer"]],
        "best_incorrect_mean_logprob": mean[row["Best Incorrect Answer"]],
        "n_correct_candidates": len(correct),
        "n_incorrect_candidates": len(incorrect),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    questions_path = root / args.questions_input
    questions = load_frozen_questions(questions_path)
    if args.limit:
        questions = questions.iloc[: args.limit].copy()
    device = choose_device(args.device)
    tokenizer, model, resolved = load_qwen(args.model_name, args.revision, device)
    rows = [
        score_question(row, model, tokenizer, device)
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Reference MC")
    ]
    result = pd.DataFrame(rows)
    output = root / args.output
    summary_output = root / args.summary_output
    metadata_output = root / args.metadata_output
    for path in [output, summary_output, metadata_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = pd.DataFrame([{
        "n_prompts": len(result),
        "binary_accuracy": result["binary_correct"].mean(),
        "mc1_accuracy": result["mc1_correct"].mean(),
        "mc2_score": result["mc2_score"].mean(),
        "binary_accuracy_se": result["binary_correct"].std(ddof=1) / np.sqrt(len(result)),
        "mc1_accuracy_se": result["mc1_correct"].std(ddof=1) / np.sqrt(len(result)),
        "mc2_score_se": result["mc2_score"].std(ddof=1) / np.sqrt(len(result)),
    }])
    summary.to_csv(summary_output, index=False)
    metadata = {
        "metric_scope": "reference-candidate likelihood; not generated-answer accuracy",
        "model_name": args.model_name,
        "requested_revision": args.revision,
        "resolved_revision": resolved,
        "prompt_template": "same Qwen chat prompt as generator",
        "questions_input": args.questions_input,
        "questions_sha256": sha256(questions_path),
        "n_prompts": len(result),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
