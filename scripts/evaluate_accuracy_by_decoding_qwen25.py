#!/usr/bin/env python3
"""Evaluate Qwen2.5 reference likelihood under observed decoding settings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from qwen25_reference_accuracy_common import (
    MODEL_NAME,
    MODEL_REVISION,
    answer_logits_and_targets,
    apply_repetition_penalty,
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
    parser.add_argument("--settings-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_settings(path: Path):
    columns = ["decoding_mode", "temperature", "top_p", "top_k"]
    frame = pd.read_csv(path, usecols=columns).drop_duplicates().sort_values(columns)
    if frame.duplicated(columns).any():
        raise ValueError("Duplicate decoding settings")
    settings = [
        (str(r.decoding_mode), float(r.temperature), float(r.top_p), int(r.top_k))
        for r in frame.itertuples(index=False)
    ]
    if not settings:
        raise ValueError("No decoding settings found")
    return settings


def score_under_settings(logits, targets, settings):
    token_count = int(targets.numel())
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    target_ranks = (sorted_indices == targets.unsqueeze(1)).to(torch.int64).argmax(dim=1)
    vocab = logits.shape[-1]
    row_ids = torch.arange(token_count, device=logits.device)
    result = {}
    for mode, temperature, top_p, top_k in settings:
        candidate_count = vocab if top_k == 0 else min(top_k, vocab)
        scaled = sorted_logits / temperature
        available = scaled[:, :candidate_count]
        if top_p >= 1.0:
            kept = torch.full(
                (token_count,), candidate_count, dtype=torch.long, device=logits.device
            )
        else:
            probs = torch.softmax(available, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            kept = ((cumulative - probs) < top_p).sum(dim=-1).clamp(min=1)
        included = target_ranks < kept
        target_scaled = logits[row_ids, targets] / temperature
        normalizers = torch.stack([
            torch.logsumexp(available[row, : int(kept[row])], dim=0)
            for row in range(token_count)
        ])
        token_log_probs = (target_scaled - normalizers).masked_fill(~included, -torch.inf)
        total = float(token_log_probs.sum().item())
        result[(mode, temperature, top_p, top_k)] = (
            total,
            total / token_count,
            token_count,
            int((~included).sum().item()),
        )
    return result


def stable_logsumexp(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def score_question(row, model, tokenizer, device, settings, repetition_penalty):
    prompt = qwen_chat_prompt(row["Question"])
    correct = list(dict.fromkeys(split_answers(row["Correct Answers"])))
    incorrect = list(dict.fromkeys(split_answers(row["Incorrect Answers"])))
    candidates = list(dict.fromkeys(correct + incorrect))
    scores = {}
    for answer in candidates:
        logits, targets, full_ids, prompt_length = answer_logits_and_targets(
            model, tokenizer, prompt, answer, device
        )
        logits = apply_repetition_penalty(
            logits, full_ids, prompt_length, repetition_penalty
        )
        scores[answer] = score_under_settings(logits, targets, settings)

    output = []
    for setting in settings:
        mode, temperature, top_p, top_k = setting
        correct_total = [scores[a][setting][0] for a in correct]
        incorrect_total = [scores[a][setting][0] for a in incorrect]
        correct_mean = [scores[a][setting][1] for a in correct]
        incorrect_mean = [scores[a][setting][1] for a in incorrect]
        max_correct, max_incorrect = max(correct_total), max(incorrect_total)
        max_correct_mean, max_incorrect_mean = max(correct_mean), max(incorrect_mean)
        total_defined = math.isfinite(max_correct) and math.isfinite(max_incorrect)
        mean_defined = math.isfinite(max_correct_mean) and math.isfinite(max_incorrect_mean)
        correct_mass = stable_logsumexp(correct_total)
        incorrect_mass = stable_logsumexp(incorrect_total)
        denominator = stable_logsumexp([correct_mass, incorrect_mass])
        mc2_defined = math.isfinite(correct_mass) and math.isfinite(incorrect_mass)
        output.append({
            "prompt_id": int(row["prompt_id"]),
            "source_row_id": int(row["source_row_id"]),
            "category": row["Category"],
            "question": row["Question"],
            "decoding_mode": mode,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_logprob_accuracy": int(max_correct > max_incorrect) if total_defined else float("nan"),
            "max_logprob_margin": max_correct - max_incorrect if total_defined else float("nan"),
            "max_logprob_score_defined": int(total_defined),
            "max_mean_logprob_accuracy": int(max_correct_mean > max_incorrect_mean) if mean_defined else float("nan"),
            "max_mean_logprob_score_defined": int(mean_defined),
            # MC2 is meaningful only when both reference classes retain at
            # least one finite-probability candidate.  Otherwise a value of
            # 0 or 1 merely reflects support truncation, not model preference.
            "mc2_truncated_score": math.exp(correct_mass - denominator) if mc2_defined else float("nan"),
            "mc2_score_defined": int(mc2_defined),
            "correct_candidates_with_finite_score": sum(math.isfinite(v) for v in correct_total),
            "incorrect_candidates_with_finite_score": sum(math.isfinite(v) for v in incorrect_total),
            "n_correct_candidates": len(correct),
            "n_incorrect_candidates": len(incorrect),
            "correct_reference_tokens_pruned": sum(scores[a][setting][3] for a in correct),
            "incorrect_reference_tokens_pruned": sum(scores[a][setting][3] for a in incorrect),
        })
    return output


def main():
    args = parse_args()
    if args.repetition_penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    root = Path(__file__).resolve().parents[1]
    question_path = root / args.questions_input
    settings_path = root / args.settings_input
    questions = load_frozen_questions(question_path)
    if args.limit:
        questions = questions.iloc[: args.limit].copy()
    settings = load_settings(settings_path)
    device = choose_device(args.device)
    tokenizer, model, resolved = load_qwen(args.model_name, args.revision, device)
    rows = []
    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Decoding likelihood"):
        rows.extend(score_question(
            row, model, tokenizer, device, settings, args.repetition_penalty
        ))
    result = pd.DataFrame(rows)
    group = ["decoding_mode", "temperature", "top_p", "top_k"]
    summary = result.groupby(group, as_index=False).agg(
        n_prompts=("prompt_id", "size"),
        max_logprob_accuracy=("max_logprob_accuracy", "mean"),
        max_logprob_defined_rate=("max_logprob_score_defined", "mean"),
        max_mean_logprob_accuracy=("max_mean_logprob_accuracy", "mean"),
        max_mean_logprob_defined_rate=("max_mean_logprob_score_defined", "mean"),
        mc2_truncated_score=("mc2_truncated_score", "mean"),
        mc2_defined_rate=("mc2_score_defined", "mean"),
        correct_finite_candidates_mean=("correct_candidates_with_finite_score", "mean"),
        incorrect_finite_candidates_mean=("incorrect_candidates_with_finite_score", "mean"),
        correct_reference_tokens_pruned_mean=("correct_reference_tokens_pruned", "mean"),
        incorrect_reference_tokens_pruned_mean=("incorrect_reference_tokens_pruned", "mean"),
    )
    output, summary_output, metadata_output = [
        root / value for value in [args.output, args.summary_output, args.metadata_output]
    ]
    for path in [output, summary_output, metadata_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary.to_csv(summary_output, index=False)
    metadata = {
        "metric_scope": "truncated reference-candidate likelihood; not generated-answer accuracy",
        "model_name": args.model_name,
        "requested_revision": args.revision,
        "resolved_revision": resolved,
        "prompt_template": "same Qwen chat prompt as generator",
        "repetition_penalty": args.repetition_penalty,
        "warper_order": ["repetition_penalty", "temperature", "top_k", "top_p"],
        "questions_input": args.questions_input,
        "questions_sha256": sha256(question_path),
        "settings_input": args.settings_input,
        "settings_sha256": sha256(settings_path),
        "n_prompts": len(questions),
        "n_settings": len(settings),
        "n_rows": len(result),
        "interpretation_warning": (
            "Accuracy is conditional on both a correct and an incorrect reference candidate "
            "retaining finite probability after truncation. Always report defined-rate coverage."
        ),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
