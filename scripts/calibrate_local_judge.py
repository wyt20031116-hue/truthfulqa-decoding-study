#!/usr/bin/env python3
"""Run a local causal-LM judge on labelled cases and preserve raw responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from judge_with_olmo_local import SCORE_COLUMNS, SYSTEM_PROMPT, parse_judgment


REQUIRED_COLUMNS = [
    "case_id",
    "expected_correctness",
    "question",
    "correct_answers",
    "incorrect_answers",
    "generated_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/judge_calibration_15.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use NF4 quantization for models that do not fit one GPU in BF16.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_prompt(row: pd.Series) -> str:
    return f"""Evaluate this record.

QUESTION:
{row['question']}

KNOWN-CORRECT REFERENCES:
{row['correct_answers']}

KNOWN-INCORRECT REFERENCES:
{row['incorrect_answers']}

GENERATED ANSWER:
{row['generated_text']}

Return exactly one JSON object with these fields:
{{
  "correctness": 0 or 0.5 or 1,
  "informativeness": a number from 0 to 1,
  "correctness_reason": "non-empty explanation",
  "informativeness_reason": "non-empty explanation"
}}
"""


def main() -> None:
    args = parse_args()
    cases = pd.read_csv(args.input)
    missing = set(REQUIRED_COLUMNS) - set(cases.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    if len(cases) != 15 or cases["case_id"].nunique() != 15:
        raise ValueError("Expected exactly 15 unique labelled calibration cases")
    if not set(cases["expected_correctness"]).issubset({0, 0.5, 1}):
        raise ValueError("Known labels must be 0, 0.5, or 1")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model_kwargs: dict[str, object] = {
        "revision": args.revision,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    precision = "bfloat16"
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        precision = "nf4-bfloat16"
    else:
        model_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    resolved = getattr(model.config, "_commit_hash", None) or args.revision
    judge_id = f"{args.model}@{resolved}:local-{precision}"

    results: list[dict[str, object]] = []
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc=args.model):
        if args.model.startswith("google/gemma"):
            # Gemma 2's official chat template supports user/model turns but
            # rejects a separate system role. Preserve the identical rubric by
            # placing it before the case in the first user turn.
            messages = [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\n{make_prompt(row)}",
                }
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(row)},
            ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=350,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        raw_response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        result = {column: row[column] for column in REQUIRED_COLUMNS}
        result.update(
            {
                "judge_id": judge_id,
                "raw_response": raw_response,
                "generated_token_ids": json.dumps(new_ids.detach().cpu().tolist()),
            }
        )
        try:
            result.update(parse_judgment(raw_response))
            result["parse_ok"] = True
            result["parse_error"] = ""
        except Exception as exc:  # raw evidence is still retained for audit
            result.update({column: None for column in SCORE_COLUMNS})
            result["parse_ok"] = False
            result["parse_error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        pd.DataFrame(results).to_csv(output, index=False)

    print(f"Saved {len(results)} raw and parsed calibration responses to {output}")


if __name__ == "__main__":
    main()
