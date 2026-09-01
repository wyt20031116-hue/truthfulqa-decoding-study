#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "outputs/qwen3_pilot_150/judgments.csv"
COMPARISON = ROOT / "outputs/qwen3_pilot_150/comparison_with_manual_by_prompt.csv"
OUT_DIR = ROOT / "outputs/qwen3_pilot_150"
OUT_CSV = OUT_DIR / "qwen3_priority_40row_adjudication.csv"
OUT_JSON = OUT_DIR / "qwen3_priority_40row_adjudication_meta.json"

# Largest absolute prompt-level strict-count discrepancies, descending.
PRIORITY_PROMPTS = [10, 12, 1, 3]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    judgments = read_csv(JUDGMENTS)
    comparison = {int(r["prompt_id"]): r for r in read_csv(COMPARISON)}
    priority_rank = {prompt_id: i + 1 for i, prompt_id in enumerate(PRIORITY_PROMPTS)}

    rows = []
    for row in judgments:
        prompt_id = int(row["prompt_id"])
        if prompt_id not in priority_rank:
            continue
        manual = comparison[prompt_id]
        rows.append({
            "priority_rank": priority_rank[prompt_id],
            "prompt_id": prompt_id,
            "repetition_id": int(row["repetition_id"]),
            "question": row["question"],
            "generated_text": row["generated_text"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
            "manual_prompt_strict_correct_n": int(manual["strict_correct_n"]),
            "qwen3_prompt_strict_correct_n": int(manual["qwen3_strict_correct_n"]),
            "prompt_strict_count_difference": int(manual["strict_count_difference"]),
            "manual_prompt_main_finding": manual["main_finding"],
            "qwen3_correctness": float(row["correctness"]),
            "qwen3_correctness_reason": row["correctness_reason"],
            "qwen3_informativeness": float(row["informativeness"]),
            "qwen3_informativeness_reason": row["informativeness_reason"],
            "reviewer_final_correctness": "",
            "reviewer_correctness_reason": "",
            "reviewer_final_informativeness": "",
            "reviewer_informativeness_reason": "",
            "reviewer_status": "Pending",
        })

    rows.sort(key=lambda r: (r["priority_rank"], r["repetition_id"]))
    assert len(rows) == 40, len(rows)
    assert len({(r["prompt_id"], r["repetition_id"]) for r in rows}) == 40
    assert all(sum(r["prompt_id"] == p for r in rows) == 10 for p in PRIORITY_PROMPTS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for p in PRIORITY_PROMPTS:
        r = comparison[p]
        summary.append({
            "priority_rank": priority_rank[p],
            "prompt_id": p,
            "question": r["question"],
            "manual_strict_correct_n": int(r["strict_correct_n"]),
            "qwen3_strict_correct_n": int(r["qwen3_strict_correct_n"]),
            "strict_count_difference": int(r["strict_count_difference"]),
            "qwen3_mean_correctness": float(r["qwen3_mean_correctness"]),
            "qwen3_mean_informativeness": float(r["qwen3_mean_informativeness"]),
            "manual_main_finding": r["main_finding"],
        })
    OUT_JSON.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} review rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
