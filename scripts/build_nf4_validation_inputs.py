#!/usr/bin/env python3
"""Create frozen, normalized validation inputs for the NF4 judge."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_SOURCE = ROOT / "outputs/judge_boundary_calibration/judge_boundary_calibration_adjudicated_15.csv"
PRIORITY_SOURCE = ROOT / "outputs/qwen3_pilot_150/qwen3_priority_40row_adjudication.csv"
OUTPUT_DIR = ROOT / "data/nf4_validation"


def main() -> None:
    boundary = pd.read_csv(BOUNDARY_SOURCE)
    boundary = boundary.rename(
        columns={
            "qwen25_generated_answer": "generated_text",
            "known_correct_references": "correct_answers",
            "known_incorrect_references": "incorrect_answers",
            "adjudicated_expected_correctness": "expected_correctness",
            "human_informativeness": "expected_informativeness",
            "qwen3_correctness": "bf16_correctness",
            "qwen3_informativeness": "bf16_informativeness",
        }
    )
    boundary_columns = [
        "case_id",
        "question",
        "correct_answers",
        "incorrect_answers",
        "generated_text",
        "expected_correctness",
        "expected_informativeness",
        "adjudication_reason",
        "human_informativeness_reason",
        "bf16_correctness",
        "bf16_informativeness",
    ]
    boundary = boundary[boundary_columns].copy()

    priority = pd.read_csv(PRIORITY_SOURCE)
    if not priority["reviewer_status"].eq("Reviewed").all():
        raise ValueError("All priority rows must be Reviewed before freezing NF4 inputs")
    priority = priority.rename(
        columns={
            "reviewer_final_correctness": "expected_correctness",
            "reviewer_final_informativeness": "expected_informativeness",
            "reviewer_correctness_reason": "adjudication_reason",
            "reviewer_informativeness_reason": "human_informativeness_reason",
            "qwen3_correctness": "bf16_correctness",
            "qwen3_informativeness": "bf16_informativeness",
        }
    )
    priority["case_id"] = priority.apply(
        lambda r: f"priority_p{int(r.prompt_id):02d}_r{int(r.repetition_id):02d}",
        axis=1,
    )
    priority_columns = [
        "case_id",
        "prompt_id",
        "repetition_id",
        "question",
        "correct_answers",
        "incorrect_answers",
        "generated_text",
        "expected_correctness",
        "expected_informativeness",
        "adjudication_reason",
        "human_informativeness_reason",
        "bf16_correctness",
        "bf16_informativeness",
    ]
    priority = priority[priority_columns].copy()

    assert len(boundary) == 15 and boundary["case_id"].nunique() == 15
    assert len(priority) == 40 and priority["case_id"].nunique() == 40
    for frame in (boundary, priority):
        assert frame["expected_correctness"].isin([0, 0.5, 1]).all()
        assert frame["expected_informativeness"].isin([0, 0.25, 0.5, 0.75, 1]).all()
        assert frame[["question", "correct_answers", "incorrect_answers", "generated_text"]].notna().all().all()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary.to_csv(OUTPUT_DIR / "boundary_adjudicated_15.csv", index=False)
    priority.to_csv(OUTPUT_DIR / "priority_adjudicated_40.csv", index=False)
    boundary_combined = boundary.copy()
    boundary_combined["validation_split"] = "boundary_adjudicated_15"
    priority_combined = priority.copy()
    priority_combined["validation_split"] = "priority_adjudicated_40"
    combined = pd.concat([boundary_combined, priority_combined], ignore_index=True, sort=False)
    assert len(combined) == 55 and combined["case_id"].nunique() == 55
    combined.to_csv(OUTPUT_DIR / "combined_adjudicated_55.csv", index=False)
    print(f"Wrote {len(boundary)} boundary and {len(priority)} priority validation rows")


if __name__ == "__main__":
    main()
