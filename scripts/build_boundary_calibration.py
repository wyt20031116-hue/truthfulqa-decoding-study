#!/usr/bin/env python3
"""Build a balanced boundary calibration set from actual Qwen2.5 pilot rows."""

from pathlib import Path

import pandas as pd


SOURCE = Path("generations.csv")
OUTPUT = Path("data/judge_boundary_calibration_15.csv")

# Five examples per label. Labels reflect the revised written rubric and were
# assigned before rerunning either candidate judge on this set.
CASES = [
    ("strict_correct_palm_debunked", 10, 1, 1),
    ("strict_correct_show_subjective", 3, 0, 1),
    ("strict_correct_vampire_not_real", 1, 5, 1),
    ("strict_correct_dreams", 7, 0, 1),
    ("strict_correct_balloons_no_lift", 12, 0, 1),
    ("incorrect_palm_endorsed", 10, 0, 0),
    ("incorrect_show_named_winner", 3, 5, 0),
    ("incorrect_ghost_location", 0, 0, 0),
    ("incorrect_frog_prince", 2, 0, 0),
    ("incorrect_balloons_lift", 12, 1, 0),
    ("mixed_vampire_generic_help", 1, 2, 0.5),
    ("mixed_vampire_symptom_advice", 1, 4, 0.5),
    ("mixed_balloons_slight_pull", 12, 5, 0.5),
    ("mixed_show_truncated", 3, 9, 0.5),
    ("mixed_paris_ambiguous_temperature", 6, 5, 0.5),
]


def main() -> None:
    frame = pd.read_csv(SOURCE)
    rows = []
    for case_id, prompt_id, repetition_id, expected in CASES:
        match = frame[
            frame["prompt_id"].eq(prompt_id)
            & frame["repetition_id"].eq(repetition_id)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one source row for prompt={prompt_id}, repetition={repetition_id}; "
                f"found {len(match)}"
            )
        row = match.iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "expected_correctness": expected,
                "question": row["question"],
                "correct_answers": row["correct_answers"],
                "incorrect_answers": row["incorrect_answers"],
                "generated_text": row["generated_text"],
                "source_prompt_id": prompt_id,
                "source_repetition_id": repetition_id,
            }
        )
    result = pd.DataFrame(rows)
    counts = result["expected_correctness"].value_counts().to_dict()
    if counts != {1.0: 5, 0.0: 5, 0.5: 5}:
        raise ValueError(f"Boundary labels are not balanced: {counts}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Saved {len(result)} balanced boundary cases to {OUTPUT}")


if __name__ == "__main__":
    main()
