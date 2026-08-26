"""One-off backfill: recover the pre-gate majority answer for the specific rows
where uncertainty.py's confidence gate fired but the original run didn't persist
majority_answer/self_consistency_answers (fixed in uncertainty.py/evaluate_sepsis.py
for future runs -- see compute_forced_choice_accuracy.py's docstring).

Only touches rows matching TARGET_ROWS below (the 7 gated rows with a real Yes/No
ground truth, identified by compute_forced_choice_accuracy.py). Everything else
in sepsis_eval_results.jsonl -- including these rows' own final_answer,
self_consistency_ratio, combined_confidence, confidence_bucket, forced_insufficient --
is left untouched, so the numbers already published in README.md do not change.
This is 7 x SELF_CONSISTENCY_RUNS = 35 new API calls, a fresh measurement (not a
replay of the original, unlogged sample).
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_PATH = BASE_DIR / "data" / "processed" / "sepsis_eval_results.jsonl"
TOP_K = 5

sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation.generate_sepsis import SEP1_QUESTIONS  # noqa: E402
from retrieval.retrieve_sepsis import retrieve_sepsis  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from uncertainty import self_consistency  # noqa: E402

TARGET_ROWS = {
    (10003, 20003, "q1_antibiotics_3h"),
    (10004, 20004, "q1_antibiotics_3h"),
    (10005, 20005, "q1_antibiotics_3h"),
    (10010, 20010, "q2_blood_culture_before_antibiotics"),
    (10018, 20018, "q2_blood_culture_before_antibiotics"),
    (10043, 20043, "q1_antibiotics_3h"),
    (10055, 20055, "q1_antibiotics_3h"),
}

QUESTIONS_BY_ID = {q["id"]: q for q in SEP1_QUESTIONS}


def load_rows():
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_rows(rows):
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = load_rows()
    updated = 0

    for row in rows:
        key = (row["subject_id"], row["hadm_id"], row["question_id"])
        if key not in TARGET_ROWS:
            continue

        question = QUESTIONS_BY_ID[row["question_id"]]
        retrieved = retrieve_sepsis(question["text"], row["subject_id"], row["hadm_id"], top_k=TOP_K)
        context_texts = [c["text"] for c in retrieved]

        consistency = self_consistency(question, context_texts)
        row["majority_answer"] = consistency["majority_answer"]
        row["self_consistency_answers"] = consistency["all_answers"]

        print(
            f"{row['subject_id']}/{row['hadm_id']} {row['question_id']}: "
            f"ground_truth={row['ground_truth']}, recovered majority_answer={consistency['majority_answer']} "
            f"(samples: {consistency['all_answers']})"
        )
        updated += 1

    assert updated == len(TARGET_ROWS), f"expected to update {len(TARGET_ROWS)} rows, updated {updated}"
    save_rows(rows)
    print(f"\nBackfilled {updated} rows in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
