"""Run the full SEP-1 uncertainty-aware RAG pipeline and score it against ground truth.

With the synthetic fixtures, the resulting numbers are NOT clinically meaningful --
they only confirm the pipeline's logic (chunking -> retrieval -> self-consistency ->
confidence gating -> scoring) runs end-to-end correctly. Meaningful accuracy/coverage
numbers require running this against real MIMIC-IV-Note data.
"""

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CONFIDENCE_BUCKETS,
    EVAL_DETAILED_PATH,
    EVAL_SUMMARY_PATH,
    GROUND_TRUTH_PATH,
    SEP1_QUESTIONS,
    TOP_K,
)
from data_loader import chunk_notes, load_notes  # noqa: E402
from retrieval import build_index, search  # noqa: E402
from uncertainty import assess  # noqa: E402


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> dict:
    ground_truth = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["subject_id"]), int(row["hadm_id"]), row["question_id"])
            ground_truth[key] = row["ground_truth"]
    return ground_truth


def is_correct(final_answer: str, ground_truth: str) -> bool:
    # "None" in ground truth means the note doesn't document enough to answer --
    # the correct pipeline behavior is to have flagged it INSUFFICIENT EVIDENCE.
    if ground_truth == "None":
        return final_answer == "INSUFFICIENT EVIDENCE"
    return final_answer == ground_truth


def run_pipeline() -> list[dict]:
    notes = load_notes()
    chunks = chunk_notes(notes)
    index = build_index(chunks)
    ground_truth = load_ground_truth()

    rows = []
    for note in notes:
        for question in SEP1_QUESTIONS:
            retrieved = search(
                question["text"],
                index,
                chunks,
                top_k=TOP_K,
                subject_id=note["subject_id"],
                hadm_id=note["hadm_id"],
            )
            result = assess(question, retrieved)
            gt = ground_truth[(note["subject_id"], note["hadm_id"], question["id"])]

            rows.append(
                {
                    "subject_id": note["subject_id"],
                    "hadm_id": note["hadm_id"],
                    "question_id": question["id"],
                    "question_text": question["text"],
                    "ground_truth": gt,
                    "final_answer": result["final_answer"],
                    "correct": is_correct(result["final_answer"], gt),
                    "retrieval_confidence": round(result["retrieval_confidence"], 4),
                    "self_consistency_ratio": round(result["self_consistency_ratio"], 4),
                    "combined_confidence": round(result["combined_confidence"], 4),
                    "confidence_bucket": result["confidence_bucket"],
                    "forced_insufficient": result["forced_insufficient"],
                    "justification": result["justification"],
                }
            )
    return rows


def save_detailed(rows: list[dict], path: Path = EVAL_DETAILED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for label, _, _ in CONFIDENCE_BUCKETS:
        bucket_rows = [r for r in rows if r["confidence_bucket"] == label]
        n = len(bucket_rows)
        accuracy = sum(r["correct"] for r in bucket_rows) / n if n else None
        insufficient_rate = (
            sum(r["final_answer"] == "INSUFFICIENT EVIDENCE" for r in bucket_rows) / n if n else None
        )
        summary.append(
            {
                "confidence_bucket": label,
                "n": n,
                "accuracy": round(accuracy, 4) if accuracy is not None else "",
                "insufficient_evidence_rate": round(insufficient_rate, 4)
                if insufficient_rate is not None
                else "",
            }
        )
    return summary


def save_summary(summary: list[dict], path: Path = EVAL_SUMMARY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def print_summary(summary: list[dict], overall_accuracy: float):
    print(f"\n{'Bucket':<10} {'n':>4} {'Accuracy':>10} {'Insuff.Evid.Rate':>18}")
    print("-" * 46)
    for row in summary:
        acc = f"{row['accuracy']:.2%}" if row["accuracy"] != "" else "n/a"
        rate = f"{row['insufficient_evidence_rate']:.2%}" if row["insufficient_evidence_rate"] != "" else "n/a"
        print(f"{row['confidence_bucket']:<10} {row['n']:>4} {acc:>10} {rate:>18}")
    print("-" * 46)
    print(f"Overall accuracy: {overall_accuracy:.2%}\n")


def main():
    rows = run_pipeline()
    save_detailed(rows)

    summary = summarize(rows)
    save_summary(summary)

    overall_accuracy = sum(r["correct"] for r in rows) / len(rows)
    print_summary(summary, overall_accuracy)

    print(f"Saved detailed results to {EVAL_DETAILED_PATH}")
    print(f"Saved summary to {EVAL_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
