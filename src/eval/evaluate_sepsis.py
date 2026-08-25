"""Run the SEP-1 uncertainty-aware RAG pipeline and score it against ground truth (Step 5).

Parallel to evaluate_faithfulness.py, but for the SEP-1 sepsis-notes task: instead
of RAGAS faithfulness, this scores answer correctness against ground truth, bucketed
by the combined confidence score from uncertainty.py, to check that confidence
tracks accuracy.

With the synthetic fixtures (data/raw/sepsis_notes_synthetic.jsonl), the resulting
numbers are NOT clinically meaningful -- they only confirm the pipeline's logic
(chunking -> retrieval -> self-consistency -> confidence gating -> scoring) runs
end-to-end correctly. Meaningful accuracy/coverage numbers require running this
against real MIMIC-IV-Note data (see generate_sepsis_notes.py's docstring).
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation.generate_sepsis import SEP1_QUESTIONS  # noqa: E402
from retrieval.retrieve_sepsis import retrieve_sepsis  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from uncertainty import CONFIDENCE_BUCKETS, assess  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
NOTES_PATH = BASE_DIR / "data" / "raw" / "sepsis_notes_synthetic.jsonl"
GROUND_TRUTH_PATH = BASE_DIR / "data" / "raw" / "sepsis_ground_truth.jsonl"
DETAILED_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sepsis_eval_results.jsonl"
SUMMARY_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sepsis_eval_summary.jsonl"

TOP_K = 5


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_correct(final_answer: str, ground_truth: str) -> bool:
    # "None" in ground truth means the note doesn't document enough to answer --
    # the correct pipeline behavior is to have flagged it INSUFFICIENT EVIDENCE.
    if ground_truth == "None":
        return final_answer == "INSUFFICIENT EVIDENCE"
    return final_answer == ground_truth


def run_pipeline() -> list[dict]:
    notes = load_jsonl(NOTES_PATH)
    ground_truth = {(row["subject_id"], row["hadm_id"]): row["ground_truth"] for row in load_jsonl(GROUND_TRUTH_PATH)}

    rows = []
    for note in notes:
        gt_for_note = ground_truth[(note["subject_id"], note["hadm_id"])]
        for question in SEP1_QUESTIONS:
            retrieved = retrieve_sepsis(
                question["text"], note["subject_id"], note["hadm_id"], top_k=TOP_K
            )
            result = assess(question, retrieved)
            gt = gt_for_note[question["id"]]

            rows.append(
                {
                    "subject_id": note["subject_id"],
                    "hadm_id": note["hadm_id"],
                    "question_id": question["id"],
                    "question_text": question["text"],
                    "ground_truth": gt,
                    "final_answer": result["final_answer"],
                    "correct": is_correct(result["final_answer"], gt),
                    "retrieval_confidence": result["retrieval_confidence"],
                    "self_consistency_ratio": result["self_consistency_ratio"],
                    "combined_confidence": result["combined_confidence"],
                    "confidence_bucket": result["confidence_bucket"],
                    "forced_insufficient": result["forced_insufficient"],
                    "justification": result["justification"],
                }
            )
    return rows


def save_results(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
                "accuracy": accuracy,
                "insufficient_evidence_rate": insufficient_rate,
            }
        )
    return summary


def print_summary(summary: list[dict], overall_accuracy: float):
    print(f"\n{'Bucket':<10} {'n':>4} {'Accuracy':>10} {'Insuff.Evid.Rate':>18}")
    print("-" * 46)
    for row in summary:
        acc = f"{row['accuracy']:.2%}" if row["accuracy"] is not None else "n/a"
        rate = (
            f"{row['insufficient_evidence_rate']:.2%}"
            if row["insufficient_evidence_rate"] is not None
            else "n/a"
        )
        print(f"{row['confidence_bucket']:<10} {row['n']:>4} {acc:>10} {rate:>18}")
    print("-" * 46)
    print(f"Overall accuracy: {overall_accuracy:.2%}")


def main():
    rows = run_pipeline()
    save_results(rows, DETAILED_OUTPUT_PATH)

    summary = summarize(rows)
    save_results(summary, SUMMARY_OUTPUT_PATH)

    overall_accuracy = sum(r["correct"] for r in rows) / len(rows)
    print_summary(summary, overall_accuracy)

    print(f"\nSaved detailed results to {DETAILED_OUTPUT_PATH}")
    print(f"Saved summary to {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
