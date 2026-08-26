"""Compute forced-choice accuracy from existing eval results, no new API calls.

Checks the claim that the confidence gate in uncertainty.py reduces the risk of
confidently-wrong answers, by comparing:
  - current (gating) accuracy: accuracy on questions the system actually answered
  - forced-choice accuracy: accuracy if the gate were never allowed to defer,
    using the model's real pre-gate majority-vote answer wherever it was recorded

Uses `majority_answer` / `self_consistency_answers` if present in the results
file (added to uncertainty.py / evaluate_sepsis.py so future evaluate_sepsis.py
runs record them). If a row predates that fix (final_answer was overwritten to
"INSUFFICIENT EVIDENCE" with no majority_answer saved), this script reports it
as unrecoverable rather than guessing, and prints exactly how many targeted
re-calls (n_rows x SELF_CONSISTENCY_RUNS) would be needed to fill the gap.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_PATH = BASE_DIR / "data" / "processed" / "sepsis_eval_results.jsonl"

sys.path.append(str(Path(__file__).resolve().parent))
from uncertainty import SELF_CONSISTENCY_RUNS  # noqa: E402


def load_rows():
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pct(x, n):
    return f"{x / n:.2%}" if n else "n/a"


def main():
    rows = load_rows()
    n_total = len(rows)

    # Only rows with a real Yes/No ground truth have a well-defined "correct answer"
    # to force a choice against -- forcing a guess on a ground_truth=None row (the
    # note genuinely doesn't document that fact) has no correct answer by definition,
    # so those are reported separately, not folded into forced-choice accuracy.
    answerable = [r for r in rows if r["ground_truth"] in ("Yes", "No")]
    none_gt = [r for r in rows if r["ground_truth"] == "None"]

    answered = [r for r in answerable if r["final_answer"] in ("Yes", "No")]
    gated = [r for r in answerable if r["forced_insufficient"]]
    genuine_refusal = [r for r in answerable if r["final_answer"] == "INSUFFICIENT EVIDENCE" and not r["forced_insufficient"]]

    gated_on_none = [r for r in none_gt if r["forced_insufficient"]]

    print(f"Total eval rows: {n_total}")
    print(f"  ground_truth = Yes/No (a real answer exists): {len(answerable)}")
    print(f"  ground_truth = None (note genuinely lacks the info): {len(none_gt)}")
    print()
    print("Note on README's published 90.33%: that number is OVERALL accuracy across")
    print("all 300 rows, where answering INSUFFICIENT EVIDENCE on a ground_truth=None")
    print("row counts as correct. It is not \"accuracy restricted to answered questions\"")
    print("(that number is different, computed below).")
    print()

    print("=== Among the 181 questions with a real Yes/No ground truth ===")
    print(f"Answered (Yes/No): {len(answered)}")
    print(f"  Current (gating) accuracy, answered-only: {pct(sum(r['correct'] for r in answered), len(answered))} (n={len(answered)})")
    print(f"Gated to INSUFFICIENT EVIDENCE by the confidence threshold: {len(gated)}")
    print(f"Genuinely answered INSUFFICIENT EVIDENCE by the model itself (not gated): {len(genuine_refusal)}")
    print()
    print(f"(For reference: the gate also fired on {len(gated_on_none)}/{len(none_gt)} ground_truth=None rows --")
    print(" deferring there was the objectively correct call, not a cost.)")
    print()

    print("=== Opportunity cost of gating (the numbers this analysis needs) ===")
    if not gated:
        print("No gated rows with a real ground truth in this dataset.")
        return

    has_majority_field = all("majority_answer" in r for r in gated)
    if has_majority_field:
        would_be_correct = [r for r in gated if r["majority_answer"] == r["ground_truth"]]
        would_be_wrong = [r for r in gated if r["majority_answer"] != r["ground_truth"]]
        n = len(gated)
        print(f"Of {n} gated rows with a real ground truth:")
        print(f"  Would have been CORRECT if not gated: {len(would_be_correct)} ({pct(len(would_be_correct), n)}) -- opportunity cost of deferring")
        print(f"  Would have been WRONG if not gated:   {len(would_be_wrong)} ({pct(len(would_be_wrong), n)}) -- real benefit of deferring")
        print()
        forced_choice_answers = list(answered)
        for r in gated:
            forced_choice_answers.append({**r, "correct": r["majority_answer"] == r["ground_truth"]})
        for r in genuine_refusal:
            if "majority_answer" in r and r["majority_answer"] in ("Yes", "No"):
                forced_choice_answers.append({**r, "correct": r["majority_answer"] == r["ground_truth"]})
            else:
                print(f"  Note: row {r['subject_id']}/{r['hadm_id']}/{r['question_id']} was a genuine model "
                      f"refusal with majority_answer=INSUFFICIENT EVIDENCE -- no Yes/No to force, excluded.")
        n_fc = len(forced_choice_answers)
        acc_fc = sum(r["correct"] for r in forced_choice_answers) / n_fc
        acc_cur = sum(r["correct"] for r in answered) / len(answered)
        print(f"Current (gating) accuracy:  {acc_cur:.2%} (n={len(answered)}, answered-only)")
        print(f"Forced-choice accuracy:     {acc_fc:.2%} (n={n_fc}, gate disabled for these {len(gated)} rows)")
        print(f"Difference (current - forced): {acc_cur - acc_fc:+.2%}")
    else:
        print(f"CANNOT compute: {len(gated)} rows were gated to INSUFFICIENT EVIDENCE, but this")
        print("results file predates the majority_answer/self_consistency_answers fields")
        print("(uncertainty.py only stored the post-gate final_answer and the agreement")
        print("ratio, not the actual pre-gate majority vote or the 5 raw samples).")
        print()
        print(f"To recover this precisely: re-run self-consistency generation for just")
        print(f"these {len(gated)} (question, note) pairs -- {len(gated)} x {SELF_CONSISTENCY_RUNS} = "
              f"{len(gated) * SELF_CONSISTENCY_RUNS} new API calls, not the full 1,500.")
        print("(Re-running is a fresh temperature>0.7 measurement, not a replay of the")
        print("original -- the exact original sample can't be replayed since it wasn't logged.)")
        print()
        print("Rows affected:")
        for r in gated:
            print(f"  {r['subject_id']}/{r['hadm_id']} {r['question_id']}: ground_truth={r['ground_truth']}, "
                  f"self_consistency_ratio={r['self_consistency_ratio']}, combined_confidence={r['combined_confidence']:.3f}")


if __name__ == "__main__":
    main()
