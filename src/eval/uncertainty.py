"""Uncertainty / confidence scoring for SEP-1 answers -- the core of this extension.

Inspired by COMPOSER's (Boussina et al., npj Digital Medicine 2024, Nemati Lab,
UCSD) use of conformal prediction to let a sepsis model "know when it doesn't
know" and flag a case as indeterminate rather than guess. This module is a much
simpler, same-spirit analog for a RAG QA setting rather than conformal
prediction itself (no calibration set with a formal coverage guarantee here) --
just a heuristic gate built from two signals that are cheap to compute for a
small text-QA pipeline:

1. Retrieval confidence: how well the retrieved context actually matches the
   question (top cosine similarity from retrieve_sepsis.py / retrieve.search()).
2. Self-consistency: how often independent, temperature>0 samples of the LLM
   agree with each other. Low agreement is itself evidence of uncertainty,
   regardless of how confident any single answer sounds.

Combining them and thresholding low scores to a forced "INSUFFICIENT EVIDENCE"
answer is the "know when you don't know" gate for this extension -- the same
"don't guess" principle generate.py/generate_sepsis.py already apply at the
single-answer level, extended here across repeated samples and retrieval quality.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation.generate_sepsis import generate_sepsis_answer  # noqa: E402

SELF_CONSISTENCY_RUNS = 5
# Must be > 0: self-consistency needs sampling variance to be meaningful. At
# temperature=0 every run would be identical and agreement would trivially be 100%.
SELF_CONSISTENCY_TEMPERATURE = 0.7

# Weighted combination of the two confidence signals (must sum to 1.0).
RETRIEVAL_CONFIDENCE_WEIGHT = 0.5
SELF_CONSISTENCY_WEIGHT = 0.5

# Below this combined score, the final answer is force-overridden to
# "INSUFFICIENT EVIDENCE" regardless of what generation/self-consistency produced.
LOW_CONFIDENCE_THRESHOLD = 0.6

# Bucket boundaries used by evaluate_sepsis.py to report accuracy-by-confidence.
# (label, lo, hi) as half-open [lo, hi) intervals over the combined score.
CONFIDENCE_BUCKETS = [
    ("Low", 0.0, 0.5),
    ("Medium", 0.5, 0.75),
    ("High", 0.75, 1.01),  # 1.01 so a perfect 1.0 score is inclusive
]


def retrieval_confidence(retrieved_chunks: list[dict]) -> float:
    """Top similarity score among retrieved chunks, already normalized to [0, 1]."""
    if not retrieved_chunks:
        return 0.0
    return max(chunk["score"] for chunk in retrieved_chunks)


def self_consistency(
    question: dict,
    context_texts: list[str],
    n_runs: int = SELF_CONSISTENCY_RUNS,
    temperature: float = SELF_CONSISTENCY_TEMPERATURE,
) -> dict:
    """Sample n_runs independent answers and measure agreement with the majority."""
    runs = [
        generate_sepsis_answer(question, context_texts, temperature=temperature) for _ in range(n_runs)
    ]
    answers = [r["answer"] for r in runs]

    majority_answer, majority_count = Counter(answers).most_common(1)[0]
    agreement_ratio = majority_count / len(answers)
    majority_justification = next(
        (r["justification"] for r in runs if r["answer"] == majority_answer), ""
    )

    return {
        "majority_answer": majority_answer,
        "agreement_ratio": agreement_ratio,
        "justification": majority_justification,
        "all_answers": answers,
    }


def combined_confidence(
    retrieval_conf: float,
    consistency_ratio: float,
    retrieval_weight: float = RETRIEVAL_CONFIDENCE_WEIGHT,
    consistency_weight: float = SELF_CONSISTENCY_WEIGHT,
) -> float:
    return retrieval_weight * retrieval_conf + consistency_weight * consistency_ratio


def confidence_bucket(score: float) -> str:
    for label, lo, hi in CONFIDENCE_BUCKETS:
        if lo <= score < hi:
            return label
    return CONFIDENCE_BUCKETS[-1][0]  # fall back to the top bucket for a 1.0 edge case


def assess(question: dict, retrieved_chunks: list[dict]) -> dict:
    """Run the full uncertainty-aware answer pipeline for one (question, context) pair.

    Returns the final (possibly overridden) answer plus every intermediate signal,
    so evaluate_sepsis.py can report on retrieval confidence, self-consistency, the
    combined score, and whether the low-confidence gate fired -- independently.
    """
    context_texts = [chunk["text"] for chunk in retrieved_chunks]

    r_conf = retrieval_confidence(retrieved_chunks)
    consistency = self_consistency(question, context_texts)
    combined = combined_confidence(r_conf, consistency["agreement_ratio"])

    forced_insufficient = combined < LOW_CONFIDENCE_THRESHOLD
    final_answer = "INSUFFICIENT EVIDENCE" if forced_insufficient else consistency["majority_answer"]

    return {
        "question_id": question["id"],
        "final_answer": final_answer,
        # The un-gated majority vote, kept distinct from final_answer so a forced
        # override doesn't erase what generation actually produced -- needed to
        # measure the gate's real effect (e.g. forced-choice accuracy analyses).
        "majority_answer": consistency["majority_answer"],
        "justification": consistency["justification"],
        "retrieval_confidence": r_conf,
        "self_consistency_ratio": consistency["agreement_ratio"],
        "self_consistency_answers": consistency["all_answers"],
        "combined_confidence": combined,
        "confidence_bucket": confidence_bucket(combined),
        "forced_insufficient": forced_insufficient,
    }
