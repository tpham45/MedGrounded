"""Uncertainty / confidence scoring -- the core of this extension.

Inspired by COMPOSER's (Boussina et al., npj Digital Medicine 2024) use of
conformal prediction to let a sepsis model "know when it doesn't know" and
flag a case as indeterminate rather than guess. This module is a much simpler
analog for a RAG QA setting: it isn't conformal prediction (no calibration set
with a formal coverage guarantee), just a same-spirit heuristic gate built from
two signals that are cheap to compute for a small text-QA pipeline:

1. Retrieval confidence: how well the retrieved context actually matches the
   question (top cosine similarity from retrieval.py).
2. Self-consistency: how often independent, temperature>0 samples of the LLM
   agree with each other. Low agreement is itself evidence the model is
   uncertain, regardless of how confident any single answer sounds.

Combining them and thresholding low scores to a forced "INSUFFICIENT EVIDENCE"
answer is the "know when you don't know" gate for this project.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CONFIDENCE_BUCKETS,
    LOW_CONFIDENCE_THRESHOLD,
    RETRIEVAL_CONFIDENCE_WEIGHT,
    SELF_CONSISTENCY_RUNS,
    SELF_CONSISTENCY_TEMPERATURE,
    SELF_CONSISTENCY_WEIGHT,
)
from generation import generate_answer  # noqa: E402


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
    runs = [generate_answer(question, context_texts, temperature=temperature) for _ in range(n_runs)]
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
    return CONFIDENCE_BUCKETS[-1][0]  # fall back to the top bucket for score == 1.0 edge cases


def assess(question: dict, retrieved_chunks: list[dict]) -> dict:
    """Run the full uncertainty-aware answer pipeline for one (question, context) pair.

    Returns the final (possibly overridden) answer plus every intermediate signal,
    so evaluate.py can report on retrieval confidence, self-consistency, the
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
        "justification": consistency["justification"],
        "retrieval_confidence": r_conf,
        "self_consistency_ratio": consistency["agreement_ratio"],
        "self_consistency_answers": consistency["all_answers"],
        "combined_confidence": combined,
        "confidence_bucket": confidence_bucket(combined),
        "forced_insufficient": forced_insufficient,
    }
