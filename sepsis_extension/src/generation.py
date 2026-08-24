"""Answer a SEP-1 bundle question from retrieved context only, via Claude.

Returns structured {"answer": "Yes" | "No" | "INSUFFICIENT EVIDENCE", "justification": str}.
Falls back to a rule-based mock when ANTHROPIC_API_KEY isn't set, so the rest of
the pipeline (retrieval -> uncertainty -> evaluation) can still be exercised
without API access.
"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LLM_MODEL  # noqa: E402

load_dotenv()

VALID_ANSWERS = {"Yes", "No", "INSUFFICIENT EVIDENCE"}

PROMPT_TEMPLATE = """You are a clinical evidence-review assistant checking SEP-1 sepsis \
bundle compliance from a clinical note excerpt. Answer the question using ONLY the \
context below -- do not use outside medical knowledge or assume anything not stated.

If the context does not document enough information to answer with confidence, you \
MUST answer "INSUFFICIENT EVIDENCE" rather than guessing.

Context:
{context}

Question: {question}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"answer": "Yes" | "No" | "INSUFFICIENT EVIDENCE", "justification": "<one short sentence>"}}"""

# --- Mock mode (no API key) --------------------------------------------------
# Deliberately simple keyword heuristics -- only meant to exercise the pipeline's
# control flow (chunking -> retrieval -> uncertainty -> evaluation), not to be a
# clinically meaningful answer. Real answers come from Claude in live mode.
_MOCK_TOPIC_KEYWORDS = {
    "q1_antibiotics_3h": ["antibiotic"],
    "q2_blood_culture_before_antibiotics": ["blood culture", "culture"],
    "q3_lactate_6h": ["lactate"],
    "q4_lactate_repeat": ["repeat lactate"],
    "q5_fluid_30ml_kg": ["fluid", "bolus", "crystalloid"],
}
_MOCK_NEGATION_KEYWORDS = {
    "q1_antibiotics_3h": ["not started", "delay", "deferred", "hours after presentation"],
    "q2_blood_culture_before_antibiotics": ["not obtain", "deferred", "were not"],
    "q3_lactate_6h": ["pending", "not yet available", "not documented"],
    "q4_lactate_repeat": ["not mentioned", "no repeat"],
    "q5_fluid_30ml_kg": ["no bolus", "not given", "not indicated", "normotensive"],
}


def _mock_generate(question: dict, context_texts: list[str], temperature: float) -> dict:
    combined = " ".join(context_texts).lower()
    topic_keywords = _MOCK_TOPIC_KEYWORDS[question["id"]]

    if not any(kw in combined for kw in topic_keywords):
        return {
            "answer": "INSUFFICIENT EVIDENCE",
            "justification": "Mock mode: context does not mention this topic.",
        }

    negation_keywords = _MOCK_NEGATION_KEYWORDS[question["id"]]
    base_answer = "No" if any(neg in combined for neg in negation_keywords) else "Yes"

    # Inject variance so self-consistency has something to measure in mock mode:
    # short/sparse context is treated as more ambiguous and flips more often.
    flip_prob = min(0.4, temperature * 0.3) if len(combined) < 300 else temperature * 0.05
    if random.random() < flip_prob:
        base_answer = "No" if base_answer == "Yes" else "Yes"

    return {
        "answer": base_answer,
        "justification": "Mock mode: keyword heuristic (not clinically validated).",
    }


def _parse_response(raw_text: str) -> dict:
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"answer": "INSUFFICIENT EVIDENCE", "justification": "Failed to parse model output."}
    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return {"answer": "INSUFFICIENT EVIDENCE", "justification": "Failed to parse model output."}

    answer = parsed.get("answer")
    if answer not in VALID_ANSWERS:
        return {"answer": "INSUFFICIENT EVIDENCE", "justification": "Model returned an unrecognized answer."}
    return {"answer": answer, "justification": str(parsed.get("justification", ""))[:500]}


_client = None
_warned_mock = False


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def is_mock_mode() -> bool:
    return not os.environ.get("ANTHROPIC_API_KEY")


def generate_answer(question: dict, context_texts: list[str], temperature: float = 0.0) -> dict:
    """Answer `question` (a SEP1_QUESTIONS entry) from `context_texts` only."""
    global _warned_mock
    if is_mock_mode():
        if not _warned_mock:
            print("[MOCK MODE] ANTHROPIC_API_KEY not set -- using rule-based mock generation.")
            _warned_mock = True
        return _mock_generate(question, context_texts, temperature)

    context = "\n\n".join(context_texts) if context_texts else "(no context retrieved)"
    prompt = PROMPT_TEMPLATE.format(context=context, question=question["text"])

    response = _get_client().messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text
    return _parse_response(raw_text)


if __name__ == "__main__":
    example_question = {
        "id": "q1_antibiotics_3h",
        "text": "Were broad-spectrum antibiotics administered within 3 hours of sepsis presentation?",
    }
    example_context = [
        "Antibiotics (vancomycin, piperacillin-tazobactam) started 45 minutes after presentation."
    ]
    print(generate_answer(example_question, example_context))
