"""Answer SEP-1 sepsis bundle compliance questions from retrieved note context via Claude.

Parallel to generate.py, but for the SEP-1 clinical-notes task instead of PubMedQA:
- Answers one of 5 fixed SEP-1 bundle questions (SEP1_QUESTIONS below).
- Returns structured {"answer": ..., "justification": ...} instead of free text, so
  src/eval/uncertainty.py can build on the answer programmatically.
- Same "don't guess" rule as generate.py's PROMPT_TEMPLATE: if the context doesn't
  document enough to answer, the model must say so -- "INSUFFICIENT EVIDENCE" here,
  the exact string "insufficient evidence" there.
- Temperature is passed per call (not fixed at construction like generate.py's `_llm`),
  because src/eval/uncertainty.py's self-consistency check needs several independent,
  temperature>0 samples of the same question.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

LLM_MODEL = "claude-haiku-4-5"

_llm = ChatAnthropic(model=LLM_MODEL)

# Five core SEP-1 ("Severe Sepsis/Septic Shock Early Management Bundle") elements.
SEP1_QUESTIONS = [
    {
        "id": "q1_antibiotics_3h",
        "text": "Were broad-spectrum antibiotics administered within 3 hours of sepsis presentation?",
    },
    {
        "id": "q2_blood_culture_before_antibiotics",
        "text": "Were blood cultures drawn before antibiotics were administered?",
    },
    {
        "id": "q3_lactate_6h",
        "text": "Was an initial lactate level measured within 6 hours of sepsis presentation?",
    },
    {
        "id": "q4_lactate_repeat",
        "text": "If the initial lactate was elevated (>2 mmol/L), was a repeat lactate measured?",
    },
    {
        "id": "q5_fluid_30ml_kg",
        "text": "Was a 30 mL/kg crystalloid fluid bolus administered for septic shock or sepsis-induced hypotension?",
    },
]

VALID_ANSWERS = {"Yes", "No", "INSUFFICIENT EVIDENCE"}

PROMPT_TEMPLATE = """You are a clinical evidence-review assistant checking SEP-1 sepsis \
bundle compliance from a clinical note excerpt. Answer the question using ONLY the \
context below. Do not use outside medical knowledge or assume anything not stated.

If the context does not document enough information to answer with confidence, you \
MUST answer "INSUFFICIENT EVIDENCE" instead of guessing.

Context:
{context}

Question: {query}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"answer": "Yes" | "No" | "INSUFFICIENT EVIDENCE", "justification": "<one short sentence>"}}"""


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


def generate_sepsis_answer(question: dict, context_texts: list[str], temperature: float = 0.0) -> dict:
    """Answer one SEP1_QUESTIONS entry from context_texts only.

    `temperature` is applied via .bind() rather than a second ChatAnthropic instance,
    so repeated calls (e.g. self-consistency sampling) reuse the same underlying client.
    """
    context = "\n\n".join(context_texts) if context_texts else "(no context retrieved)"
    prompt = PROMPT_TEMPLATE.format(context=context, query=question["text"])

    response = _llm.bind(temperature=temperature).invoke(prompt)
    return _parse_response(response.content)


if __name__ == "__main__":
    example_question = SEP1_QUESTIONS[0]
    example_context = [
        "Antibiotics (vancomycin, piperacillin-tazobactam) started 45 minutes after "
        "sepsis protocol activation."
    ]
    print(generate_sepsis_answer(example_question, example_context))
