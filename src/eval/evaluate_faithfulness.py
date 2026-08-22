"""Evaluate RAG vs no-RAG faithfulness on PubMedQA questions using RAGAS, concurrently."""

import asyncio
import json
import os
import random
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

# ragas==0.4.3 unconditionally imports langchain_community.chat_models.vertexai,
# a module that no longer exists in the installed langchain-community version.
# We stub it out since VertexAI is unused here; this only affects our process.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAI:
    pass


_vertexai_stub.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

from anthropic import AsyncAnthropic  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import Faithfulness  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation.generate import PROMPT_TEMPLATE, _llm  # noqa: E402
from retrieval.retrieve import retrieve  # noqa: E402

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "raw" / "pubmedqa_labeled.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "eval_results.jsonl"

JUDGE_MODEL = "claude-haiku-4-5"
NUM_SAMPLES = 50
RANDOM_SEED = 42
MAX_CONCURRENCY = 5

NO_RAG_PROMPT_TEMPLATE = """You are a biomedical assistant. Answer the question using your own knowledge.

Question: {query}

Answer:"""


def load_questions(path: Path, n: int, seed: int):
    with path.open(encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    random.Random(seed).shuffle(samples)
    return [sample["question"] for sample in samples[:n]]


def build_faithfulness_metric() -> Faithfulness:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    llm = llm_factory(JUDGE_MODEL, provider="anthropic", client=client, max_tokens=4096)
    # This model rejects requests with both temperature and top_p set; drop the default top_p.
    del llm.model_args["top_p"]
    return Faithfulness(llm=llm)


async def generate_rag_answer(query: str, context_texts: list[str]) -> str:
    context = "\n\n".join(context_texts)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    response = await _llm.ainvoke(prompt)
    return response.content


async def generate_no_rag_answer(query: str) -> str:
    response = await _llm.ainvoke(NO_RAG_PROMPT_TEMPLATE.format(query=query))
    return response.content


async def process_question(
    question: str, context_texts: list[str], faithfulness: Faithfulness, semaphore: asyncio.Semaphore
):
    async with semaphore:
        # Both calls use the async client (ainvoke) exclusively; never mix with the sync
        # invoke() on the same ChatAnthropic instance, which corrupts its shared connection pool.
        rag_answer, no_rag_answer = await asyncio.gather(
            generate_rag_answer(question, context_texts),
            generate_no_rag_answer(question),
        )

        # Score the no-RAG answer against the SAME retrieved context to see how well
        # an ungrounded answer happens to align with the evidence a RAG system would use.
        rag_score, no_rag_score = await asyncio.gather(
            faithfulness.ascore(
                user_input=question, response=rag_answer, retrieved_contexts=context_texts
            ),
            faithfulness.ascore(
                user_input=question, response=no_rag_answer, retrieved_contexts=context_texts
            ),
        )

        return {
            "question": question,
            "generated_answer_rag": rag_answer,
            "generated_answer_no_rag": no_rag_answer,
            "faithfulness_score_rag": float(rag_score.value),
            "faithfulness_score_no_rag": float(no_rag_score.value),
        }


def save_results(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def main():
    questions = load_questions(INPUT_PATH, NUM_SAMPLES, RANDOM_SEED)
    faithfulness = build_faithfulness_metric()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Retrieval (embedding + FAISS) is CPU-bound and not thread-safe to call concurrently,
    # so resolve all contexts sequentially up front; only the LLM calls run concurrently below.
    contexts = [[chunk["text"] for chunk in retrieve(q)] for q in questions]

    results = await asyncio.gather(
        *(
            process_question(q, ctx, faithfulness, semaphore)
            for q, ctx in zip(questions, contexts)
        )
    )

    save_results(results, OUTPUT_PATH)

    avg_rag = sum(r["faithfulness_score_rag"] for r in results) / len(results)
    avg_no_rag = sum(r["faithfulness_score_no_rag"] for r in results) / len(results)

    print(f"Average faithfulness RAG ({len(results)} questions): {avg_rag:.4f}")
    print(f"Average faithfulness no-RAG ({len(results)} questions): {avg_no_rag:.4f}")
    print(f"Difference (RAG - no-RAG): {avg_rag - avg_no_rag:+.4f}")
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
