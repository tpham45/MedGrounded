"""Generate answers grounded in retrieved PubMedQA context via Claude."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

sys.path.append(str(Path(__file__).resolve().parents[1]))
from retrieval.retrieve import retrieve  # noqa: E402

load_dotenv()

LLM_MODEL = "claude-haiku-4-5"

_llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

PROMPT_TEMPLATE = """You are a biomedical assistant. Answer the question using ONLY the context below.
Do not use outside knowledge. If the context does not contain enough information to answer,
respond exactly with "insufficient evidence" instead of guessing.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, top_k: int = 5):
    retrieved_chunks = retrieve(query, top_k=top_k)

    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    response = _llm.invoke(prompt)

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "generated_answer": response.content,
    }


if __name__ == "__main__":
    test_queries = [
        "What is the effect of mitochondria on plant cell death?",
        "Does exercise reduce risk of diabetes?",
    ]

    for query in test_queries:
        result = generate_answer(query)
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['generated_answer']}")
