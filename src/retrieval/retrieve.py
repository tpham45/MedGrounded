"""Retrieve nearest PubMedQA chunks for a query using the prebuilt FAISS index."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
FAISS_INDEX_PATH = BASE_DIR / "data" / "processed" / "faiss_index" / "index.faiss"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Loaded once at import time so retrieve() stays cheap to call repeatedly.
_index = faiss.read_index(str(FAISS_INDEX_PATH))
_model = SentenceTransformer(EMBEDDING_MODEL)

with CHUNKS_PATH.open(encoding="utf-8") as f:
    _chunks = [json.loads(line) for line in f]


def retrieve(query: str, top_k: int = 5):
    query_embedding = _model.encode([query], convert_to_numpy=True)
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        chunk = _chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "question": chunk["question"],
                "answer": chunk["answer"],
                "score": float(score),
            }
        )
    return results


def search(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """Generic cosine-similarity search over an arbitrary FAISS inner-product index.

    Unlike retrieve() above (PubMedQA-specific, fixed to the module-level
    _index/_chunks/_model, and returning a raw L2 distance), this takes its
    index/chunks/model as parameters and returns a similarity score normalized
    to [0, 1] -- a confidence signal other retrieval consumers (e.g. the sepsis
    pipeline's retrieve_sepsis.py) can plug directly into an uncertainty score.
    Callers must build `index` from L2-normalized embeddings with
    faiss.IndexFlatIP for the returned score to be a true cosine similarity.
    retrieve() and the module-level PubMedQA index above are untouched.
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    query_embedding = query_embedding / norm

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({**chunks[idx], "score": float(max(0.0, min(1.0, score)))})
    return results


if __name__ == "__main__":
    test_queries = [
        "What is the effect of mitochondria on plant cell death?",
        "Does exercise reduce risk of diabetes?",
        "Is aspirin effective for preventing heart attacks?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        for rank, result in enumerate(retrieve(query, top_k=5), start=1):
            snippet = result["text"][:150].replace("\n", " ")
            print(f"  [{rank}] score={result['score']:.4f} | question={result['question']}")
            print(f"       text={snippet}...")
