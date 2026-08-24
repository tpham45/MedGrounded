"""Embed chunks with sentence-transformers, index them with FAISS, and search.

Cosine similarity (via L2-normalized vectors + an inner-product index) is used
instead of raw L2 distance, so the similarity score search() returns is already
in [0, 1] and can feed directly into uncertainty.py as a confidence signal.
"""

import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    TOP_K,
)

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on an all-zero embedding
    return vectors / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return _normalize(embeddings.astype("float32"))


def build_index(chunks: list[dict]) -> faiss.Index:
    """Embed all chunks and build+persist a FAISS inner-product (cosine) index.

    This dataset is tiny (a handful of synthetic notes), so a single flat index
    over every chunk from every patient is built and search() filters by
    subject_id/hadm_id after retrieval. At real MIMIC-IV-Note scale (many
    patients, many chunks each), per-patient indexing or FAISS metadata
    pre-filtering would be worth switching to instead.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    return index


def search(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = TOP_K,
    subject_id=None,
    hadm_id=None,
) -> list[dict]:
    """Return the top_k nearest chunks, each with its text and similarity score.

    If subject_id/hadm_id are given, results are scoped to that single note's
    chunks (SEP-1 questions are answered per-patient, not across patients).
    """
    query_embedding = embed_texts([query])

    # Scoped search: retrieve every candidate, then filter down to this note.
    # Fine at this corpus size; see build_index()'s docstring for scaling notes.
    k = index.ntotal if (subject_id is not None or hadm_id is not None) else top_k
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if subject_id is not None and chunk["subject_id"] != subject_id:
            continue
        if hadm_id is not None and chunk["hadm_id"] != hadm_id:
            continue
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "subject_id": chunk["subject_id"],
                "hadm_id": chunk["hadm_id"],
                "text": chunk["text"],
                # Cosine similarity of normalized vectors is in [-1, 1]; clip to
                # [0, 1] so this can be used directly as a confidence signal.
                "score": float(max(0.0, min(1.0, score))),
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    from data_loader import chunk_notes, load_notes

    notes = load_notes()
    chunks = chunk_notes(notes)
    index = build_index(chunks)
    print(f"Indexed {index.ntotal} chunks from {len(notes)} notes")

    example = notes[0]
    results = search(
        "Were antibiotics given within 3 hours?",
        index,
        chunks,
        subject_id=example["subject_id"],
        hadm_id=example["hadm_id"],
    )
    for rank, r in enumerate(results, start=1):
        print(f"  [{rank}] score={r['score']:.4f} | {r['text'][:100]}...")
