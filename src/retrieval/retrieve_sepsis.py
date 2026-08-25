"""Retrieve nearest SEP-1 note chunks for a query, scoped to one patient (Step 3).

Parallel to retrieve.py, but for the sepsis notes FAISS index instead of PubMedQA's,
and scoped to a single patient's chunks -- SEP-1 compliance questions are answered
per-patient, not across patients. Uses retrieve.py's search() (added alongside its
existing retrieve(), which this file does not modify) so the returned score is a
cosine similarity in [0, 1] rather than a raw L2 distance.
"""

import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent))
from retrieve import search  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "sepsis_chunks.jsonl"
FAISS_INDEX_PATH = BASE_DIR / "data" / "processed" / "sepsis_faiss_index" / "index.faiss"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Loaded once at import time, mirroring retrieve.py's pattern.
_index = faiss.read_index(str(FAISS_INDEX_PATH))
_model = SentenceTransformer(EMBEDDING_MODEL)

with CHUNKS_PATH.open(encoding="utf-8") as f:
    _chunks = [json.loads(line) for line in f]


def retrieve_sepsis(query: str, subject_id: int, hadm_id: int, top_k: int = 5) -> list[dict]:
    """Search within one patient's chunks and return text + cosine similarity score.

    This corpus is tiny (a handful of synthetic notes), so scoping is done by
    retrieving every candidate (k=_index.ntotal) and filtering by subject_id/
    hadm_id, rather than maintaining a separate index per patient. At real
    MIMIC-IV-Note scale, per-patient indexing or FAISS metadata pre-filtering
    would be worth switching to instead.
    """
    candidates = search(query, _index, _chunks, _model, top_k=_index.ntotal)
    matches = [c for c in candidates if c["subject_id"] == subject_id and c["hadm_id"] == hadm_id]
    return matches[:top_k]


if __name__ == "__main__":
    example = _chunks[0]
    results = retrieve_sepsis(
        "Were antibiotics given within 3 hours?",
        example["subject_id"],
        example["hadm_id"],
    )
    print(f"\nPatient {example['subject_id']}/{example['hadm_id']}:")
    for rank, r in enumerate(results, start=1):
        print(f"  [{rank}] score={r['score']:.4f} | {r['text'][:100]}...")
