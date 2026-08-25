"""Chunk synthetic SEP-1 clinical notes and embed them (Step 2 of the sepsis pipeline).

Parallel to chunk_and_embed.py, but for the sepsis notes instead of PubMedQA, and
building a *separate* FAISS index so PubMedQA's index/chunks/embeddings are never
touched. Reuses embed_chunks() from chunk_and_embed.py rather than duplicating it;
that file's build_faiss_index()/main() (PubMedQA-specific, IndexFlatL2) are untouched.

Unlike chunk_and_embed.py's IndexFlatL2 (raw L2 distance), this builds an
IndexFlatIP over L2-normalized embeddings, so retrieve_sepsis.py's search returns
a cosine similarity in [0, 1] -- the retrieval confidence signal
src/eval/uncertainty.py needs.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parent))
from chunk_and_embed import embed_chunks  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "raw" / "sepsis_notes_synthetic.jsonl"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "sepsis_chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "sepsis_embeddings.npy"
FAISS_INDEX_PATH = BASE_DIR / "data" / "processed" / "sepsis_faiss_index" / "index.faiss"


def load_notes(path: Path = INPUT_PATH):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_notes(notes):
    # Same 500/50 char chunk size as chunk_and_embed.py, for consistency; notes are
    # short enough that most yield 1-2 chunks each.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for note in notes:
        for i, chunk_text in enumerate(splitter.split_text(note["text"])):
            chunks.append(
                {
                    "chunk_id": f"{note['subject_id']}_{note['hadm_id']}_{i}",
                    "subject_id": note["subject_id"],
                    "hadm_id": note["hadm_id"],
                    "text": chunk_text,
                }
            )
    return chunks


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on an all-zero embedding
    return vectors / norms


def save_chunks(chunks, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def build_cosine_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def main():
    notes = load_notes()
    chunks = chunk_notes(notes)
    print(f"Number of sepsis chunks: {len(chunks)}")

    embeddings = normalize(embed_chunks(chunks).astype("float32"))
    print(f"Embedding shape: {embeddings.shape}")

    save_chunks(chunks, CHUNKS_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved chunks to {CHUNKS_PATH}")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    index = build_cosine_index(embeddings)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Total vectors in FAISS index: {index.ntotal}")
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
