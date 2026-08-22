"""Chunk PubMedQA contexts and embed them with sentence-transformers."""

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "raw" / "pubmedqa_labeled.jsonl"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"
FAISS_INDEX_PATH = BASE_DIR / "data" / "processed" / "faiss_index" / "index.faiss"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_samples(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_samples(samples):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for idx, sample in enumerate(samples):
        sample_id = sample.get("pubid", idx)
        question = sample["question"]
        answer = sample["final_decision"]
        contexts = sample["context"]["contexts"]

        for context_text in contexts:
            for chunk_text in splitter.split_text(context_text):
                chunks.append(
                    {
                        "text": chunk_text,
                        "sample_id": sample_id,
                        "question": question,
                        "answer": answer,
                    }
                )
    return chunks


def embed_chunks(chunks, model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def save_chunks(chunks, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def main():
    samples = load_samples(INPUT_PATH)
    chunks = chunk_samples(samples)
    print(f"Number of chunks: {len(chunks)}")

    embeddings = embed_chunks(chunks)
    print(f"Embedding shape: {embeddings.shape}")

    save_chunks(chunks, CHUNKS_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved chunks to {CHUNKS_PATH}")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    index = build_faiss_index(embeddings)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Total vectors in FAISS index: {index.ntotal}")
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
