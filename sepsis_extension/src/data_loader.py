"""Load clinical notes and split them into overlapping chunks for retrieval."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CHUNK_OVERLAP, CHUNK_SIZE, NOTES_PATH  # noqa: E402


def load_notes(path: Path = NOTES_PATH):
    """Load notes as a list of {subject_id, hadm_id, text} dicts."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring to break on whitespace.

    Simple character-window splitter (no extra dependency) -- fine at note-length
    scale. If chunk_size grows much larger relative to typical note length, a
    sentence-aware splitter would preserve context boundaries better.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            break_point = text.rfind(" ", start, end)
            if break_point > start:
                end = break_point
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks


def chunk_notes(notes: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Chunk every note, tagging each chunk with its source patient/admission."""
    chunks = []
    for note in notes:
        for i, chunk_text_ in enumerate(chunk_text(note["text"], chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{note['subject_id']}_{note['hadm_id']}_{i}",
                    "subject_id": note["subject_id"],
                    "hadm_id": note["hadm_id"],
                    "text": chunk_text_,
                }
            )
    return chunks


if __name__ == "__main__":
    notes = load_notes()
    chunks = chunk_notes(notes)
    print(f"Loaded {len(notes)} notes -> {len(chunks)} chunks")
    for chunk in chunks[:3]:
        print(f"  [{chunk['chunk_id']}] {chunk['text'][:100]}...")
