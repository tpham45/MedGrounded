"""Load PubMedQA (pqa_labeled) from HuggingFace and cache it as JSONL."""

import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "pubmedqa_labeled.jsonl"


def load_pubmedqa_labeled():
    return load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")


def save_as_jsonl(dataset, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    dataset = load_pubmedqa_labeled()

    print(f"Number of samples: {len(dataset)}")

    example = dataset[0]
    print("\nSample example:")
    print("Question:", example["question"])
    print("Context:", example["context"])
    print("Answer:", example["final_decision"])

    save_as_jsonl(dataset, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
