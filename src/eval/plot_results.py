"""Plot RAG vs no-RAG faithfulness averages as a paper-style bar chart."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "eval_results.jsonl"
OUTPUT_PATH = BASE_DIR / "docs" / "images" / "faithfulness_comparison.png"

RAG_COLOR = "#1f4e8c"
NO_RAG_COLOR = "#b0b0b0"


def load_averages(path: Path) -> tuple[float, float]:
    with path.open(encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    avg_rag = sum(r["faithfulness_score_rag"] for r in results) / len(results)
    avg_no_rag = sum(r["faithfulness_score_no_rag"] for r in results) / len(results)
    return avg_rag, avg_no_rag


def plot_comparison(avg_rag: float, avg_no_rag: float, path: Path):
    labels = ["RAG", "No-RAG"]
    values = [avg_rag, avg_no_rag]
    colors = [RAG_COLOR, NO_RAG_COLOR]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Faithfulness score", fontsize=11)
    ax.set_title("RAG vs No-RAG Faithfulness (RAGAS, n=50)", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    avg_rag, avg_no_rag = load_averages(INPUT_PATH)
    plot_comparison(avg_rag, avg_no_rag, OUTPUT_PATH)
    print(f"Average faithfulness RAG: {avg_rag:.4f}")
    print(f"Average faithfulness no-RAG: {avg_no_rag:.4f}")
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
