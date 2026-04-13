from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .io_utils import read_csv


def plot_recall(metrics: dict, output_path: Path) -> None:
    recall_keys = ["Recall@1", "Recall@3", "Recall@5"]
    values = [metrics.get(key, 0.0) for key in recall_keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(recall_keys, values)
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval Recall")
    fig.tight_layout()
    fig.savefig(output_path)


def plot_grounding(rows: list[dict], output_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title("Groundedness Label Distribution")
    fig.tight_layout()
    fig.savefig(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simple paper-ready plots.")
    parser.add_argument("--retrieval-metrics", type=Path)
    parser.add_argument("--grounding-annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.retrieval_metrics:
        metrics = json.loads(args.retrieval_metrics.read_text(encoding="utf-8"))["overall"]
        plot_recall(metrics, args.output_dir / "retrieval_recall.png")
    if args.grounding_annotations:
        rows = read_csv(args.grounding_annotations)
        plot_grounding(rows, args.output_dir / "grounding_distribution.png")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()

