from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .io_utils import read_csv


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must have the same length")
    total = len(labels_a)
    if total == 0:
        return 0.0
    agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / total) * (counts_b[label] / total) for label in set(counts_a) | set(counts_b))
    if expected == 1:
        return 1.0
    return (agreement - expected) / (1 - expected)


def summarize(rows: list[dict]) -> dict:
    label_counts = Counter(row["label"] for row in rows)
    output = {
        "total_rows": len(rows),
        "label_distribution": dict(label_counts),
    }
    duplicate_pairs = {}
    for row in rows:
        key = row.get("duplicate_group", "").strip()
        if not key:
            continue
        duplicate_pairs.setdefault(key, []).append(row["label"])
    duplicate_labels = [labels for labels in duplicate_pairs.values() if len(labels) == 2]
    if duplicate_labels:
        output["intra_annotator_kappa"] = cohens_kappa(
            [labels[0] for labels in duplicate_labels],
            [labels[1] for labels in duplicate_labels],
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize groundedness annotations and compute intra-annotator agreement.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = read_csv(args.annotations)
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote grounding metrics to {args.output}")


if __name__ == "__main__":
    main()

