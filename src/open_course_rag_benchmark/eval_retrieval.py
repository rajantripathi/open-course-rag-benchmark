from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl


def reciprocal_rank(gold: set[str], ranked: list[str], cutoff: int) -> float:
    for rank, chunk_id in enumerate(ranked[:cutoff], start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def dcg(gold: set[str], ranked: list[str], cutoff: int) -> float:
    score = 0.0
    for rank, chunk_id in enumerate(ranked[:cutoff], start=1):
        if chunk_id in gold:
            score += 1.0 / (1 if rank == 1 else __import__("math").log2(rank))
    return score


def ndcg(gold: set[str], ranked: list[str], cutoff: int) -> float:
    ideal = dcg(gold, list(gold), cutoff)
    if ideal == 0:
        return 0.0
    return dcg(gold, ranked, cutoff) / ideal


def evaluate(retrieval_rows: list[dict], gold_rows: list[dict]) -> dict:
    gold_by_qid = {row["qid"]: set(row["gold_chunk_ids"]) for row in gold_rows}
    metrics = defaultdict(list)
    for row in retrieval_rows:
        ranked = [item["chunk_id"] for item in row["ranked_results"]]
        gold = gold_by_qid[row["qid"]]
        for k in (1, 3, 5):
            metrics[f"Recall@{k}"].append(1.0 if any(chunk_id in gold for chunk_id in ranked[:k]) else 0.0)
        metrics["MRR@10"].append(reciprocal_rank(gold, ranked, 10))
        metrics["nDCG@10"].append(ndcg(gold, ranked, 10))
    return {name: sum(values) / len(values) for name, values in metrics.items()}


def slice_metrics(retrieval_rows: list[dict], gold_rows: list[dict], questions: list[dict], key: str) -> dict:
    question_by_qid = {row["qid"]: row for row in questions}
    grouped_retrieval: dict[str, list[dict]] = defaultdict(list)
    grouped_gold: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval_rows:
        grouped_retrieval[question_by_qid[row["qid"]][key]].append(row)
    for row in gold_rows:
        grouped_gold[question_by_qid[row["qid"]][key]].append(row)
    return {
        group: evaluate(grouped_retrieval[group], grouped_gold[group])
        for group in grouped_retrieval
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs against gold labels.")
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    retrieval_rows = read_jsonl(args.retrieval)
    gold_rows = read_jsonl(args.gold)
    summary = {"overall": evaluate(retrieval_rows, gold_rows)}
    if args.questions:
        questions = read_jsonl(args.questions)
        summary["by_language"] = slice_metrics(retrieval_rows, gold_rows, questions, "language")
        summary["by_course"] = slice_metrics(retrieval_rows, gold_rows, questions, "course_id")
        summary["by_question_type"] = slice_metrics(retrieval_rows, gold_rows, questions, "question_type")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote retrieval metrics to {args.output}")


if __name__ == "__main__":
    main()

