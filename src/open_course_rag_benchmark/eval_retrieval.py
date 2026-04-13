from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from .io_utils import read_jsonl, write_csv


def reciprocal_rank(gold: set[str], ranked: list[str], cutoff: int) -> float:
    for rank, chunk_id in enumerate(ranked[:cutoff], start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg(gold: set[str], ranked: list[str], cutoff: int) -> float:
    import math

    dcg = 0.0
    for rank, chunk_id in enumerate(ranked[:cutoff], start=1):
        if chunk_id in gold:
            dcg += 1.0 / (1.0 if rank == 1 else math.log2(rank))
    ideal_hits = min(len(gold), cutoff)
    ideal = sum(1.0 / (1.0 if rank == 1 else math.log2(rank)) for rank in range(1, ideal_hits + 1))
    return 0.0 if ideal == 0 else dcg / ideal


def score_rows(retrieval_rows: list[dict], gold_rows: list[dict], questions: list[dict]) -> list[dict]:
    gold_by_qid = {row["qid"]: set(row["gold_chunk_ids"]) for row in gold_rows}
    question_by_qid = {row["qid"]: row for row in questions}
    scored: list[dict] = []
    for row in retrieval_rows:
        ranked = [item["chunk_id"] for item in row["ranked_chunks"]]
        gold = gold_by_qid[row["qid"]]
        question = question_by_qid[row["qid"]]
        scored.append(
            {
                "qid": row["qid"],
                "system": row["system"],
                "language": row["language"],
                "course_id": question["course_id"],
                "question_type": question["question_type"],
                "Recall@1": float(any(chunk in gold for chunk in ranked[:1])),
                "Recall@3": float(any(chunk in gold for chunk in ranked[:3])),
                "Recall@5": float(any(chunk in gold for chunk in ranked[:5])),
                "MRR@10": reciprocal_rank(gold, ranked, 10),
                "nDCG@10": ndcg(gold, ranked, 10),
            }
        )
    return scored


def aggregate(rows: list[dict], group_cols: list[str]) -> list[dict]:
    df = pd.DataFrame(rows)
    metrics = ["Recall@1", "Recall@3", "Recall@5", "MRR@10", "nDCG@10"]
    grouped = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped.to_dict(orient="records")


def significance(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    systems = sorted(df["system"].unique())
    output: list[dict] = []
    for i, left in enumerate(systems):
        for right in systems[i + 1 :]:
            left_scores = df[df["system"] == left].sort_values(["qid", "language"])["Recall@5"].tolist()
            right_scores = df[df["system"] == right].sort_values(["qid", "language"])["Recall@5"].tolist()
            if left_scores and right_scores and len(left_scores) == len(right_scores):
                stat, pvalue = wilcoxon(left_scores, right_scores, zero_method="zsplit")
                output.append({"left": left, "right": right, "metric": "Recall@5", "p_value": float(pvalue)})
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs and export grouped tables.")
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    gold_rows = read_jsonl(args.gold)
    questions = read_jsonl(args.questions)
    all_rows: list[dict] = []
    for path in sorted(args.retrieval_dir.glob("*_results.jsonl")):
        all_rows.extend(score_rows(read_jsonl(path), gold_rows, questions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "retrieval_overall.csv", aggregate(all_rows, ["system"]))
    write_csv(args.output_dir / "retrieval_by_course.csv", aggregate(all_rows, ["system", "course_id"]))
    write_csv(args.output_dir / "retrieval_by_language.csv", aggregate(all_rows, ["system", "language"]))
    write_csv(args.output_dir / "retrieval_by_question_type.csv", aggregate(all_rows, ["system", "question_type"]))
    write_csv(args.output_dir / "retrieval_significance.csv", significance(all_rows))
    print(f"Wrote retrieval tables to {args.output_dir}")

