from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from .eval_retrieval import ndcg, reciprocal_rank
from .io_utils import read_jsonl, write_csv


METRICS = ("Recall@5", "MRR@10", "nDCG@10")


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
                "Recall@5": float(any(chunk in gold for chunk in ranked[:5])),
                "MRR@10": reciprocal_rank(gold, ranked, 10),
                "nDCG@10": ndcg(gold, ranked, 10),
            }
        )
    return scored


def cohens_d_paired(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    diffs = [a - b for a, b in zip(left, right)]
    mean_diff = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        return 0.0
    variance = sum((x - mean_diff) ** 2 for x in diffs) / (len(diffs) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return mean_diff / std


def paired_wilcoxon(left: list[float], right: list[float]) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        return 0.0, 1.0
    if all(a == b for a, b in zip(left, right)):
        return 0.0, 1.0
    stat, p_value = wilcoxon(left, right, zero_method="zsplit")
    return float(stat), float(p_value)


def build_significance_rows(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    output: list[dict] = []
    for language in sorted(df["language"].unique()):
        lang_df = df[df["language"] == language]
        systems = sorted(lang_df["system"].unique())
        for metric in METRICS:
            pivot = lang_df.pivot_table(index="qid", columns="system", values=metric)
            for system_a, system_b in combinations(systems, 2):
                subset = pivot[[system_a, system_b]].dropna()
                left = subset[system_a].tolist()
                right = subset[system_b].tolist()
                stat, p_value = paired_wilcoxon(left, right)
                output.append(
                    {
                        "metric": metric,
                        "language": language,
                        "system_a": system_a,
                        "system_b": system_b,
                        "wilcoxon_statistic": stat,
                        "p_value": p_value,
                        "cohens_d": cohens_d_paired(left, right),
                        "n": len(left),
                    }
                )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired retrieval significance tests.")
    parser.add_argument("--bm25-results", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--hybrid-results", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    gold_rows = read_jsonl(args.gold)
    questions = read_jsonl(args.questions)
    scored_rows: list[dict] = []
    for path in (args.bm25_results, args.dense_results, args.hybrid_results):
        scored_rows.extend(score_rows(read_jsonl(path), gold_rows, questions))
    significance_rows = build_significance_rows(scored_rows)
    write_csv(args.output_csv, significance_rows)
    print(f"Wrote significance table to {args.output_csv}")
    for row in significance_rows:
        print(
            f"{row['metric']} {row['language']} {row['system_a']} vs {row['system_b']}: "
            f"p={row['p_value']:.4g}, d={row['cohens_d']:.4f}, n={row['n']}"
        )


if __name__ == "__main__":
    main()
