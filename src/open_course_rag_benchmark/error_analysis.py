from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .io_utils import read_jsonl, write_csv


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def overlap_ratio(question_text: str, other_text: str) -> float:
    q_tokens = tokenize(question_text)
    o_tokens = tokenize(other_text)
    if not q_tokens:
        return 0.0
    return len(q_tokens & o_tokens) / len(q_tokens)


def chapter_prefix(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else chunk_id


def categorize_failure(question: dict, gold_chunk: dict, retrieved_chunks: list[dict]) -> tuple[str, bool]:
    correct_course_in_top5 = any(chunk["course_id"] == question["course_id"] for chunk in retrieved_chunks)
    if any(chunk["doc_id"] == gold_chunk["doc_id"] for chunk in retrieved_chunks):
        return "chunk_boundary", True
    if any(chapter_prefix(chunk["chunk_id"]) == chapter_prefix(gold_chunk["chunk_id"]) for chunk in retrieved_chunks):
        return "chunk_boundary", True
    if not correct_course_in_top5:
        return "domain_confusion", False
    if question["language"] == "uz":
        return "cross_lingual", correct_course_in_top5
    gold_overlap = overlap_ratio(question["question_text"], gold_chunk["text"])
    best_retrieved_overlap = max((overlap_ratio(question["question_text"], chunk["text"]) for chunk in retrieved_chunks), default=0.0)
    if best_retrieved_overlap > gold_overlap:
        return "term_mismatch", correct_course_in_top5
    return "other", correct_course_in_top5


def build_rows(hybrid_results: list[dict], questions: list[dict], gold_rows: list[dict], chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    question_by_qid = {row["qid"]: row for row in questions}
    gold_by_qid = {row["qid"]: row for row in gold_rows}
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    failures: list[dict] = []
    for row in hybrid_results:
        ranked_ids = [item["chunk_id"] for item in row["ranked_chunks"][:5]]
        gold = gold_by_qid[row["qid"]]
        gold_set = set(gold["gold_chunk_ids"])
        if gold_set.intersection(ranked_ids):
            continue
        question = question_by_qid[row["qid"]]
        gold_chunk = chunk_by_id[gold["gold_chunk_ids"][0]]
        retrieved_chunks = [chunk_by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in chunk_by_id]
        category, correct_chapter_hit = categorize_failure(question, gold_chunk, retrieved_chunks)
        failures.append(
            {
                "qid": row["qid"],
                "language": row["language"],
                "course_id": question["course_id"],
                "question_type": question["question_type"],
                "category": category,
                "correct_chapter_in_top5": str(correct_chapter_hit).lower(),
                "question_text": question["question_text"],
                "gold_chunk_id": gold_chunk["chunk_id"],
                "gold_chunk_text": gold_chunk["text"],
                "retrieved_chunk_ids": " ".join(ranked_ids),
                "retrieved_chunk_texts": json.dumps(
                    [{"chunk_id": chunk["chunk_id"], "text": chunk["text"]} for chunk in retrieved_chunks],
                    ensure_ascii=False,
                ),
                "question_gold_overlap": overlap_ratio(question["question_text"], gold_chunk["text"]),
                "best_retrieved_overlap": max(
                    (overlap_ratio(question["question_text"], chunk["text"]) for chunk in retrieved_chunks),
                    default=0.0,
                ),
            }
        )
    counter = Counter((row["category"], row["language"], row["course_id"]) for row in failures)
    aggregate_rows = [
        {"category": category, "language": language, "course_id": course_id, "count": count}
        for (category, language, course_id), count in sorted(counter.items())
    ]
    return aggregate_rows, failures


def sample_failures(failures: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for language, n in (("en", 10), ("uz", 10)):
        subset = [row for row in failures if row["language"] == language]
        subset = sorted(subset, key=lambda row: (row["category"], row["qid"]))
        samples.extend(subset[:n])
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval failure analysis for hybrid misses.")
    parser.add_argument("--hybrid-results", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-categories", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    aggregate_rows, failures = build_rows(
        read_jsonl(args.hybrid_results),
        read_jsonl(args.questions),
        read_jsonl(args.gold),
        read_jsonl(args.chunks),
    )
    write_csv(args.output_categories, aggregate_rows)
    write_csv(args.output_samples, sample_failures(failures))
    print(f"Wrote error categories to {args.output_categories}")
    print(f"Wrote error samples to {args.output_samples}")


if __name__ == "__main__":
    main()
