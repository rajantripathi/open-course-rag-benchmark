from __future__ import annotations

import argparse
from pathlib import Path

from rank_bm25 import BM25Okapi

from .io_utils import read_jsonl, write_jsonl
from .text import tokenize


def run_bm25(chunks: list[dict], questions: list[dict], top_k: int) -> list[dict]:
    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    token_sets = [set(tokens) for tokens in corpus_tokens]
    bm25 = BM25Okapi(corpus_tokens)
    results: list[dict] = []
    for question in questions:
        tokens = tokenize(question["question_text"])
        scores = bm25.get_scores(tokens)
        if max(scores, default=0.0) == 0.0:
            query_terms = set(tokens)
            scores = [float(len(query_terms & token_set)) for token_set in token_sets]
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        results.append(
            {
                "qid": question["qid"],
                "system": "bm25",
                "ranked_results": [
                    {"chunk_id": chunk["chunk_id"], "score": float(score)}
                    for chunk, score in ranked
                ],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    results = run_bm25(chunks, questions, args.top_k)
    write_jsonl(args.output, results)
    print(f"Wrote BM25 results to {args.output}")


if __name__ == "__main__":
    main()
