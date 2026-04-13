from __future__ import annotations

import argparse
from pathlib import Path

from rank_bm25 import BM25Okapi

from .io_utils import read_jsonl, read_yaml, write_jsonl
from .text import tokenize


def run_bm25(chunks: list[dict], questions: list[dict], top_k: int, k1: float, b: float) -> list[dict]:
    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    token_sets = [set(tokens) for tokens in corpus_tokens]
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    results: list[dict] = []
    for question in questions:
        tokens = tokenize(question["question_text"])
        scores = bm25.get_scores(tokens)
        if max(scores, default=0.0) == 0.0:
            query_terms = set(tokens)
            scores = [float(len(query_terms & token_set)) for token_set in token_sets]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results.append(
            {
                "qid": question["qid"],
                "system": "bm25",
                "language": question["language"],
                "ranked_chunks": [
                    {
                        "chunk_id": chunks[index]["chunk_id"],
                        "rank": rank,
                        "score": float(score),
                    }
                    for rank, (index, score) in enumerate(ranked, start=1)
                ],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = read_yaml(args.config) if args.config else {}
    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    results = run_bm25(chunks, questions, cfg.get("top_k", 10), cfg.get("k1", 1.5), cfg.get("b", 0.75))
    write_jsonl(args.output, results)
    print(f"Wrote BM25 results to {args.output}")


if __name__ == "__main__":
    main()

