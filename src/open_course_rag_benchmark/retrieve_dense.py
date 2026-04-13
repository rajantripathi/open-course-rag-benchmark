from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .io_utils import read_jsonl, write_jsonl


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def run_dense(
    chunks: list[dict],
    questions: list[dict],
    model_name: str,
    top_k: int,
    query_prefix: str,
    document_prefix: str,
) -> list[dict]:
    model = SentenceTransformer(model_name)
    chunk_embeddings = model.encode(
        [document_prefix + chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embeddings = model.encode(
        [query_prefix + question["question_text"] for question in questions],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores = query_embeddings @ chunk_embeddings.T
    results: list[dict] = []
    for question, row in zip(questions, scores):
        top_indices = np.argsort(row)[::-1][:top_k]
        results.append(
            {
                "qid": question["qid"],
                "system": "dense",
                "ranked_results": [
                    {"chunk_id": chunks[index]["chunk_id"], "score": float(row[index])}
                    for index in top_indices
                ],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dense retrieval with a multilingual embedding model.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-large")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--document-prefix", default="passage: ")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    results = run_dense(
        chunks=chunks,
        questions=questions,
        model_name=args.model_name,
        top_k=args.top_k,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    write_jsonl(args.output, results)
    print(f"Wrote dense results to {args.output}")


if __name__ == "__main__":
    main()

