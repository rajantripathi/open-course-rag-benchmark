from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .io_utils import read_jsonl, read_yaml, write_jsonl


def run_dense(
    chunks: list[dict],
    questions: list[dict],
    model_name: str,
    top_k: int,
    query_prefix: str,
    passage_prefix: str,
    batch_size: int,
    index_output: Path | None = None,
) -> list[dict]:
    model = SentenceTransformer(model_name)
    chunk_embeddings = model.encode(
        [passage_prefix + chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    if index_output:
        index_output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_output))
    query_embeddings = model.encode(
        [query_prefix + question["question_text"] for question in questions],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    scores, indices = index.search(query_embeddings, top_k)
    results: list[dict] = []
    for question, row_scores, row_indices in zip(questions, scores, indices):
        results.append(
            {
                "qid": question["qid"],
                "system": "dense",
                "language": question["language"],
                "ranked_chunks": [
                    {"chunk_id": chunks[index]["chunk_id"], "rank": rank, "score": float(score)}
                    for rank, (index, score) in enumerate(zip(row_indices, row_scores), start=1)
                ],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dense retrieval with a multilingual embedding model.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = read_yaml(args.config) if args.config else {}
    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    results = run_dense(
        chunks=chunks,
        questions=questions,
        model_name=cfg.get("model", "intfloat/multilingual-e5-large"),
        top_k=cfg.get("top_k", 10),
        query_prefix=cfg.get("query_prefix", "query: "),
        passage_prefix=cfg.get("passage_prefix", "passage: "),
        batch_size=cfg.get("batch_size", 64),
        index_output=args.index_output,
    )
    write_jsonl(args.output, results)
    print(f"Wrote dense results to {args.output}")


if __name__ == "__main__":
    main()

