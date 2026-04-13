from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_jsonl, read_yaml, write_jsonl


def as_rank_map(row: dict) -> dict[str, int]:
    return {item["chunk_id"]: item["rank"] for item in row["ranked_chunks"]}


def rrf(rank_maps: list[dict[str, int]], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_map in rank_maps:
        for chunk_id, rank in rank_map.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def run_hybrid(bm25_results: list[dict], dense_results: list[dict], top_k: int, k: int) -> list[dict]:
    dense_by_qid = {(row["qid"], row["language"]): row for row in dense_results}
    output: list[dict] = []
    for bm25_row in bm25_results:
        key = (bm25_row["qid"], bm25_row["language"])
        dense_row = dense_by_qid[key]
        fused = rrf([as_rank_map(bm25_row), as_rank_map(dense_row)], k)[:top_k]
        output.append(
            {
                "qid": bm25_row["qid"],
                "system": "hybrid",
                "language": bm25_row["language"],
                "ranked_chunks": [
                    {"chunk_id": chunk_id, "rank": rank, "score": float(score)}
                    for rank, (chunk_id, score) in enumerate(fused, start=1)
                ],
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse BM25 and dense retrieval runs with RRF.")
    parser.add_argument("--bm25-results", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = read_yaml(args.config) if args.config else {}
    results = run_hybrid(
        read_jsonl(args.bm25_results),
        read_jsonl(args.dense_results),
        cfg.get("top_k", 10),
        cfg.get("k", 60),
    )
    write_jsonl(args.output, results)
    print(f"Wrote hybrid results to {args.output}")


if __name__ == "__main__":
    main()

