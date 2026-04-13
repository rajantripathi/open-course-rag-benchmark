from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl


def as_rank_map(result: dict) -> dict[str, int]:
    return {
        item["chunk_id"]: rank
        for rank, item in enumerate(result["ranked_results"], start=1)
    }


def reciprocal_rank_fusion(*rank_maps: dict[str, int], rrf_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_map in rank_maps:
        for chunk_id, rank in rank_map.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def run_hybrid(bm25_results: list[dict], dense_results: list[dict], top_k: int, rrf_k: int) -> list[dict]:
    dense_by_qid = {row["qid"]: row for row in dense_results}
    output: list[dict] = []
    for bm25_row in bm25_results:
        dense_row = dense_by_qid[bm25_row["qid"]]
        fused = reciprocal_rank_fusion(
            as_rank_map(bm25_row),
            as_rank_map(dense_row),
            rrf_k=rrf_k,
        )[:top_k]
        output.append(
            {
                "qid": bm25_row["qid"],
                "system": "hybrid",
                "ranked_results": [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in fused
                ],
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse BM25 and dense retrieval results with reciprocal rank fusion.")
    parser.add_argument("--bm25-results", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    bm25_results = read_jsonl(args.bm25_results)
    dense_results = read_jsonl(args.dense_results)
    output = run_hybrid(bm25_results, dense_results, args.top_k, args.rrf_k)
    write_jsonl(args.output, output)
    print(f"Wrote hybrid results to {args.output}")


if __name__ == "__main__":
    main()

