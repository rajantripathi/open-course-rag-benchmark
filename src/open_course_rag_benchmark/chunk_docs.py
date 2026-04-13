from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from .io_utils import read_jsonl, read_yaml, write_jsonl
from .text import chunk_sentences


def section_for_document(document: dict) -> str:
    return document.get("title") or document.get("doc_id", "section")


def chunk_documents(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        text_chunks = chunk_sentences(document["text"], chunk_size, overlap)
        for chunk_index, chunk in enumerate(text_chunks):
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}_{chunk_index:03d}",
                    "doc_id": document["doc_id"],
                    "course_id": document["course_id"],
                    "section": section_for_document(document),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
    return chunks


def print_stats(chunks: list[dict]) -> None:
    lengths = [len(chunk["text"].split()) for chunk in chunks]
    by_course: dict[str, int] = {}
    for chunk in chunks:
        by_course[chunk["course_id"]] = by_course.get(chunk["course_id"], 0) + 1
    print(
        "total_chunks={} mean={:.2f} median={} min={} max={}".format(
            len(chunks),
            statistics.mean(lengths) if lengths else 0.0,
            statistics.median(lengths) if lengths else 0,
            min(lengths) if lengths else 0,
            max(lengths) if lengths else 0,
        )
    )
    for course_id in sorted(by_course):
        print(f"{course_id}: chunks={by_course[course_id]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk documents.jsonl into chunks.jsonl.")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = read_yaml(args.config) if args.config else {}
    chunk_size = args.chunk_size or cfg.get("chunk_size", 512)
    overlap = args.overlap or cfg.get("overlap", 64)
    documents = read_jsonl(args.documents)
    chunks = chunk_documents(documents, chunk_size, overlap)
    write_jsonl(args.output, chunks)
    print_stats(chunks)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()

