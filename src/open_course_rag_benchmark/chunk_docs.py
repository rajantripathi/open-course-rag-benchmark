from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .text import chunk_words, tokenize


def section_for_document(document: dict) -> str:
    return document.get("title") or document.get("doc_id", "section")


def chunk_documents(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        words = tokenize(document["text"])
        for chunk_index, chunk in enumerate(chunk_words(words, chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}_{chunk_index:03d}",
                    "doc_id": document["doc_id"],
                    "course_id": document["course_id"],
                    "section": section_for_document(document),
                    "chunk_index": chunk_index,
                    "text": " ".join(chunk),
                }
            )
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk documents.jsonl into chunks.jsonl.")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    documents = read_jsonl(args.documents)
    chunks = chunk_documents(documents, args.chunk_size, args.overlap)
    write_jsonl(args.output, chunks)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()

