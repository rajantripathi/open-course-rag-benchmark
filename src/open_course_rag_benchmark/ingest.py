from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_csv, write_jsonl
from .text import normalize_whitespace, strip_html


def read_source_text(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        return strip_html(raw_text)
    return normalize_whitespace(raw_text)


def discover_documents(input_dir: Path) -> list[dict]:
    documents: list[dict] = []
    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
            continue
        doc_id = file_path.stem
        documents.append(
            {
                "doc_id": doc_id,
                "course_id": "unknown_course",
                "title": file_path.stem.replace("_", " ").title(),
                "source_type": "unknown",
                "license": "unknown",
                "source_url": "",
                "download_date": "",
                "text": read_source_text(file_path),
            }
        )
    return documents


def build_from_manifest(manifest_path: Path, base_dir: Path) -> list[dict]:
    rows = read_csv(manifest_path)
    documents: list[dict] = []
    for row in rows:
        file_path = base_dir / row["file_path"]
        documents.append(
            {
                "doc_id": row["doc_id"],
                "course_id": row["course_id"],
                "title": row["title"],
                "source_type": row["source_type"],
                "license": row["license"],
                "source_url": row["source_url"],
                "download_date": row["download_date"],
                "text": read_source_text(file_path),
            }
        )
    return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build documents.jsonl from raw open course materials.")
    parser.add_argument("--input-dir", type=Path, help="Directory of raw txt/md/html files.")
    parser.add_argument("--manifest", type=Path, help="CSV manifest describing raw files.")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Base directory for manifest file paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.manifest:
        documents = build_from_manifest(args.manifest, args.base_dir)
    elif args.input_dir:
        documents = discover_documents(args.input_dir)
    else:
        parser.error("either --manifest or --input-dir is required")
    write_jsonl(args.output, documents)
    print(f"Wrote {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()

