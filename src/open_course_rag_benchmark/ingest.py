from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .io_utils import write_jsonl
from .text import normalize_whitespace


COURSE_META = {
    "openstax_data_science": {
        "license": "CC BY-NC-SA 4.0",
        "book_slug": "principles-data-science",
        "doc_prefix": "ds",
    },
    "openstax_philosophy": {
        "license": "CC BY 4.0",
        "book_slug": "introduction-philosophy",
        "doc_prefix": "phil",
    },
}


def load_toc(raw_dir: Path) -> dict[str, dict]:
    toc_path = raw_dir / "toc.json"
    entries = json.loads(toc_path.read_text(encoding="utf-8"))
    return {entry["section_slug"]: entry for entry in entries}


def section_identifiers(section_slug: str, prefix: str) -> tuple[str, str]:
    parts = section_slug.split("-")
    numeric = [part for part in parts if part.isdigit()]
    if len(numeric) >= 2:
        chapter, section = numeric[0], numeric[1]
        doc_id = f"{prefix}_ch{int(chapter):02d}_s{int(section):02d}"
    elif len(numeric) == 1:
        doc_id = f"{prefix}_ch{int(numeric[0]):02d}_s00"
    else:
        doc_id = f"{prefix}_{section_slug.replace('-', '_')}"
    return doc_id, section_slug


def build_documents(raw_root: Path) -> list[dict]:
    today = str(date.today())
    documents: list[dict] = []
    for course_id, meta in COURSE_META.items():
        course_dir = raw_root / course_id
        toc = load_toc(course_dir)
        for section_file in sorted(course_dir.glob("*.md")):
            if section_file.name == "toc.json":
                continue
            section_slug = section_file.stem
            toc_entry = toc.get(section_slug, {})
            doc_id, _ = section_identifiers(section_slug, meta["doc_prefix"])
            title = toc_entry.get("title", section_slug.replace("-", " ").title())
            source_url = toc_entry.get("url", f"https://openstax.org/books/{meta['book_slug']}/pages/{section_slug}")
            text = normalize_whitespace(section_file.read_text(encoding="utf-8"))
            documents.append(
                {
                    "doc_id": doc_id,
                    "course_id": course_id,
                    "title": title,
                    "source_type": "textbook_section",
                    "license": meta["license"],
                    "source_url": source_url,
                    "download_date": today,
                    "text": text,
                }
            )
    return documents


def print_stats(documents: list[dict]) -> None:
    counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    for document in documents:
        course_id = document["course_id"]
        counts[course_id] = counts.get(course_id, 0) + 1
        token_counts[course_id] = token_counts.get(course_id, 0) + len(document["text"].split())
    total_tokens = sum(token_counts.values())
    print(f"total_documents={len(documents)} total_tokens={total_tokens}")
    for course_id in sorted(counts):
        print(f"{course_id}: documents={counts[course_id]} tokens={token_counts[course_id]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build documents.jsonl from scraped OpenStax sections.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    documents = build_documents(args.raw_root)
    write_jsonl(args.output, documents)
    print_stats(documents)
    print(f"Wrote {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()

