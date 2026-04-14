from __future__ import annotations

import argparse
from pathlib import Path

from open_course_rag_benchmark.io_utils import read_jsonl, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export candidate questions to a curation CSV.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.candidates)
    translations = {row["candidate_id"]: row for row in read_jsonl(args.translations)}
    export_rows = [
        {
            "candidate_id": row["candidate_id"],
            "course_id": row["course_id"],
            "chunk_id": row["chunk_id"],
            "question_type": row["question_type"],
            "question_text_en": row["question_text"],
            "reference_answer_en": row["reference_answer"],
            "question_text_uz": translations.get(row["candidate_id"], {}).get("question_text", ""),
            "reference_answer_uz": translations.get(row["candidate_id"], {}).get("reference_answer", ""),
            "selected": "",
            "gold_chunk_ids": "",
            "notes": "",
        }
        for row in rows
    ]
    write_csv(args.output, export_rows)


if __name__ == "__main__":
    main()
