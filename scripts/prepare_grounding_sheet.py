from __future__ import annotations

import argparse
import random
from pathlib import Path

from open_course_rag_benchmark.io_utils import read_jsonl, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a grounding evaluation CSV with duplicate rows.")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    answers = read_jsonl(args.answers)
    questions = {row["qid"]: row for row in read_jsonl(args.questions)}
    chunks = {row["chunk_id"]: row["text"] for row in read_jsonl(args.chunks)}
    sample = answers[:60]
    rows = []
    for idx, row in enumerate(sample):
        q = questions[row["qid"]]
        rows.append(
            {
                "qid": row["qid"],
                "system": row["system"],
                "language": row["language"],
                "course_id": q["course_id"],
                "question_type": q["question_type"],
                "question_text": q["question_text"],
                "retrieved_chunk_ids": " ".join(row["retrieved_chunk_ids"]),
                "evidence_text": "\n\n".join(chunks.get(chunk_id, "") for chunk_id in row["retrieved_chunk_ids"][:5]),
                "answer": row["answer"],
                "label": "",
                "notes": "",
                "duplicate_group": "",
                "is_duplicate": "false",
            }
        )
    duplicates = random.sample(rows, min(20, len(rows)))
    for idx, row in enumerate(duplicates, start=1):
        dup = dict(row)
        dup["duplicate_group"] = f"g{idx}"
        row["duplicate_group"] = f"g{idx}"
        dup["is_duplicate"] = "true"
        rows.append(dup)
    random.shuffle(rows)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()

