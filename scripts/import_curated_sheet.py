from __future__ import annotations

import argparse
from pathlib import Path

from open_course_rag_benchmark.io_utils import read_csv, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the completed curation CSV into benchmark files.")
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--questions-output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for row in read_csv(args.sheet) if row["selected"].strip().lower() in {"1", "true", "yes", "y"}]
    translations = {row["candidate_id"]: row for row in read_jsonl(args.translations)}
    questions = []
    labels = []
    for idx, row in enumerate(rows, start=1):
        qid = f"Q{idx:03d}"
        questions.append(
            {
                "qid": qid,
                "course_id": row["course_id"],
                "language": "en",
                "question_type": row["question_type"],
                "question_text": row["question_text_en"],
                "reference_answer": row["reference_answer_en"],
            }
        )
        translated = translations.get(row["candidate_id"])
        if translated:
            questions.append(
                {
                    "qid": qid,
                    "course_id": row["course_id"],
                    "language": "uz",
                    "question_type": row["question_type"],
                    "question_text": translated["question_text"],
                    "reference_answer": translated["reference_answer"],
                }
            )
        labels.append(
            {
                "qid": qid,
                "gold_doc_ids": sorted({chunk_id.rsplit("_", 1)[0] for chunk_id in row["gold_chunk_ids"].split() if chunk_id}),
                "gold_chunk_ids": [chunk_id.strip() for chunk_id in row["gold_chunk_ids"].replace(",", " ").split() if chunk_id.strip()],
            }
        )
    write_jsonl(args.questions_output, questions)
    write_jsonl(args.labels_output, labels)


if __name__ == "__main__":
    main()

