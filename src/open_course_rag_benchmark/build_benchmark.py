from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import read_jsonl


EXPECTED_TYPES = {"factual": 20, "conceptual": 20, "procedural": 10, "comparative": 10}
EXPECTED_COURSES = {"openstax_data_science", "openstax_philosophy"}


def validate(questions: list[dict], labels: list[dict], chunks: list[dict]) -> list[str]:
    errors: list[str] = []
    if len(questions) != 120:
        errors.append(f"expected 120 questions, found {len(questions)}")
    if len(labels) != 120:
        errors.append(f"expected 120 labels, found {len(labels)}")
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    seen_prompts: set[tuple[str, str, str]] = set()
    by_course_type = defaultdict(Counter)
    translations = defaultdict(set)
    question_ids = set()
    for question in questions:
        question_ids.add(question["qid"])
        by_course_type[question["course_id"]][question["question_type"]] += 1
        translations[question["qid"]].add(question["language"])
        key = (question["course_id"], question["language"], question["question_text"].strip().lower())
        if key in seen_prompts:
            errors.append(f"duplicate question text detected: {question['qid']}")
        seen_prompts.add(key)
    for course_id in EXPECTED_COURSES:
        course_total = sum(by_course_type[course_id].values())
        if course_total != 60:
            errors.append(f"{course_id} expected 60 questions, found {course_total}")
        for question_type, expected in EXPECTED_TYPES.items():
            actual = by_course_type[course_id][question_type]
            if actual not in {expected, expected * 2}:
                errors.append(f"{course_id} {question_type} expected {expected} per language or {expected*2} total, found {actual}")
    label_qids = set()
    for label in labels:
        label_qids.add(label["qid"])
        if not label.get("gold_chunk_ids"):
            errors.append(f"{label['qid']} missing gold chunks")
        for chunk_id in label.get("gold_chunk_ids", []):
            if chunk_id not in chunk_ids:
                errors.append(f"{label['qid']} references missing chunk {chunk_id}")
    if question_ids != label_qids:
        errors.append("question and label qid sets do not match")
    for qid, langs in translations.items():
        if langs != {"en", "uz"}:
            errors.append(f"{qid} missing bilingual pair: {sorted(langs)}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate benchmark question and label files.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    questions = read_jsonl(args.questions)
    labels = read_jsonl(args.gold)
    chunks = read_jsonl(args.chunks)
    errors = validate(questions, labels, chunks)
    print(f"questions={len(questions)} labels={len(labels)} chunks={len(chunks)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Benchmark validation passed.")


if __name__ == "__main__":
    main()

