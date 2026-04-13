from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .io_utils import read_jsonl


def summarize_questions(questions: list[dict]) -> dict:
    return {
        "total_questions": len(questions),
        "by_course": dict(Counter(question["course_id"] for question in questions)),
        "by_language": dict(Counter(question["language"] for question in questions)),
        "by_question_type": dict(Counter(question["question_type"] for question in questions)),
    }


def validate_labels(questions: list[dict], labels: list[dict]) -> None:
    question_ids = {question["qid"] for question in questions}
    label_ids = {label["qid"] for label in labels}
    missing = sorted(question_ids - label_ids)
    if missing:
        raise ValueError(f"missing labels for question ids: {missing[:5]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate benchmark question and label files.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    questions = read_jsonl(args.questions)
    labels = read_jsonl(args.gold)
    validate_labels(questions, labels)
    summary = summarize_questions(questions)
    summary["expected_total"] = args.expected_total
    summary["meets_expected_total"] = summary["total_questions"] == args.expected_total
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote benchmark summary to {args.output}")


if __name__ == "__main__":
    main()

