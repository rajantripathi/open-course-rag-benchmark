from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .io_utils import read_jsonl, write_csv


ABSTAIN = "Insufficient evidence to answer this question."


def cited_chunk_ids(answer: str) -> list[str]:
    return re.findall(r"\[([^\[\]]+)\]", answer)


def load_gold(path: Path) -> dict[str, dict]:
    return {row["qid"]: row for row in read_jsonl(path)}


def assess_answer(answer_row: dict, gold_row: dict) -> tuple[bool, str, str]:
    retrieved = answer_row.get("retrieved_chunk_ids", [])
    gold_chunks = set(gold_row.get("gold_chunk_ids", []))
    retrieved_gold = gold_chunks.intersection(retrieved)
    answer = answer_row.get("answer", "")
    abstained = bool(answer_row.get("abstained")) or answer.strip() == ABSTAIN
    citations = cited_chunk_ids(answer)
    cited_retrieved = set(citations).intersection(retrieved)

    if abstained and not retrieved_gold:
        return False, "unsupported", "auto_unsupported_no_gold"
    if abstained and retrieved_gold:
        return True, "", "review_abstained_with_gold"
    if not citations:
        return True, "", "review_missing_citations"
    if not cited_retrieved:
        return True, "", "review_invalid_citations"
    if not retrieved_gold:
        return True, "", "review_no_gold_retrieved"
    return False, "correct_grounded", "auto_grounded_with_gold"


def build_review_rows(answer_rows: list[dict], gold_by_qid: dict[str, dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    counts = Counter()
    auto_label_counts = Counter()
    for answer_row in answer_rows:
        gold_row = gold_by_qid[answer_row["qid"]]
        needs_review, auto_label, reason = assess_answer(answer_row, gold_row)
        counts[reason] += 1
        if needs_review:
            rows.append(
                {
                    "qid": answer_row["qid"],
                    "system": answer_row.get("system", ""),
                    "language": answer_row.get("language", ""),
                    "retrieved_chunk_ids": " ".join(answer_row.get("retrieved_chunk_ids", [])),
                    "gold_chunk_ids": " ".join(gold_row.get("gold_chunk_ids", [])),
                    "answer": answer_row.get("answer", ""),
                    "auto_label": auto_label,
                    "review_reason": reason,
                    "label": "",
                    "notes": "",
                }
            )
        elif auto_label:
            auto_label_counts[auto_label] += 1
    summary = {
        "total_answers": len(answer_rows),
        "manual_review_rows": len(rows),
        "auto_label_counts": dict(auto_label_counts),
        "reason_counts": dict(counts),
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-label grounded answers and emit only rows that need manual review.")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-sheet", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    answer_rows = read_jsonl(args.answers)
    gold_by_qid = load_gold(args.gold)
    review_rows, summary = build_review_rows(answer_rows, gold_by_qid)
    write_csv(args.output_sheet, review_rows)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
