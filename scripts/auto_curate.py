from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from open_course_rag_benchmark.io_utils import read_csv, write_csv


TARGETS = {
    "openstax_data_science": {"factual": 20, "conceptual": 20, "procedural": 10, "comparative": 10},
    "openstax_philosophy": {"factual": 20, "conceptual": 20, "procedural": 10, "comparative": 10},
}


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def section_key(row: dict) -> str:
    return row["chunk_id"].rsplit("_", 1)[0]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def question_quality(row: dict) -> tuple[int, int, int]:
    question = row["question_text_en"].strip()
    answer = row["reference_answer_en"].strip()
    score = 0
    if question.endswith("?"):
        score += 3
    if "..." not in question:
        score += 2
    q_words = word_count(question)
    if 6 <= q_words <= 28:
        score += 2
    elif q_words < 4 or q_words > 40:
        score -= 3
    a_words = word_count(answer)
    if 12 <= a_words <= 80:
        score += 2
    elif a_words < 8 or a_words > 120:
        score -= 2
    if row["question_type"] == "comparative" and "differ" in question.lower():
        score += 1
    # Stable tie-breakers: shorter questions, then candidate id.
    return (score, -q_words, -a_words)


def suspicious_uz(row: dict) -> list[str]:
    text = row["question_text_uz"].strip()
    flags: list[str] = []
    if not text:
        flags.append("missing_uz_question")
        return flags
    if "..." in text:
        flags.append("ellipsis_uz")
    if word_count(text) < 4:
        flags.append("very_short_uz")
    # Current machine output often drifts into mixed Cyrillic/Latin gibberish.
    if re.search(r"[А-Яа-яЁё]", text):
        flags.append("contains_cyrillic")
    ratio_non_ascii = sum(1 for ch in text if ord(ch) > 127) / max(len(text), 1)
    if ratio_non_ascii > 0.2:
        flags.append("high_non_ascii_ratio")
    return flags


def round_robin_sections(rows: list[dict], limit: int) -> list[dict]:
    buckets: dict[str, deque[dict]] = {}
    for row in rows:
        buckets.setdefault(section_key(row), deque()).append(row)
    ordered_sections = sorted(
        buckets,
        key=lambda key: (question_quality(buckets[key][0]), key),
        reverse=True,
    )
    selected: list[dict] = []
    used_ids: set[str] = set()
    while len(selected) < limit and ordered_sections:
        next_sections: list[str] = []
        for key in ordered_sections:
            bucket = buckets[key]
            while bucket and bucket[0]["candidate_id"] in used_ids:
                bucket.popleft()
            if not bucket:
                continue
            row = bucket.popleft()
            used_ids.add(row["candidate_id"])
            selected.append(row)
            if bucket:
                next_sections.append(key)
            if len(selected) == limit:
                break
        ordered_sections = next_sections
    return selected


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], int]:
    best_by_candidate: dict[str, dict] = {}
    duplicate_count = 0
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in best_by_candidate:
            duplicate_count += 1
            current = best_by_candidate[candidate_id]
            if (question_quality(row), row["candidate_id"]) > (question_quality(current), current["candidate_id"]):
                best_by_candidate[candidate_id] = row
        else:
            best_by_candidate[candidate_id] = row
    deduped = sorted(best_by_candidate.values(), key=lambda row: row["candidate_id"])
    return deduped, duplicate_count


def candidate_copy(row: dict, *, question_type: str | None = None, note: str = "") -> dict:
    updated = dict(row)
    if question_type is not None:
        updated["question_type"] = question_type
    updated["selected"] = "1"
    updated["gold_chunk_ids"] = row["chunk_id"]
    updated["notes"] = note
    return updated


def build_review_flag(row: dict, flag: str) -> dict:
    return {
        "candidate_id": row["candidate_id"],
        "course_id": row["course_id"],
        "chunk_id": row["chunk_id"],
        "question_type": row["question_type"],
        "flag": flag,
        "question_text_en": row["question_text_en"],
        "question_text_uz": row["question_text_uz"],
        "notes": row.get("notes", ""),
    }


def select_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    by_course_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_course_type[row["course_id"]][row["question_type"]].append(row)
    for course_id in by_course_type:
        for question_type in by_course_type[course_id]:
            by_course_type[course_id][question_type].sort(
                key=lambda row: (question_quality(row), row["candidate_id"]),
                reverse=True,
            )

    selected: list[dict] = []
    review_flags: list[dict] = []
    summary_lines: list[str] = []

    for course_id, targets in TARGETS.items():
        chosen_ids: set[str] = set()
        chosen_prompts: set[tuple[str, str]] = set()
        course_rows: list[dict] = []
        for question_type, target in targets.items():
            pool = []
            seen_pool_prompts: set[tuple[str, str]] = set()
            for row in by_course_type[course_id][question_type]:
                prompt_keys = {
                    ("en", normalize_text(row["question_text_en"])),
                    ("uz", normalize_text(row["question_text_uz"])),
                }
                if row["candidate_id"] in chosen_ids or prompt_keys & chosen_prompts or prompt_keys & seen_pool_prompts:
                    continue
                pool.append(row)
                seen_pool_prompts |= prompt_keys
            picks = round_robin_sections(pool, target)
            if len(picks) < target and course_id == "openstax_philosophy" and question_type == "procedural":
                fallback_pool = [
                    row
                    for row in by_course_type[course_id]["comparative"]
                    if row["candidate_id"] not in chosen_ids
                    and row["candidate_id"] not in {pick["candidate_id"] for pick in picks}
                    and {
                        ("en", normalize_text(row["question_text_en"])),
                        ("uz", normalize_text(row["question_text_uz"])),
                    }.isdisjoint(chosen_prompts)
                ]
                fallback_pool.sort(
                    key=lambda row: (
                        "how" in row["question_text_en"].lower(),
                        "differ" in row["question_text_en"].lower() or "compare" in row["question_text_en"].lower(),
                        question_quality(row),
                        row["candidate_id"],
                    ),
                    reverse=True,
                )
                if fallback_pool:
                    relabeled = candidate_copy(
                        fallback_pool[0],
                        question_type="procedural",
                        note="auto_relabelled_from_comparative_to_fill_procedural_quota",
                    )
                    picks.append(relabeled)
                    review_flags.append(build_review_flag(relabeled, "relabelled_type"))
            if len(picks) < target:
                raise SystemExit(f"insufficient candidates for {course_id} {question_type}: need {target}, found {len(picks)}")
            normalized_picks: list[dict] = []
            for pick in picks[:target]:
                if pick.get("selected") != "1":
                    pick = candidate_copy(pick)
                normalized_picks.append(pick)
                chosen_ids.add(pick["candidate_id"])
                chosen_prompts.add(("en", normalize_text(pick["question_text_en"])))
                chosen_prompts.add(("uz", normalize_text(pick["question_text_uz"])))
            course_rows.extend(normalized_picks)
        selected.extend(course_rows)
        summary_lines.append(
            f"{course_id}: "
            + ", ".join(f"{question_type}={sum(1 for row in course_rows if row['question_type'] == question_type)}" for question_type in targets)
        )

    for row in selected:
        for flag in suspicious_uz(row):
            review_flags.append(build_review_flag(row, flag))
        if len(row["question_text_en"]) > 220:
            review_flags.append(build_review_flag(row, "long_english_question"))

    return selected, review_flags, summary_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically select a balanced benchmark draft and flag items for review.")
    parser.add_argument("--input", type=Path, required=True, help="Path to bilingual curation_sheet.csv")
    parser.add_argument("--output", type=Path, required=True, help="Path to auto_curated_sheet.csv")
    parser.add_argument("--review-flags", type=Path, required=True, help="Path to review_flags.csv")
    args = parser.parse_args()

    source_rows = read_csv(args.input)
    rows, duplicate_input_rows = dedupe_rows(source_rows)
    selected, review_flags, summary_lines = select_rows(rows)

    selected_by_id = {row["candidate_id"]: row for row in selected}
    output_rows = []
    for row in rows:
        updated = dict(row)
        if row["candidate_id"] in selected_by_id:
            chosen = selected_by_id[row["candidate_id"]]
            updated["selected"] = "1"
            updated["gold_chunk_ids"] = chosen["gold_chunk_ids"]
            updated["notes"] = chosen["notes"]
            updated["question_type"] = chosen["question_type"]
        else:
            updated["selected"] = ""
            updated["gold_chunk_ids"] = ""
            updated["notes"] = ""
        output_rows.append(updated)

    write_csv(args.output, output_rows)
    review_flags.sort(key=lambda row: (row["course_id"], row["flag"], row["candidate_id"]))
    write_csv(args.review_flags, review_flags)

    selected_counter = Counter((row["course_id"], row["question_type"]) for row in selected)
    print("=== Auto-curation summary ===")
    print(f"input_rows={len(source_rows)}")
    print(f"deduped_rows={len(rows)}")
    print(f"duplicate_input_rows={duplicate_input_rows}")
    print(f"selected_rows={len(selected)}")
    print(f"review_flags={len(review_flags)}")
    for line in summary_lines:
        print(line)
    print("selected_breakdown:")
    for course_id in sorted(TARGETS):
        for question_type in ["factual", "conceptual", "procedural", "comparative"]:
            print(f"  {course_id} {question_type}={selected_counter[(course_id, question_type)]}")


if __name__ == "__main__":
    main()
