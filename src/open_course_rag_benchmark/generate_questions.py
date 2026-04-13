from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .io_utils import read_jsonl, write_jsonl


PROMPT = """You are creating a question benchmark for evaluating AI course assistants.
Given the following passage from a {course_name} textbook, generate exactly
3 questions that a student might ask. Include:
- 1 factual question
- 1 conceptual question
- 1 procedural or comparative question

For each question, also provide a short reference answer (2-3 sentences)
using only information from the passage.

Passage:
{chunk_text}

Output as JSON array."""


def course_name(course_id: str) -> str:
    return {
        "openstax_data_science": "data science",
        "openstax_philosophy": "philosophy",
    }.get(course_id, course_id)


def extract_json_array(text: str) -> list[dict] | None:
    match = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate candidate questions from chunk passages.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--per-course-target", type=int, default=200)
    args = parser.parse_args(argv)

    chunks = read_jsonl(args.chunks)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device=0 if torch.cuda.is_available() else -1,
    )
    counts: dict[str, int] = {}
    rows: list[dict] = []
    for chunk in chunks[:: args.stride]:
        if counts.get(chunk["course_id"], 0) >= args.per_course_target:
            continue
        prompt = PROMPT.format(course_name=course_name(chunk["course_id"]), chunk_text=chunk["text"])
        text = generator(
            prompt,
            max_new_tokens=400,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"].strip()
        items = extract_json_array(text)
        if items is None:
            continue
        for idx, item in enumerate(items):
            if counts.get(chunk["course_id"], 0) >= args.per_course_target:
                break
            if not isinstance(item, dict):
                continue
            if not {"question_type", "question_text", "reference_answer"} <= set(item):
                continue
            rows.append(
                {
                    "candidate_id": f"{chunk['chunk_id']}_{idx}",
                    "chunk_id": chunk["chunk_id"],
                    "course_id": chunk["course_id"],
                    "question_type": item["question_type"],
                    "question_text": item["question_text"],
                    "reference_answer": item["reference_answer"],
                }
            )
            counts[chunk["course_id"]] = counts.get(chunk["course_id"], 0) + 1
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
