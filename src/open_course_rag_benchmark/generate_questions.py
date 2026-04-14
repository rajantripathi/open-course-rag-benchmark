from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

from .io_utils import append_jsonl, read_jsonl, write_jsonl


QUESTION_PROMPTS = {
    "factual": """You are creating a question benchmark for evaluating AI course assistants.

Given the following passage from a {course_name} textbook, write ONE factual question
that asks for a specific fact stated in the passage.

Also provide a short reference answer (2-3 sentences) using only information from the passage.

Passage:
{chunk_text}

Respond with ONLY a JSON object, no other text:
{{"question_type": "factual", "question_text": "...", "reference_answer": "..."}}""",
    "conceptual": """You are creating a question benchmark for evaluating AI course assistants.

Given the following passage from a {course_name} textbook, write ONE conceptual question
that asks a student to explain a concept or relationship described in the passage.

Also provide a short reference answer (2-3 sentences) using only information from the passage.

Passage:
{chunk_text}

Respond with ONLY a JSON object, no other text:
{{"question_type": "conceptual", "question_text": "...", "reference_answer": "..."}}""",
    "procedural": """You are creating a question benchmark for evaluating AI course assistants.

Given the following passage from a {course_name} textbook, write ONE question that either
asks how to perform a procedure described in the passage, or asks the student to compare
two concepts mentioned in the passage. Set question_type to "procedural" or "comparative"
as appropriate.

Also provide a short reference answer (2-3 sentences) using only information from the passage.

Passage:
{chunk_text}

Respond with ONLY a JSON object, no other text:
{{"question_type": "procedural", "question_text": "...", "reference_answer": "..."}}""",
}


def course_name(course_id: str) -> str:
    return {
        "openstax_data_science": "data science",
        "openstax_philosophy": "philosophy",
    }.get(course_id, course_id)


def extract_json_from_response(raw: str) -> Optional[list[dict]]:
    """Extract JSON from model output, handling fences and preambles."""
    text = raw.strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            payload = json.loads(fence_match.group(1).strip())
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [payload]
        except json.JSONDecodeError:
            pass

    bracket_match = re.search(r"(\[.*\])", text, re.DOTALL)
    if bracket_match:
        try:
            payload = json.loads(bracket_match.group(1))
            if isinstance(payload, list):
                return payload
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            payload = json.loads(brace_match.group(1))
            if isinstance(payload, dict):
                return [payload]
        except json.JSONDecodeError:
            pass

    objects: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
    return objects or None


def sample_chunks(all_chunks: list[dict], seed: int) -> list[dict]:
    chunks_by_course: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        chunks_by_course.setdefault(chunk["course_id"], []).append(chunk)

    rng = random.Random(seed)
    selected_chunks: list[dict] = []
    for course_id, course_chunks in chunks_by_course.items():
        n_sample = min(70, len(course_chunks))
        selected_chunks.extend(rng.sample(course_chunks, n_sample))
    rng.shuffle(selected_chunks)
    return selected_chunks


def save_raw_output(debug_dir: Path, chunk_id: str, prompt_key: str, raw_output: str) -> None:
    debug_path = debug_dir / f"{chunk_id}__{prompt_key}.txt"
    debug_path.write_text(raw_output, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate candidate questions from chunk passages.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--per-course-target", type=int, default=200)
    parser.add_argument("--course-id")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    all_chunks = read_jsonl(args.chunks)
    if args.course_id:
        all_chunks = [chunk for chunk in all_chunks if chunk["course_id"] == args.course_id]
    chunks = sample_chunks(all_chunks, seed=args.seed)
    chunks = chunks[args.start_index :]
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    debug_dir = args.debug_dir or (args.output.parent / "debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    parse_failure_log = debug_dir / "parse_failures.log"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device=0 if torch.cuda.is_available() else -1,
    )
    counts: dict[str, int] = {}
    course_counts: dict[str, int] = {}
    if args.append and args.output.exists():
        for row in read_jsonl(args.output):
            course_id = row.get("course_id")
            if course_id:
                counts[course_id] = counts.get(course_id, 0) + 1
                course_counts[course_id] = course_counts.get(course_id, 0) + 1
    rows: list[dict] = []
    chunks_processed = 0
    raw_outputs_saved = 0
    parse_successes = 0
    parse_failures = 0
    total_candidates = 0

    for chunk in chunks:
        if counts.get(chunk["course_id"], 0) >= args.per_course_target:
            continue
        chunks_processed += 1
        for prompt_key, prompt_template in QUESTION_PROMPTS.items():
            if counts.get(chunk["course_id"], 0) >= args.per_course_target:
                break
            prompt = prompt_template.format(course_name=course_name(chunk["course_id"]), chunk_text=chunk["text"])
            raw_output = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )[0]["generated_text"].strip()
            raw_outputs_saved += 1
            save_raw_output(debug_dir, chunk["chunk_id"], prompt_key, raw_output)
            items = extract_json_from_response(raw_output)
            if items is None:
                parse_failures += 1
                with parse_failure_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"{chunk['chunk_id']}\t{prompt_key}\t{raw_output[:200]!r}\n")
                continue
            parse_successes += 1
            fresh_rows: list[dict] = []
            for idx, item in enumerate(items):
                if counts.get(chunk["course_id"], 0) >= args.per_course_target:
                    break
                if not isinstance(item, dict):
                    continue
                if not {"question_type", "question_text", "reference_answer"} <= set(item):
                    continue
                question_type = item["question_type"]
                if prompt_key != "procedural" and question_type != prompt_key:
                    question_type = prompt_key
                if prompt_key == "procedural" and question_type not in {"procedural", "comparative"}:
                    question_type = "procedural"
                row = {
                    "candidate_id": f"{chunk['chunk_id']}_{question_type}_{idx}",
                    "chunk_id": chunk["chunk_id"],
                    "course_id": chunk["course_id"],
                    "question_type": question_type,
                    "question_text": item["question_text"],
                    "reference_answer": item["reference_answer"],
                }
                fresh_rows.append(row)
                rows.append(row)
                counts[chunk["course_id"]] = counts.get(chunk["course_id"], 0) + 1
                course_counts[chunk["course_id"]] = course_counts.get(chunk["course_id"], 0) + 1
                total_candidates += 1
            if args.append and fresh_rows:
                append_jsonl(args.output, fresh_rows)
                print(
                    "chunk={} type={} wrote={} totals={}".format(
                        chunk["chunk_id"],
                        prompt_key,
                        len(fresh_rows),
                        ",".join(f"{key}:{counts[key]}" for key in sorted(counts)),
                    ),
                    flush=True,
                )
    if not args.append:
        write_jsonl(args.output, rows)

    print("\n=== Generation Summary ===")
    print(f"Chunks processed: {chunks_processed}")
    print(f"Raw outputs saved: {raw_outputs_saved}")
    print(f"Parse successes: {parse_successes}")
    print(f"Parse failures: {parse_failures}")
    print(f"Total candidates written: {total_candidates}")
    print("Candidates per course:")
    for course_id, count in sorted(course_counts.items()):
        print(f"  {course_id}: {count}")
    print(f"Wrote {len(rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
