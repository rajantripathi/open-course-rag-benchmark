from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import append_jsonl, read_jsonl, write_jsonl


ABSTAIN = "Insufficient evidence to answer this question."


def evidence_prompt(question: str, evidence_chunks: list[dict]) -> str:
    evidence_text = "\n".join(f"[{chunk['chunk_id']}]: {chunk['text']}" for chunk in evidence_chunks)
    return (
        "You are an AI course assistant. Answer the student's question using ONLY\n"
        "the evidence provided below. Cite the chunk IDs you use in square brackets\n"
        "(e.g., [ds_ch03_007]).\n\n"
        "If the evidence does not contain enough information to answer the question\n"
        f"confidently, respond with: \"{ABSTAIN}\"\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {question}\nAnswer:"
    )


def normalize_existing_keys(rows: list[dict]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        keys.add((row.get("qid", ""), row.get("language", ""), row.get("system", "")))
    return keys


def build_generator(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device=0 if torch.cuda.is_available() else -1,
    )
    if getattr(generator.model, "generation_config", None) is not None:
        generator.model.generation_config.max_length = None
    return generator


def generate_answer_row(
    row: dict,
    *,
    chunk_by_id: dict[str, dict],
    question_by_qid: dict[str, dict],
    generator,
    max_new_tokens: int,
) -> dict:
    question = question_by_qid[row["qid"]]
    retrieved_ids = [item["chunk_id"] for item in row["ranked_chunks"][:5]]
    evidence = [chunk_by_id[chunk_id] for chunk_id in retrieved_ids if chunk_id in chunk_by_id]
    prompt = evidence_prompt(question["question_text"], evidence)
    generated = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    answer = generated.strip()
    return {
        "qid": row["qid"],
        "system": row["system"],
        "language": row["language"],
        "retrieved_chunk_ids": retrieved_ids,
        "answer": answer,
        "abstained": answer == ABSTAIN,
    }


def run_generation(
    chunks: list[dict],
    retrieval_results: list[dict],
    questions: list[dict],
    model_name: str,
    max_new_tokens: int,
    existing_keys: set[tuple[str, str, str]] | None = None,
    start_index: int = 0,
    max_records: int | None = None,
    output_path: Path | None = None,
    append: bool = False,
) -> list[dict]:
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    question_by_qid = {question["qid"]: question for question in questions}
    generator = build_generator(model_name)
    outputs: list[dict] = []
    processed = 0
    selected_rows = retrieval_results[start_index:]
    if max_records is not None:
        selected_rows = selected_rows[:max_records]
    for offset, row in enumerate(selected_rows, start=start_index):
        row_key = (row["qid"], row["language"], row["system"])
        if existing_keys and row_key in existing_keys:
            continue
        output_row = generate_answer_row(
            row,
            chunk_by_id=chunk_by_id,
            question_by_qid=question_by_qid,
            generator=generator,
            max_new_tokens=max_new_tokens,
        )
        processed += 1
        if append and output_path is not None:
            append_jsonl(output_path, [output_row])
        else:
            outputs.append(output_row)
        print(
            f"generated_batch={processed} generated_total={processed} retrieval_index={offset} "
            f"qid={row['qid']} language={row['language']}",
            flush=True,
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate grounded answers from retrieved evidence.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--append", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    existing_rows = read_jsonl(args.output) if args.append and args.output.exists() else []
    outputs = run_generation(
        read_jsonl(args.chunks),
        read_jsonl(args.retrieval),
        read_jsonl(args.questions),
        args.model_name,
        args.max_new_tokens,
        normalize_existing_keys(existing_rows),
        args.start_index,
        args.max_records,
        args.output,
        args.append,
    )
    if args.append:
        if outputs:
            append_jsonl(args.output, outputs)
    else:
        write_jsonl(args.output, outputs)
    print(f"Wrote generated answers to {args.output}")


if __name__ == "__main__":
    main()
