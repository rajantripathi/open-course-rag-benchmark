from __future__ import annotations

import argparse
import re
from pathlib import Path

from .io_utils import append_jsonl, read_jsonl, write_jsonl


ABSTAIN = "Insufficient evidence to answer this question."
LEAKAGE_MARKERS = [
    "However, I've",
    "Note:",
    "The revised response",
    "Here is the revised response",
]


def evidence_prompt(question: str, evidence_chunks: list[dict], language: str) -> str:
    evidence_text = "\n".join(f"[{chunk['chunk_id']}]: {chunk['text']}" for chunk in evidence_chunks)
    target_language = "English" if language == "en" else "Uzbek"
    return (
        "You are an AI course assistant.\n"
        f"Respond in the SAME LANGUAGE as the question. Use {target_language} only.\n"
        "Use ONLY the evidence provided below.\n"
        "Cite chunk IDs inline in square brackets inside your sentences, for example [ds_ch03_007].\n"
        "Do not add notes, revisions, explanations about the prompt, or meta-commentary.\n"
        "End your answer with the exact token <END>.\n\n"
        "Example 1\n"
        "Question: What is the median?\n"
        "Evidence:\n"
        "[ex_en_001]: The median is the middle value in an ordered dataset.\n"
        "Answer: The median is the middle value in an ordered dataset [ex_en_001]. <END>\n\n"
        "Example 2\n"
        "Question: Maksimum nima?\n"
        "Evidence:\n"
        "[ex_uz_001]: Maksimum ma'lumotlar to'plamidagi eng katta qiymatdir.\n"
        "Answer: Maksimum ma'lumotlar to'plamidagi eng katta qiymatdir [ex_uz_001]. <END>\n\n"
        "If the evidence does not contain enough information to answer the question confidently,\n"
        f'respond exactly: "{ABSTAIN} [chunk_id] <END>" using one or more retrieved chunk IDs.\n\n'
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {question}\nAnswer:"
    )


def normalize_existing_keys(rows: list[dict]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        keys.add((row.get("qid", ""), row.get("language", ""), row.get("system", "")))
    return keys


def strip_after_markers(text: str) -> str:
    cleaned = text
    for marker in LEAKAGE_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def cited_chunk_ids(answer: str) -> list[str]:
    return re.findall(r"\[([^\[\]]+)\]", answer)


def ascii_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 1.0
    ascii_chars = sum(1 for ch in visible if ord(ch) < 128)
    return ascii_chars / len(visible)


def build_generator(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device


def generate_answer_row(
    row: dict,
    *,
    chunk_by_id: dict[str, dict],
    question_by_qid: dict[str, dict],
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
) -> dict:
    question = question_by_qid[row["qid"]]
    retrieved_ids = [item["chunk_id"] for item in row["ranked_chunks"][:5]]
    evidence = [chunk_by_id[chunk_id] for chunk_id in retrieved_ids if chunk_id in chunk_by_id]
    prompt = evidence_prompt(question["question_text"], evidence, question["language"])
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        max_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    answer = generated.split("<END>", 1)[0].strip()
    answer = strip_after_markers(answer)
    valid_citations = [chunk_id for chunk_id in cited_chunk_ids(answer) if chunk_id in chunk_by_id]
    generation_failures: list[str] = []
    if question["language"] == "en" and ascii_ratio(answer) < 0.7:
        generation_failures.append("language_mismatch")
    if cited_chunk_ids(answer) and len(valid_citations) != len(cited_chunk_ids(answer)):
        generation_failures.append("invalid_citations")
    if not answer:
        generation_failures.append("empty_answer")
    return {
        "qid": row["qid"],
        "system": row["system"],
        "language": row["language"],
        "retrieved_chunk_ids": retrieved_ids,
        "answer": answer,
        "abstained": answer == ABSTAIN,
        "generation_failure": "|".join(generation_failures),
        "valid_citation_count": len(valid_citations),
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
    model, tokenizer, device = build_generator(model_name)
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
            model=model,
            tokenizer=tokenizer,
            device=device,
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
