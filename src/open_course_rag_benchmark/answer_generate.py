from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .io_utils import read_jsonl, write_jsonl


def evidence_prompt(question: str, evidence_chunks: list[dict]) -> str:
    evidence_text = "\n\n".join(
        f"[{chunk['chunk_id']}] {chunk['text']}" for chunk in evidence_chunks
    )
    return (
        "You are answering educational questions using retrieved evidence only.\n"
        "If the evidence is insufficient, answer exactly: insufficient evidence.\n"
        "Cite supporting chunk ids in the answer.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Answer:"
    )


def run_generation(
    chunks: list[dict],
    retrieval_results: list[dict],
    questions: list[dict],
    model_name: str,
    max_new_tokens: int,
) -> list[dict]:
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    question_by_id = {question["qid"]: question for question in questions}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    outputs: list[dict] = []
    for row in retrieval_results:
        question = question_by_id[row["qid"]]
        chunk_ids = [item["chunk_id"] for item in row["ranked_results"]]
        evidence = [chunk_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_by_id]
        prompt = evidence_prompt(question["question_text"], evidence)
        generated = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
        answer = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated.strip()
        outputs.append(
            {
                "qid": row["qid"],
                "system": row["system"],
                "retrieved_chunk_ids": chunk_ids,
                "answer": answer,
                "abstained": answer.lower() == "insufficient evidence.",
            }
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate answers from retrieved evidence.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    chunks = read_jsonl(args.chunks)
    retrieval_results = read_jsonl(args.retrieval)
    questions = read_jsonl(args.questions)
    outputs = run_generation(
        chunks=chunks,
        retrieval_results=retrieval_results,
        questions=questions,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(args.output, outputs)
    print(f"Wrote generated answers to {args.output}")


if __name__ == "__main__":
    main()

