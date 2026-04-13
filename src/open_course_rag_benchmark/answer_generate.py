from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .io_utils import read_jsonl, write_jsonl


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


def run_generation(chunks: list[dict], retrieval_results: list[dict], questions: list[dict], model_name: str, max_new_tokens: int) -> list[dict]:
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    question_by_qid = {question["qid"]: question for question in questions}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outputs: list[dict] = []
    for row in retrieval_results:
        question = question_by_qid[row["qid"]]
        retrieved_ids = [item["chunk_id"] for item in row["ranked_chunks"][:5]]
        evidence = [chunk_by_id[chunk_id] for chunk_id in retrieved_ids if chunk_id in chunk_by_id]
        prompt = evidence_prompt(question["question_text"], evidence)
        generated = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
        answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
        outputs.append(
            {
                "qid": row["qid"],
                "system": row["system"],
                "language": row["language"],
                "retrieved_chunk_ids": retrieved_ids,
                "answer": answer,
                "abstained": answer == ABSTAIN,
            }
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate grounded answers from retrieved evidence.")
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
    outputs = run_generation(
        read_jsonl(args.chunks),
        read_jsonl(args.retrieval),
        read_jsonl(args.questions),
        args.model_name,
        args.max_new_tokens,
    )
    write_jsonl(args.output, outputs)
    print(f"Wrote generated answers to {args.output}")


if __name__ == "__main__":
    main()

