from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import append_jsonl, read_jsonl


def translate_text(generator, text: str) -> str:
    prompt = (
        "Translate the following educational question from English to Uzbek.\n"
        "Maintain the academic register and technical terminology.\n"
        "If a technical term has no standard Uzbek equivalent, keep it in English\n"
        "with a brief Uzbek explanation in parentheses.\n\n"
        f"English: {text}\nUzbek:"
    )
    generated = generator(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"]
    return generated.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Translate curated questions and answers to Uzbek.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--flush-every", type=int, default=10)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    all_records = read_jsonl(args.input)
    records = all_records[args.start_index :]
    if args.max_records is not None:
        records = records[: args.max_records]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device=0 if torch.cuda.is_available() else -1,
    )
    pending: list[dict] = []
    for index, row in enumerate(records, start=1):
        absolute_index = args.start_index + index
        translated = dict(row)
        translated["language"] = "uz"
        translated["question_text"] = translate_text(generator, row["question_text"])
        translated["reference_answer"] = translate_text(generator, row["reference_answer"])
        pending.append(translated)
        if len(pending) >= args.flush_every:
            append_jsonl(args.output, pending)
            pending = []
        if index % 10 == 0:
            print(
                f"translated_batch={index} translated_total={absolute_index} source_total={len(all_records)}",
                flush=True,
            )
    if pending:
        append_jsonl(args.output, pending)
    print(
        f"Wrote translations to {args.output} records={len(records)} start_index={args.start_index}",
        flush=True,
    )


if __name__ == "__main__":
    main()
