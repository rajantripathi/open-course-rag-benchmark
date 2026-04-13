from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .io_utils import read_jsonl, write_jsonl


def translate_text(generator, text: str) -> str:
    prompt = (
        "Translate the following educational question from English to Uzbek.\n"
        "Maintain the academic register and technical terminology.\n"
        "If a technical term has no standard Uzbek equivalent, keep it in English\n"
        "with a brief Uzbek explanation in parentheses.\n\n"
        f"English: {text}\nUzbek:"
    )
    generated = generator(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    return generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Translate curated questions and answers to Uzbek.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args(argv)
    records = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = []
    for row in records:
        translated = dict(row)
        translated["language"] = "uz"
        translated["question_text"] = translate_text(generator, row["question_text"])
        translated["reference_answer"] = translate_text(generator, row["reference_answer"])
        output.append(translated)
    write_jsonl(args.output, output)
    print(f"Wrote translations to {args.output}")


if __name__ == "__main__":
    main()
