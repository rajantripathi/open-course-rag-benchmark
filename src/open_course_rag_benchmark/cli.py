from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "scrape-openstax": "scrape_openstax",
    "ingest": "ingest",
    "chunk": "chunk_docs",
    "build-benchmark": "build_benchmark",
    "generate-questions": "generate_questions",
    "translate-questions": "translate_questions",
    "bm25": "retrieve_bm25",
    "dense": "retrieve_dense",
    "hybrid": "retrieve_hybrid",
    "answer": "answer_generate",
    "eval-retrieval": "eval_retrieval",
    "eval-grounding": "eval_grounding",
    "plots": "plots",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        available = ", ".join(COMMANDS)
        raise SystemExit(f"usage: ocrb <command> [args]\navailable commands: {available}")
    module = importlib.import_module(f".{COMMANDS[sys.argv[1]]}", package=__package__)
    module.main(sys.argv[2:])


if __name__ == "__main__":
    main()

