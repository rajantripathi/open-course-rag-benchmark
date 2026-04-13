from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def to_booktabs(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, escape=False, longtable=False, bold_rows=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSV summaries as LaTeX tables.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "retrieval_overall.csv": "retrieval_main.tex",
        "retrieval_by_language.csv": "benchmark_stats.tex",
        "grounding_by_system.csv": "grounding_results.tex",
        "retrieval_by_course.csv": "corpus_stats.tex",
    }
    for src, dst in mapping.items():
        df = pd.read_csv(args.results_dir / src)
        (args.output_dir / dst).write_text(to_booktabs(df), encoding="utf-8")


if __name__ == "__main__":
    main()

