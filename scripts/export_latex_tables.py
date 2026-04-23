from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def to_booktabs(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, escape=False, longtable=False, bold_rows=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSV summaries as LaTeX tables.")
    default_results = Path(os.environ.get("SCRATCH_ROOT", "")) / "results" / "tables" if os.environ.get("SCRATCH_ROOT") else Path("results/tables")
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/raw/source_manifest_template.csv"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_main = pd.read_csv(args.results_dir / "retrieval_overall.csv")
    (args.output_dir / "retrieval_main.tex").write_text(to_booktabs(retrieval_main), encoding="utf-8")

    retrieval_significance = pd.read_csv(args.results_dir / "retrieval_significance.csv")
    (args.output_dir / "retrieval_significance.tex").write_text(to_booktabs(retrieval_significance), encoding="utf-8")

    error_categories = pd.read_csv(args.results_dir / "error_categories.csv")
    (args.output_dir / "error_categories.tex").write_text(to_booktabs(error_categories), encoding="utf-8")

    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    benchmark_rows = (
        pd.DataFrame(questions)
        .assign(base_type=lambda df: df["question_type"])
        .groupby(["course_id", "language", "base_type"])
        .size()
        .reset_index(name="count")
    )
    (args.output_dir / "benchmark_stats.tex").write_text(to_booktabs(benchmark_rows), encoding="utf-8")

    corpus_stats = pd.read_csv(args.results_dir / "retrieval_by_course.csv")
    manifest = pd.read_csv(args.source_manifest)
    manifest = manifest.rename(columns={"course_id": "course_id", "license": "license"})
    corpus_stats = corpus_stats[["course_id"]].drop_duplicates().merge(manifest[["course_id", "license"]], on="course_id", how="left")
    (args.output_dir / "corpus_stats.tex").write_text(to_booktabs(corpus_stats), encoding="utf-8")


if __name__ == "__main__":
    main()
