from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        target = dst / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble retrieval-paper package artifacts.")
    parser.add_argument("--scratch-root", type=Path, default=Path("/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark"))
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark/results"))
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark/paper_package"))
    args = parser.parse_args()

    pkg = args.output_dir
    pkg.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(args.results_dir / "tables", pkg / "all_retrieval_tables")
    copy_tree_contents(args.paper_dir / "tables", pkg / "all_latex_tables")
    copy_tree_contents(args.results_dir / "figures", pkg / "all_figures")
    shutil.copy2(args.paper_dir / "data_statement.md", pkg / "data_statement.md")
    shutil.copy2(args.paper_dir / "references.bib", pkg / "references.bib")

    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_questions = []
    for language in ("en", "uz"):
        sample_questions.extend([row for row in questions if row["language"] == language][:5])
    (pkg / "sample_questions.json").write_text(json.dumps(sample_questions, indent=2, ensure_ascii=False), encoding="utf-8")

    error_samples_path = args.results_dir / "tables" / "error_samples.csv"
    sample_retrieval_results = []
    if error_samples_path.exists():
        import csv

        with error_samples_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        sample_retrieval_results = rows[:10]
    (pkg / "sample_retrieval_results.json").write_text(json.dumps(sample_retrieval_results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {}
    for name in ("retrieval_overall", "retrieval_significance", "error_categories"):
        path = args.results_dir / "tables" / f"{name}.csv"
        if path.exists():
            summary[name] = path.read_text(encoding="utf-8")
    (pkg / "summary_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (pkg / "README.md").write_text(
        "# Paper Package\n\nThis package contains retrieval tables, LaTeX tables, figures, sample questions, and summary statistics.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
