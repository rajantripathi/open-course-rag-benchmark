from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_course_rag_benchmark.eval_grounding import cohens_kappa
from open_course_rag_benchmark.io_utils import read_csv, write_csv


ALLOWED = {"correct_grounded", "partially_correct", "unsupported", "hallucinated"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import grounding labels and compute summary tables.")
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.sheet)
    for row in rows:
        if row["label"] not in ALLOWED:
            raise ValueError(f"invalid label: {row['label']}")
    unique_rows = [row for row in rows if row.get("is_duplicate", "false") != "true"]
    write_csv(args.sample_output, unique_rows)
    duplicates = {}
    for row in rows:
        key = row.get("duplicate_group", "")
        if key:
            duplicates.setdefault(key, []).append(row["label"])
    pairs = [labels for labels in duplicates.values() if len(labels) == 2]
    kappa = cohens_kappa([pair[0] for pair in pairs], [pair[1] for pair in pairs]) if pairs else 0.0
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "grounding_agreement.csv", [{"metric": "cohens_kappa", "value": kappa}])
    df = pd.DataFrame(unique_rows)
    for group_col, name in [("system", "grounding_by_system.csv"), ("language", "grounding_by_language.csv"), ("course_id", "grounding_by_course.csv")]:
        summary = df.groupby(group_col)["label"].value_counts(normalize=True).unstack(fill_value=0).reset_index()
        write_csv(args.results_dir / name, summary.to_dict(orient="records"))


if __name__ == "__main__":
    main()

