from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick


def save_plot(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pipeline_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.axis("off")
    stages = ["Corpus", "Chunking", "Retrieval", "Evaluation", "Analysis"]
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x, stage in zip(x_positions, stages):
        ax.text(x, 0.5, stage, ha="center", va="center", bbox={"boxstyle": "round,pad=0.4", "fc": "#d9edf7"})
    for start, end in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(end - 0.06, 0.5), xytext=(start + 0.06, 0.5), arrowprops={"arrowstyle": "->"})
    save_plot(fig, output_dir, "pipeline_diagram")


def add_confidence_interval(df: pd.DataFrame, metric: str, n: int) -> pd.DataFrame:
    df = df.copy()
    df["ci95"] = 1.96 * df[f"{metric}_std"].fillna(0.0) / (n ** 0.5)
    df["mean"] = df[f"{metric}_mean"]
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    sns.set_theme(style="whitegrid", palette="colorblind")
    pipeline_diagram(args.output_dir)

    by_language = pd.read_csv(args.results_dir / "tables" / "retrieval_by_language.csv")
    by_language = by_language.sort_values(["language", "system"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, language in zip(axes, ["en", "uz"]):
        subset = by_language[by_language["language"] == language].copy()
        subset = add_confidence_interval(subset, "Recall@5", 120)
        sns.barplot(data=subset, x="system", y="mean", ax=ax)
        ax.errorbar(range(len(subset)), subset["mean"], yerr=subset["ci95"], fmt="none", c="black", capsize=4)
        ax.set_title(language.upper())
        ax.set_ylabel("Recall@5")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    save_plot(fig, args.output_dir, "recall_at_k")

    by_course = pd.read_csv(args.results_dir / "tables" / "retrieval_by_course.csv")
    by_language_course = by_course.merge(by_language[["system", "language"]].drop_duplicates(), on="system")
    heatmap_df = by_language_course.assign(
        language_course=by_language_course["language"] + "-" + by_language_course["course_id"].map(
            {
                "openstax_data_science": "ds",
                "openstax_philosophy": "phil",
            }
        )
    ).pivot_table(index="system", columns="language_course", values="Recall@5_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heatmap_df, annot=True, cmap="crest", ax=ax)
    save_plot(fig, args.output_dir, "retrieval_heatmap")

    gap_df = by_language.pivot(index="system", columns="language", values="Recall@5_mean").reset_index()
    gap_df["gap"] = gap_df["en"] - gap_df["uz"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=gap_df, x="system", y="gap", ax=ax)
    ax.set_ylabel("English Recall@5 - Uzbek Recall@5")
    save_plot(fig, args.output_dir, "cross_lingual_gap")

    error_categories = pd.read_csv(args.results_dir / "tables" / "error_categories.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = error_categories.pivot_table(index="language", columns="category", values="count", aggfunc="sum", fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Count")
    save_plot(fig, args.output_dir, "error_category_distribution")


if __name__ == "__main__":
    main()
