from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_plot(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pipeline_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.axis("off")
    stages = ["Corpus", "Chunking", "Retrieval", "Answering", "Evaluation"]
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x, stage in zip(x_positions, stages):
        ax.text(x, 0.5, stage, ha="center", va="center", bbox={"boxstyle": "round,pad=0.4", "fc": "#d9edf7"})
    for start, end in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(end - 0.06, 0.5), xytext=(start + 0.06, 0.5), arrowprops={"arrowstyle": "->"})
    save_plot(fig, output_dir, "pipeline_diagram")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    sns.set_theme(style="whitegrid", palette="colorblind")
    pipeline_diagram(args.output_dir)

    by_language = pd.read_csv(args.results_dir / "tables" / "retrieval_by_language.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, language in zip(axes, ["en", "uz"]):
        subset = by_language[by_language["language"] == language]
        melted = subset.melt(id_vars=["system", "language"], value_vars=["Recall@1_mean", "Recall@3_mean", "Recall@5_mean"], var_name="metric", value_name="value")
        sns.barplot(data=melted, x="metric", y="value", hue="system", ax=ax)
        ax.set_title(language.upper())
        ax.set_ylim(0, 1)
    save_plot(fig, args.output_dir, "recall_at_k")

    by_course = pd.read_csv(args.results_dir / "tables" / "retrieval_by_course.csv")
    heatmap_df = by_course.pivot(index="system", columns="course_id", values="Recall@5_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heatmap_df, annot=True, cmap="crest", ax=ax)
    save_plot(fig, args.output_dir, "retrieval_heatmap")

    grounding = pd.read_csv(args.results_dir / "tables" / "grounding_by_system.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    grounding.set_index("system")[["correct_grounded", "partially_correct", "unsupported", "hallucinated"]].plot(kind="bar", stacked=True, ax=ax)
    save_plot(fig, args.output_dir, "groundedness_by_system")

    fig, ax = plt.subplots(figsize=(8, 4))
    grounding.set_index("system")["hallucinated"].sort_values().plot(kind="barh", ax=ax)
    save_plot(fig, args.output_dir, "error_type_distribution")


if __name__ == "__main__":
    main()

