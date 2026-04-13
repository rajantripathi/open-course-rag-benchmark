from __future__ import annotations

from pathlib import Path

from open_course_rag_benchmark.build_benchmark import validate
from open_course_rag_benchmark.chunk_docs import chunk_documents
from open_course_rag_benchmark.eval_grounding import summarize as summarize_grounding
from open_course_rag_benchmark.io_utils import read_csv, read_jsonl
from open_course_rag_benchmark.retrieve_bm25 import run_bm25


FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_pipeline() -> None:
    documents = [
        {
            "doc_id": "data_science_intro",
            "course_id": "openstax_data_science",
            "title": "What Is Data Science?",
            "text": (FIXTURES / "raw_docs" / "data_science_intro.txt").read_text(encoding="utf-8"),
        },
        {
            "doc_id": "philosophy_intro",
            "course_id": "openstax_philosophy",
            "title": "What Is Philosophy?",
            "text": (FIXTURES / "raw_docs" / "philosophy_intro.txt").read_text(encoding="utf-8"),
        },
    ]
    chunks = chunk_documents(documents, chunk_size=64, overlap=16)
    questions = read_jsonl(FIXTURES / "questions.jsonl")
    labels = read_jsonl(FIXTURES / "gold_labels.jsonl")
    results = run_bm25(chunks, questions, top_k=2, k1=1.5, b=0.75)
    assert len(results) == 4
    assert any(row["qid"] == "Q001" and row["language"] == "en" for row in results)
    errors = validate(questions, labels, chunks)
    assert "expected 120 questions" in errors[0]
    assert all("missing" not in error for error in errors if "chunk" in error)


def test_grounding_summary() -> None:
    rows = read_csv(FIXTURES / "grounding_annotations.csv")
    summary = summarize_grounding(rows)
    assert summary["label_distribution"]["correct_grounded"] == 2
    assert summary["intra_annotator_kappa"] == 1.0
