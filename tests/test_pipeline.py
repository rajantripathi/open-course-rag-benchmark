from __future__ import annotations

from pathlib import Path

from open_course_rag_benchmark.build_benchmark import summarize_questions
from open_course_rag_benchmark.chunk_docs import chunk_documents
from open_course_rag_benchmark.eval_grounding import summarize as summarize_grounding
from open_course_rag_benchmark.eval_retrieval import evaluate
from open_course_rag_benchmark.ingest import discover_documents
from open_course_rag_benchmark.io_utils import read_csv, read_jsonl
from open_course_rag_benchmark.retrieve_bm25 import run_bm25


FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_pipeline(tmp_path) -> None:
    documents = discover_documents((FIXTURES / "raw_docs").resolve())
    course_ids = ["openstax_data_science", "mit_ethics"]
    for document, course_id in zip(documents, course_ids):
        document["course_id"] = course_id
        document["license"] = "open"

    chunks = chunk_documents(documents, chunk_size=64, overlap=16)
    assert any(chunk["chunk_id"] == "data_science_intro_000" for chunk in chunks)

    questions = read_jsonl(FIXTURES / "questions.jsonl")
    results = run_bm25(chunks, questions, top_k=2)
    gold = read_jsonl(FIXTURES / "gold_labels.jsonl")
    metrics = evaluate(results, gold)

    assert metrics["Recall@1"] == 1.0
    assert metrics["Recall@3"] == 1.0
    assert summarize_questions(questions)["total_questions"] == 2


def test_grounding_summary(tmp_path) -> None:
    rows = read_csv(FIXTURES / "grounding_annotations.csv")
    summary = summarize_grounding(rows)
    assert summary["label_distribution"]["correct_grounded"] == 2
    assert summary["intra_annotator_kappa"] == 1.0
