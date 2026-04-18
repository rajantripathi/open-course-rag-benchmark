from open_course_rag_benchmark.auto_ground import assess_answer, build_review_rows


def test_assess_answer_auto_unsupported_when_abstained_without_gold():
    answer_row = {
        "qid": "Q1",
        "retrieved_chunk_ids": ["c9"],
        "answer": "Insufficient evidence to answer this question.",
        "abstained": True,
    }
    gold_row = {"qid": "Q1", "gold_chunk_ids": ["c1"]}
    needs_review, auto_label, reason = assess_answer(answer_row, gold_row)
    assert needs_review is False
    assert auto_label == "unsupported"
    assert reason == "auto_unsupported_no_gold"


def test_assess_answer_review_when_missing_citations():
    answer_row = {
        "qid": "Q1",
        "retrieved_chunk_ids": ["c1", "c2"],
        "answer": "This answers the question but cites nothing.",
        "abstained": False,
    }
    gold_row = {"qid": "Q1", "gold_chunk_ids": ["c1"]}
    needs_review, auto_label, reason = assess_answer(answer_row, gold_row)
    assert needs_review is True
    assert auto_label == ""
    assert reason == "review_missing_citations"


def test_build_review_rows_filters_auto_labeled_rows():
    answers = [
        {
            "qid": "Q1",
            "system": "hybrid",
            "language": "en",
            "retrieved_chunk_ids": ["c9"],
            "answer": "Insufficient evidence to answer this question.",
            "abstained": True,
        },
        {
            "qid": "Q2",
            "system": "hybrid",
            "language": "en",
            "retrieved_chunk_ids": ["c2"],
            "answer": "Answer without citations.",
            "abstained": False,
        },
    ]
    gold = {
        "Q1": {"qid": "Q1", "gold_chunk_ids": ["c1"]},
        "Q2": {"qid": "Q2", "gold_chunk_ids": ["c2"]},
    }
    rows, summary = build_review_rows(answers, gold)
    assert len(rows) == 1
    assert rows[0]["qid"] == "Q2"
    assert summary["total_answers"] == 2
    assert summary["manual_review_rows"] == 1
