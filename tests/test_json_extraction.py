from open_course_rag_benchmark.generate_questions import extract_json_from_response


def test_clean_json() -> None:
    raw = '{"question_type": "factual", "question_text": "What is X?", "reference_answer": "X is Y."}'
    result = extract_json_from_response(raw)
    assert result is not None
    assert len(result) == 1
    assert result[0]["question_type"] == "factual"


def test_markdown_fenced() -> None:
    raw = '```json\n{"question_type": "factual", "question_text": "What is X?", "reference_answer": "X is Y."}\n```'
    result = extract_json_from_response(raw)
    assert result is not None
    assert len(result) == 1


def test_preamble_then_json() -> None:
    raw = 'Here is the question:\n\n{"question_type": "factual", "question_text": "What is X?", "reference_answer": "X is Y."}'
    result = extract_json_from_response(raw)
    assert result is not None
    assert len(result) == 1


def test_array_format() -> None:
    raw = '[{"question_type": "factual", "question_text": "Q1", "reference_answer": "A1"}, {"question_type": "conceptual", "question_text": "Q2", "reference_answer": "A2"}]'
    result = extract_json_from_response(raw)
    assert result is not None
    assert len(result) == 2


def test_garbage_returns_none() -> None:
    raw = "I cannot generate questions for this passage because it lacks sufficient content."
    result = extract_json_from_response(raw)
    assert result is None
