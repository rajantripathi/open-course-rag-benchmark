from open_course_rag_benchmark.answer_generate import ascii_ratio, cited_chunk_ids, strip_after_markers


def test_strip_after_markers_truncates_leakage():
    text = "Good answer [c1]. However, I've rewritten the response."
    assert strip_after_markers(text) == "Good answer [c1]."


def test_cited_chunk_ids_extracts_inline_ids():
    assert cited_chunk_ids("Answer [c1] and [c2].") == ["c1", "c2"]


def test_ascii_ratio_flags_non_english_like_output():
    assert ascii_ratio("This is English.") > 0.7
    assert ascii_ratio("Навигациялардан толук машиналиги") < 0.7
