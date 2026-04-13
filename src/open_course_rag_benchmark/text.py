from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def sentence_split(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences if sentences else [text]


def chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[list[str]] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        start += step
    return chunks


def chunk_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = sentence_split(text)
    sentence_tokens = [tokenize(sentence) for sentence in sentences]
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        token_count = len(sentence_tokens[i])
        if not current_sentences:
            current_sentences.append(sentence)
            current_tokens = token_count
            i += 1
            continue
        if current_tokens + token_count <= chunk_size:
            current_sentences.append(sentence)
            current_tokens += token_count
            i += 1
            continue
        chunks.append(" ".join(current_sentences))
        overlap_sentences: list[str] = []
        overlap_tokens = 0
        for back_sentence in reversed(current_sentences):
            back_tokens = len(tokenize(back_sentence))
            if overlap_tokens + back_tokens > overlap and overlap_sentences:
                break
            overlap_sentences.insert(0, back_sentence)
            overlap_tokens += back_tokens
            if overlap_tokens >= overlap:
                break
        current_sentences = overlap_sentences if overlap_sentences else [sentence]
        current_tokens = sum(len(tokenize(item)) for item in current_sentences)
        if current_sentences == [sentence]:
            i += 1
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks
