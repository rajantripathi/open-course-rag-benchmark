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


def split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    words = normalize_whitespace(sentence).split()
    if not words:
        return []
    return [" ".join(chunk) for chunk in chunk_words(words, chunk_size, overlap)]


def chunk_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = sentence_split(text)
    sentence_tokens = [tokenize(sentence) for sentence in sentences]
    token_counts = [len(tokens) for tokens in sentence_tokens]
    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        sentence = sentences[start]
        token_count = token_counts[start]
        if token_count > chunk_size:
            chunks.extend(split_long_sentence(sentence, chunk_size, overlap))
            start += 1
            continue
        end = start
        total = 0
        while end < len(sentences):
            next_tokens = token_counts[end]
            if total + next_tokens > chunk_size:
                break
            total += next_tokens
            end += 1
        if end == start:
            chunks.extend(split_long_sentence(sentence, chunk_size, overlap))
            start += 1
            continue
        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        overlap_tokens = 0
        next_start = end
        for idx in range(end - 1, start - 1, -1):
            back_tokens = token_counts[idx]
            if overlap_tokens + back_tokens > overlap and overlap_tokens > 0:
                break
            overlap_tokens += back_tokens
            next_start = idx
            if overlap_tokens >= overlap:
                break
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks
