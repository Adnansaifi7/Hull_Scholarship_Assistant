import re
from typing import List

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "at", "with", "as", "by", "this", "that", "it", "be",
}


def content_words(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def citation_accuracy(answer: str, citations: List[str], retrieved_chunks: List[dict]) -> float:
    if not citations:
        return 0.0
    chunk_lookup = {}
    for c in retrieved_chunks:
        for cid in c.get("window_chunk_ids", [c["chunk_id"]]):
            chunk_lookup[cid] = c["text"]
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0

    correct = 0
    for cid in citations:
        chunk_text = chunk_lookup.get(cid)
        if chunk_text is None:
            continue
        overlap = content_words(chunk_text) & answer_words
        if len(overlap) / max(len(answer_words), 1) > 0.15:
            correct += 1
    return correct / len(citations)


def faithfulness(answer: str, retrieved_chunks: List[dict]) -> float:
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    context_words = set()
    for c in retrieved_chunks:
        context_words |= content_words(c["text"])
    grounded = answer_words & context_words
    return len(grounded) / len(answer_words)
