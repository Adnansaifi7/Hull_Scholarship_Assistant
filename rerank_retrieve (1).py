"""
LLM-based reranking retrieval.

Idea: embedding similarity is a fast but imperfect proxy for "is this chunk
actually relevant to the question." A two-stage approach often does better:
  1. Over-fetch a wider candidate set using cheap dense retrieval
  2. Ask the LLM to directly score each candidate's relevance to the query
  3. Re-sort by the LLM's relevance scores, keep the top_k

This is slower and more expensive per query (one extra LLM call per
candidate chunk) but can catch cases where embedding similarity is
misleading - e.g. a chunk that shares vocabulary with the query but isn't
actually the right answer.

For a dissertation-scale corpus (32 chunks) this is affordable; note in
your methodology that this would NOT scale cheaply to a large corpus
without a dedicated (cheaper) reranker model.
"""

import os
import re
from typing import List, Optional

import requests

from retrieve import Retriever

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "phi3")

RERANK_PROMPT_TEMPLATE = """On a scale of 0 to 10, how relevant is the following \
passage to answering the question? Respond with ONLY a single number, nothing else.

Question: {query}

Passage: {passage}

Relevance score (0-10):"""


def score_relevance(query: str, passage: str) -> float:
    prompt = RERANK_PROMPT_TEMPLATE.format(query=query, passage=passage[:600])
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()
    match = re.search(r"\d+(\.\d+)?", raw)
    if not match:
        return 0.0
    score = float(match.group(0))
    return max(0.0, min(10.0, score))  # clamp to expected range in case the model misbehaves


class RerankRetriever:
    def __init__(self):
        self.base = Retriever()

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
        fetch_k: int = 8,
    ) -> List[dict]:
        # Stage 1: cheap dense over-fetch
        candidates = self.base.search(query, top_k=fetch_k, doc_type_filter=doc_type_filter)

        # Stage 2: LLM scores each candidate directly
        for c in candidates:
            c["rerank_score"] = score_relevance(query, c["text"])
            c["dense_score"] = c["score"]  # preserve original for comparison/debugging

        # Stage 3: re-sort by LLM relevance score
        candidates.sort(key=lambda c: -c["rerank_score"])
        top = candidates[:top_k]
        for c in top:
            c["score"] = c["rerank_score"]  # so downstream code (evaluate.py etc.) sees the rerank score
        return top


if __name__ == "__main__":
    r = RerankRetriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[rerank={hit['rerank_score']:.1f} dense={hit['dense_score']:.3f}] {hit['chunk_id']}: {hit['text'][:80]}")
