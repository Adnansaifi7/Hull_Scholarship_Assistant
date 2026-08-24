"""
Multi-query retrieval (related to "RAG-Fusion").

Idea: a single phrasing of a question might miss relevant chunks that use
different vocabulary. Instead of searching once with the user's exact
wording, ask the LLM to generate 2-3 alternative phrasings of the same
question, run dense retrieval for EACH phrasing, then merge the results
(here: by summing normalised scores per chunk, rewarding chunks that show
up as relevant across multiple phrasings).

Why this can help: "Am I eligible?" and "What are the entry requirements?"
and "Who can apply?" might retrieve overlapping-but-different chunks from
the same policy document, giving broader coverage than one query alone.

Why this can hurt: more retrieval calls = more chances for an off-topic
reformulation to pull in an irrelevant chunk that then dilutes the ranking.
"""

import os
import re
from typing import List, Optional

import requests

from retrieve import Retriever

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "phi3")

MULTIQUERY_PROMPT = """Generate exactly 3 different ways to ask this question about \
University of Hull scholarships and finance. Each should use different wording but \
mean the same thing. Respond with ONLY the 3 questions, one per line, numbered 1-3, \
nothing else.

Original question: {query}

Alternative phrasings:"""


def generate_query_variants(query: str, n: int = 3) -> List[str]:
    prompt = MULTIQUERY_PROMPT.format(query=query)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.5}},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    # Parse numbered list output; fall back to the original query alone if
    # the model didn't follow the format (defensive, same principle as
    # generate.py's JSON parsing - never trust the model's format compliance).
    lines = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", raw, re.MULTILINE)
    variants = [l.strip() for l in lines if l.strip()][:n]
    if not variants:
        return [query]
    return variants


class MultiQueryRetriever:
    def __init__(self):
        self.base = Retriever()

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        variants = generate_query_variants(query)
        all_queries = [query] + variants  # always include the original

        fetch_k = max(top_k * 2, 8)
        combined_scores = {}
        chunk_lookup = {}
        for q in all_queries:
            hits = self.base.search(q, top_k=fetch_k, doc_type_filter=doc_type_filter)
            for hit in hits:
                combined_scores[hit["chunk_id"]] = combined_scores.get(hit["chunk_id"], 0.0) + hit["score"]
                chunk_lookup[hit["chunk_id"]] = hit

        ranked = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for cid, score in ranked:
            hit = dict(chunk_lookup[cid])
            hit["score"] = float(score)
            hit["query_variants_used"] = all_queries  # kept for inspection/debugging
            results.append(hit)
        return results


if __name__ == "__main__":
    r = MultiQueryRetriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}: {hit['text'][:80]}")
