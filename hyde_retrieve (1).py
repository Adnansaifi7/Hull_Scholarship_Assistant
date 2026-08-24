"""
HyDE (Hypothetical Document Embeddings) retrieval.

Idea (Gao et al., 2022): instead of embedding the user's short question
directly, first ask the LLM to write a hypothetical, plausible-sounding
ANSWER to the question (even without any retrieved context), then embed
THAT hypothetical answer and search for real chunks similar to it.

Why this can help: a question like "Am I eligible?" is short and doesn't
share much vocabulary with the policy text that answers it. A hypothetical
answer like "You are eligible if you are a Home undergraduate student
starting in September..." shares far more vocabulary/structure with the
real policy chunk, which can improve embedding similarity matching.

Why this can also hurt: if the hypothetical answer is confidently WRONG
(the LLM hallucinates plausible-sounding but incorrect specifics), it can
steer retrieval toward the wrong chunks. Worth discussing this tradeoff
explicitly in your dissertation using whatever your comparison shows.

Reference: Gao, Ma, Lin, Callan, Chen, Neubig (2022), "Precise Zero-Shot
Dense Retrieval without Relevance Labels" (HyDE paper).
"""

import os
from typing import List, Optional

import requests

from retrieve import Retriever
from build_index import embed_text

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "phi3")

HYDE_PROMPT_TEMPLATE = """Write a brief, plausible-sounding answer (2-3 sentences) to \
this question about University of Hull scholarships and finance, as if you were \
answering from the official policy. Do not say you don't know - write your best \
guess at what the real policy would say, even if uncertain.

Question: {query}

Hypothetical answer:"""


def generate_hypothetical_answer(query: str) -> str:
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


class HyDERetriever:
    def __init__(self):
        self.base = Retriever()  # reuse the same vector index

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        hypothetical = generate_hypothetical_answer(query)

        # Same ranking logic as Retriever.search, but embedding the
        # hypothetical answer instead of the raw query
        import numpy as np
        q_vec = embed_text(hypothetical)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        scores = self.base.vectors @ q_vec

        candidate_idx = np.arange(len(self.base.metadata))
        if doc_type_filter:
            filtered = [i for i in candidate_idx if self.base.metadata[i]["doc_type"] in doc_type_filter]
            if filtered:
                candidate_idx = np.array(filtered)

        candidate_scores = scores[candidate_idx]
        order = np.argsort(-candidate_scores)[:top_k]
        top_idx = candidate_idx[order]

        results = []
        for idx in top_idx:
            m = self.base.metadata[idx]
            results.append({
                "chunk_id": m["chunk_id"],
                "source_file": m["source_file"],
                "doc_type": m["doc_type"],
                "authority": m["authority"],
                "text": m["text"],
                "score": float(scores[idx]),
                "hyde_hypothetical": hypothetical,  # kept for inspection/debugging
            })
        return results


if __name__ == "__main__":
    r = HyDERetriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}: {hit['text'][:80]}")
