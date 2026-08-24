"""
Hybrid retrieval: combines dense (embedding) and sparse (BM25) scores via
weighted linear fusion after min-max normalising each to [0,1] — the
simplest defensible fusion method, and a reasonable methodology default to
report and justify (more advanced options exist, e.g. Reciprocal Rank
Fusion, worth mentioning as future work if you have time).

This is the third arm of your retrieval comparison:
  1. Dense only    (retrieve.py)
  2. Sparse only    (bm25_retrieve.py)
  3. Hybrid         (this file)
"""

from typing import List, Optional

from retrieve import Retriever
from bm25_retrieve import BM25Retriever

DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5


def _normalise(scores: dict) -> dict:
    if not scores:
        return scores
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self):
        self.dense = Retriever()
        self.sparse = BM25Retriever()
        # Index metadata by chunk_id for quick lookup after fusing scores
        self.chunk_lookup = {c["chunk_id"]: c for c in self.dense.metadata}

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        # Over-fetch from each retriever so fusion has enough candidates to work with
        fetch_k = max(top_k * 3, 10)
        dense_hits = self.dense.search(query, top_k=fetch_k, doc_type_filter=doc_type_filter)
        sparse_hits = self.sparse.search(query, top_k=fetch_k, doc_type_filter=doc_type_filter)

        dense_scores = _normalise({h["chunk_id"]: h["score"] for h in dense_hits})
        sparse_scores = _normalise({h["chunk_id"]: h["score"] for h in sparse_hits})

        all_ids = set(dense_scores) | set(sparse_scores)
        fused = {
            cid: DENSE_WEIGHT * dense_scores.get(cid, 0.0) + SPARSE_WEIGHT * sparse_scores.get(cid, 0.0)
            for cid in all_ids
        }
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for cid, score in ranked:
            m = self.chunk_lookup[cid]
            results.append({
                "chunk_id": m["chunk_id"],
                "source_file": m["source_file"],
                "doc_type": m["doc_type"],
                "authority": m["authority"],
                "text": m["text"],
                "score": float(score),
            })
        return results


if __name__ == "__main__":
    r = HybridRetriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}: {hit['text'][:80]}")
