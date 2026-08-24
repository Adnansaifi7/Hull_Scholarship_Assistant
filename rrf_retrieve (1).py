"""
Reciprocal Rank Fusion (RRF) retrieval.

Your existing hybrid_retrieve.py combines dense + BM25 by normalising and
averaging their raw SCORES. RRF instead combines them by RANK POSITION,
using the formula:

    RRF_score(chunk) = sum over each ranker of  1 / (k + rank_in_that_ranker)

where k is a constant (60 is the standard default from the original paper)
and rank is 1-indexed position in that ranker's result list.

Why this is a genuinely different approach worth comparing: score-based
fusion (your hybrid_retrieve.py) is sensitive to how differently-scaled the
two rankers' raw scores are, even after normalisation. RRF only cares about
ORDER, not magnitude, which makes it more robust when one ranker's scores
are poorly calibrated or clustered - a common real-world issue. RRF is the
fusion method most production hybrid-search systems actually use (e.g.
Elasticsearch, Azure AI Search).

Reference: Cormack, Clarke, Buettcher (2009), "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods".
"""

from typing import List, Optional

from retrieve import Retriever
from bm25_retrieve import BM25Retriever

RRF_K = 60  # standard default from the original paper


class RRFRetriever:
    def __init__(self):
        self.dense = Retriever()
        self.sparse = BM25Retriever()
        self.chunk_lookup = {c["chunk_id"]: c for c in self.dense.metadata}

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        fetch_k = max(top_k * 3, 10)
        dense_hits = self.dense.search(query, top_k=fetch_k, doc_type_filter=doc_type_filter)
        sparse_hits = self.sparse.search(query, top_k=fetch_k, doc_type_filter=doc_type_filter)

        rrf_scores = {}
        for rank, hit in enumerate(dense_hits, start=1):
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0.0) + 1.0 / (RRF_K + rank)
        for rank, hit in enumerate(sparse_hits, start=1):
            rrf_scores[hit["chunk_id"]] = rrf_scores.get(hit["chunk_id"], 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]

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
    r = RRFRetriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.4f}] {hit['chunk_id']}: {hit['text'][:80]}")
