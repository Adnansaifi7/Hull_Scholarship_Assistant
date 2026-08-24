"""
Combined strategy: takes hybrid retrieval's ranking (score-fusion of dense +
sparse, which won on hit rate 0.560 and MRR 0.447 in the 8-strategy
comparison) and applies parent-window's context expansion to the results
(which won on citation accuracy 0.718 and faithfulness 0.762).

Rationale: these two strategies win on DIFFERENT, complementary axes -
hybrid finds the right chunk more often and ranks it higher; parent-window
gives the generator fuller context once a chunk is found. Combining them
means: rank with the strategy that's best at finding the right answer,
then generate with the strategy that's best at answering faithfully once
found. This is not guaranteed to beat both individually on every metric -
that must be verified empirically (Section 8 comparison) - but the design
rationale is genuine, not an attempt to game the evaluation metrics.
"""

from retrieve import Retriever
from bm25_retrieve import BM25Retriever
from parent_retrieve import ParentWindowRetriever

DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5


def _normalise(scores):
    if not scores:
        return scores
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridParentWindowRetriever:
    def __init__(self, window=1):
        self.dense = Retriever()
        self.sparse = BM25Retriever()
        self.parent_window = ParentWindowRetriever(window=window)
        self.chunk_lookup = {c["chunk_id"]: c for c in self.dense.metadata}

    def search(self, query, top_k=4, doc_type_filter=None):
        # Stage 1: rank using hybrid fusion (same logic as hybrid_retrieve.py)
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

        # Stage 2: expand each ranked hit's context using parent-window logic,
        # WITHOUT re-ranking - the hybrid order from stage 1 is preserved.
        results = []
        for cid, score in ranked:
            base_hit = dict(self.chunk_lookup[cid])
            base_hit["score"] = float(score)
            expanded = self.parent_window.expand_hit(base_hit)
            results.append(expanded)
        return results


if __name__ == "__main__":
    r = HybridParentWindowRetriever(window=1)
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}  (window: {hit['window_chunk_ids']})")
        print(f"  {hit['text'][:100]}...")
