"""
Component A, step 4: retrieval.

Loads the vector index built by build_index.py and returns the top-k most
relevant chunks for a query, optionally restricted to a set of dataset tags
(this is where Component B's intent classifier output plugs in — see
rag_pipeline.py).
"""

import json
import os
from typing import List, Optional

import numpy as np

from build_index import embed_text  # reuse the same embedding call

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "index")


class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR):
        vec_path = os.path.join(index_dir, "vectors.npy")
        meta_path = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(
                f"No index at {index_dir}. Run build_index.py first."
            )
        self.vectors = np.load(vec_path)
        self.metadata = json.load(open(meta_path))

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        q_vec = embed_text(query)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        scores = self.vectors @ q_vec  # cosine similarity (both L2-normalised)

        # Apply doc_type filter (from Component B routing) before ranking
        candidate_idx = np.arange(len(self.metadata))
        if doc_type_filter:
            candidate_idx = np.array([
                i for i in candidate_idx
                if self.metadata[i]["doc_type"] in doc_type_filter
            ])
            if len(candidate_idx) == 0:
                # Filter matched nothing — fall back to unfiltered search
                candidate_idx = np.arange(len(self.metadata))

        candidate_scores = scores[candidate_idx]
        order = np.argsort(-candidate_scores)[:top_k]
        top_idx = candidate_idx[order]

        results = []
        for idx in top_idx:
            m = self.metadata[idx]
            results.append({
                "chunk_id": m["chunk_id"],
                "source_file": m["source_file"],
                "doc_type": m["doc_type"],
                "authority": m["authority"],
                "text": m["text"],
                "score": float(scores[idx]),
            })
        return results


if __name__ == "__main__":
    # Quick manual smoke test — requires Ollama running with the embed model pulled
    r = Retriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}: {hit['text'][:100]}...")
