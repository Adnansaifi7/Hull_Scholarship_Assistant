"""
Sparse retrieval baseline: BM25, implemented from scratch (no external
dependency) so it's easy to explain and cite exactly in your methodology
section, rather than treating a library as a black box.

BM25 scores a chunk by term frequency (with saturation) and inverse document
frequency (rarer terms count more), which is the classic pre-embedding IR
approach — a natural "baseline" to compare your dense (embedding-based)
retriever against. If dense retrieval doesn't clearly beat BM25 on your
eval set, that's a genuinely interesting dissertation finding, not a
failure — report it either way.

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance
Framework: BM25 and Beyond".
"""

import json
import math
import os
import re
from collections import Counter
from typing import List, Optional

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "chunks.jsonl")

# Standard BM25 hyperparameters — k1 controls term-frequency saturation,
# b controls how much document length is penalised. These are the widely
# used defaults (Robertson & Zaragoza); worth noting in your dissertation
# that you didn't tune them, which is itself a limitation to mention.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class BM25Retriever:
    def __init__(self, chunks_path: str = CHUNKS_PATH):
        self.metadata = [json.loads(line) for line in open(chunks_path)]
        self.tokenized_docs = [tokenize(c["text"]) for c in self.metadata]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        self.doc_freqs = []  # term frequency counter per doc
        df = Counter()  # document frequency per term (how many docs contain it)
        for doc in self.tokenized_docs:
            counts = Counter(doc)
            self.doc_freqs.append(counts)
            for term in counts:
                df[term] += 1

        n_docs = len(self.tokenized_docs)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _score(self, query_terms: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        freqs = self.doc_freqs[doc_idx]
        for term in query_terms:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            f = freqs[term]
            numerator = f * (K1 + 1)
            denominator = f + K1 * (1 - B + B * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        return score

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        query_terms = tokenize(query)
        candidate_idx = range(len(self.metadata))
        if doc_type_filter:
            filtered = [i for i in candidate_idx if self.metadata[i]["doc_type"] in doc_type_filter]
            if filtered:
                candidate_idx = filtered

        scored = [(i, self._score(query_terms, i)) for i in candidate_idx]
        scored.sort(key=lambda x: -x[1])
        top = scored[:top_k]

        results = []
        for idx, score in top:
            m = self.metadata[idx]
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
    r = BM25Retriever()
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}: {hit['text'][:80]}")
