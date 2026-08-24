"""
Parent (window) retrieval.

Idea: small chunks are good for precise MATCHING (a focused chunk about
"eligibility" scores higher against an eligibility question than a chunk
mixing eligibility + deadlines + appeals). But small chunks can be too
narrow for GENERATION - if the real answer spans a sentence that got cut
at a chunk boundary, the generator never sees the full picture.

This retriever keeps precise dense-retrieval RANKING (search still uses the
small chunks), but for each retrieved chunk, also pulls in its immediate
neighbours (previous and next chunk_index from the same source_file) before
handing everything to the generator - giving it more surrounding context
without changing what got matched in the first place.

Reference: this is the "parent document retriever" / "sentence window
retrieval" pattern used in e.g. LlamaIndex - retrieve small, generate with
more context around it.
"""

import json
import os
from typing import List, Optional

from retrieve import Retriever

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "chunks.jsonl")


class ParentWindowRetriever:
    def __init__(self, window: int = 1):
        self.base = Retriever()
        self.window = window  # how many neighbouring chunks to include on each side

        # Index all chunks by (source_file, chunk_index) so neighbours can be
        # looked up directly rather than re-scanning the whole corpus each time.
        self.by_source_index = {}
        # Retriever.search() hits don't include chunk_index (only chunk_id),
        # so also index full metadata by chunk_id to recover it reliably.
        self.by_chunk_id = {}
        for c in self.base.metadata:
            key = (c["source_file"], c["chunk_index"])
            self.by_source_index[key] = c
            self.by_chunk_id[c["chunk_id"]] = c

    def _get_window_text_and_ids(self, chunk: dict):
        # chunk here is a search HIT (from Retriever.search()), which only
        # has chunk_id/source_file/text/score - NOT chunk_index directly.
        # Look up the full metadata entry by chunk_id to get the real index.
        full_meta = self.by_chunk_id.get(chunk["chunk_id"])
        if full_meta is None:
            return chunk["text"], [chunk["chunk_id"]]  # defensive fallback
        source = full_meta["source_file"]
        idx = full_meta["chunk_index"]
        parts = []
        ids = []
        for offset in range(-self.window, self.window + 1):
            neighbour = self.by_source_index.get((source, idx + offset))
            if neighbour:
                parts.append(neighbour["text"])
                ids.append(neighbour["chunk_id"])
        return " ".join(parts), ids

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        # Ranking is identical to plain dense retrieval - only the TEXT
        # handed to the generator changes.
        hits = self.base.search(query, top_k=top_k, doc_type_filter=doc_type_filter)

        results = []
        for hit in hits:
            expanded = dict(hit)
            expanded["original_chunk_text"] = hit["text"]  # kept for inspection/debugging
            window_text, window_ids = self._get_window_text_and_ids(hit)
            expanded["text"] = window_text
            expanded["window_chunk_ids"] = window_ids  # every real ID the model may legitimately cite
            results.append(expanded)
        return results


if __name__ == "__main__":
    r = ParentWindowRetriever(window=1)
    for hit in r.search("Am I eligible for the Chancellor's Scholarship?"):
        print(f"[{hit['score']:.3f}] {hit['chunk_id']}")
        print(f"  original: {hit['original_chunk_text'][:60]}...")
        print(f"  windowed: {hit['text'][:100]}...")
