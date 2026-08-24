"""
Evaluation harness for faithfulness, citation accuracy, and answer relevancy.

These are lightweight, explainable heuristic metrics you can run without
extra infra, and each is designed to be defensible in a dissertation
methodology section:

- citation_accuracy: for each cited chunk_id, does the chunk's actual text
  lexically support the answer (word-overlap check)? This directly measures
  whether the model is citing chunks that don't actually back its claim.
- faithfulness: fraction of content words in the answer that also appear
  somewhere in the retrieved context. A crude proxy for "did the model stick
  to the source" — cite this limitation explicitly and consider upgrading to
  an NLI-based or RAGAS faithfulness score if you have time budget for it.
- answer_relevancy: cosine similarity between the query embedding and the
  answer embedding (via the same Ollama embedding model used for retrieval).

Run against a labelled eval set you build yourself (a CSV of
query, expected_chunk_ids, expected_answer_keywords) — a template is
provided in eval_set_template.csv.
"""

import csv
import json
import os
import re
from typing import List

import numpy as np

from build_index import embed_text
from rag_pipeline import HullScholarshipRAG

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "at", "with", "as", "by", "this", "that", "it", "be",
}


def content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def citation_accuracy(answer: str, citations: List[str], retrieved_chunks: List[dict]) -> float:
    if not citations:
        return 0.0
    # Map EVERY id in each chunk's window (not just the anchor chunk_id) to
    # that block's text - parent_window retrieval blends several physical
    # chunks together, and the model may correctly cite any real id within
    # that blended text, not only whichever one happened to be the anchor.
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
            continue  # cited a chunk_id that wasn't even retrieved — clear failure
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


def answer_relevancy(query: str, answer: str) -> float:
    q_vec = embed_text(query)
    a_vec = embed_text(answer)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    a_vec = a_vec / (np.linalg.norm(a_vec) + 1e-8)
    return float(q_vec @ a_vec)


def run_eval(eval_csv_path: str):
    pipeline = HullScholarshipRAG()
    rows = list(csv.DictReader(open(eval_csv_path)))

    results = []
    for row in rows:
        query = row["query"]
        out = pipeline.answer(query)

        results.append({
            "query": query,
            "intent": out["intent"],
            "answer": out["answer"],
            "citation_accuracy": citation_accuracy(
                out["answer"], out["citations"], out["retrieved_chunks"]
            ),
            "faithfulness": faithfulness(out["answer"], out["retrieved_chunks"]),
            "answer_relevancy": answer_relevancy(query, out["answer"]),
            "needs_human_review": out["needs_human_review"],
        })

    print(f"{'query':<55} {'citAcc':>7} {'faith':>7} {'relev':>7} {'review?':>8}")
    for r in results:
        print(f"{r['query'][:53]:<55} {r['citation_accuracy']:.2f}   "
              f"{r['faithfulness']:.2f}   {r['answer_relevancy']:.2f}   "
              f"{str(r['needs_human_review']):>8}")

    avg = lambda key: sum(r[key] for r in results) / len(results)
    print("\n--- Aggregate ---")
    print(f"Mean citation accuracy: {avg('citation_accuracy'):.3f}")
    print(f"Mean faithfulness:      {avg('faithfulness'):.3f}")
    print(f"Mean answer relevancy:  {avg('answer_relevancy'):.3f}")
    print(f"Flagged for review:     {sum(r['needs_human_review'] for r in results)}/{len(results)}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results -> eval_results.json")


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(__file__), "eval_set_template.csv")
    run_eval(default_path)
