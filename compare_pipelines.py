"""
Runs the SAME query set through three retrieval strategies and reports
comparable metrics for each — this is the core "compare different RAG
approaches" deliverable.

Strategies compared:
  - dense   : embedding similarity only (retrieve.py)
  - sparse  : BM25 keyword matching only (bm25_retrieve.py)
  - hybrid  : weighted fusion of both (hybrid_retrieve.py)

Metrics reported per strategy:
  - retrieval_hit_rate : did the retrieved set contain at least one gold
                          chunk_id? This is the metric that most directly
                          measures "is retrieval finding the right thing,"
                          independent of what the generator then does with it.
  - mrr                 : Mean Reciprocal Rank — rewards the gold chunk
                          appearing EARLY in the ranked results, not just
                          present somewhere in top_k.
  - citation_accuracy, faithfulness, answer_relevancy : same heuristics as
    evaluate.py, run per-strategy so you can see whether better retrieval
    actually produces better generation, not just better ranking.

Usage:
    python compare_pipelines.py
Requires: qa_dataset_template.csv filled in with real gold_chunk_ids
          (see HOW_TO_FILL_QA_DATASET.md), Ollama running, index built.
"""

import csv
import json
import os

from retrieve import Retriever
from bm25_retrieve import BM25Retriever
from hybrid_retrieve import HybridRetriever
from hyde_retrieve import HyDERetriever
from rerank_retrieve import RerankRetriever
from rrf_retrieve import RRFRetriever
from multiquery_retrieve import MultiQueryRetriever
from parent_retrieve import ParentWindowRetriever
from generate import generate_answer
from evaluate import citation_accuracy, faithfulness, answer_relevancy

QA_PATH = os.path.join(os.path.dirname(__file__), "qa_dataset_template.csv")


def load_qa_set(path: str):
    rows = list(csv.DictReader(open(path)))
    for row in rows:
        row["gold_chunk_ids"] = [c.strip() for c in row["gold_chunk_ids"].split(";") if c.strip()]
    return rows


def retrieval_hit_rate(retrieved_chunks, gold_ids) -> float:
    if not gold_ids:
        return None  # no ground truth to check against
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    return 1.0 if retrieved_ids & set(gold_ids) else 0.0


def mean_reciprocal_rank(retrieved_chunks, gold_ids) -> float:
    if not gold_ids:
        return None
    for rank, c in enumerate(retrieved_chunks, start=1):
        if c["chunk_id"] in gold_ids:
            return 1.0 / rank
    return 0.0


def run_strategy(name: str, retriever, qa_rows, top_k: int = 4):
    results = []
    for i, row in enumerate(qa_rows, 1):
        query = row["query"]
        print(f"  [{name}] {i}/{len(qa_rows)}: {query[:60]}", flush=True)
        try:
            chunks = retriever.search(query, top_k=top_k)
            generated = generate_answer(query, chunks)
        except Exception as e:
            # A single bad query (retrieval error, unexpected exception) should
            # not lose the results already computed for the other 48 questions.
            print(f"    WARNING: failed ({e}) - recording as failure and continuing")
            results.append({
                "query": query,
                "hit_rate": None,
                "mrr": None,
                "citation_accuracy": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "error": str(e),
            })
            continue

        results.append({
            "query": query,
            "hit_rate": retrieval_hit_rate(chunks, row["gold_chunk_ids"]),
            "mrr": mean_reciprocal_rank(chunks, row["gold_chunk_ids"]),
            "citation_accuracy": citation_accuracy(
                generated["answer"], generated["citations"], chunks
            ),
            "faithfulness": faithfulness(generated["answer"], chunks),
            "answer_relevancy": answer_relevancy(query, generated["answer"]),
        })

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "strategy": name,
        "n_queries": len(results),
        "hit_rate": avg("hit_rate"),
        "mrr": avg("mrr"),
        "citation_accuracy": avg("citation_accuracy"),
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "per_query": results,
    }


def main():
    qa_rows = load_qa_set(QA_PATH)
    if any("FILL IN" in r["gold_answer"] for r in qa_rows):
        print("WARNING: qa_dataset_template.csv still has unfilled placeholders.")
        print("Results below will be unreliable until you complete it — see HOW_TO_FILL_QA_DATASET.md\n")

    strategies = {
        "dense": Retriever(),
        "sparse": BM25Retriever(),
        "hybrid": HybridRetriever(),
        "hyde": HyDERetriever(),          # extra LLM call per query (hypothetical answer)
        "rerank": RerankRetriever(),      # extra LLM calls per candidate chunk - slowest
        "rrf": RRFRetriever(),            # no extra LLM calls - just different fusion math
        "multiquery": MultiQueryRetriever(),  # extra LLM call per query (query reformulations)
        "parent_window": ParentWindowRetriever(window=1),  # no extra LLM calls - just wider context
    }

    summary = []
    for name, retriever in strategies.items():
        print(f"Running strategy: {name}...")
        summary.append(run_strategy(name, retriever, qa_rows))
        # Save after every strategy, not just at the end - a disconnect during
        # a later strategy (rerank is slowest) shouldn't lose earlier results.
        with open("comparison_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> progress saved to comparison_results.json ({len(summary)}/{len(strategies)} strategies done)\n")

    print(f"\n{'Strategy':<10} {'HitRate':>8} {'MRR':>8} {'CiteAcc':>8} {'Faith':>8} {'Relev':>8}")
    for s in summary:
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{s['strategy']:<10} {fmt(s['hit_rate']):>8} {fmt(s['mrr']):>8} "
              f"{fmt(s['citation_accuracy']):>8} {fmt(s['faithfulness']):>8} {fmt(s['answer_relevancy']):>8}")

    with open("comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFull per-query results -> comparison_results.json")
    print("\nFor your dissertation: report the summary table above directly, and")
    print("discuss any query where strategies disagree (per_query in the JSON) as a case study.")


if __name__ == "__main__":
    main()
