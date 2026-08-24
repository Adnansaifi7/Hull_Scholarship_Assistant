"""
Reruns ONLY the parent_window strategy (the one that didn't finish before
the job died) and merges it into the existing comparison_results.json
rather than redoing all 8 strategies from scratch.
"""
import json

from parent_retrieve import ParentWindowRetriever
from compare_pipelines import load_qa_set, run_strategy, QA_PATH

print("Loading existing results...")
summary = json.load(open("comparison_results.json"))
existing_strategies = {s["strategy"] for s in summary}
print(f"Already have: {sorted(existing_strategies)}")

if "parent_window" in existing_strategies:
    print("parent_window already present - nothing to do.")
else:
    qa_rows = load_qa_set(QA_PATH)
    print(f"\nRunning parent_window on {len(qa_rows)} questions...")
    result = run_strategy("parent_window", ParentWindowRetriever(window=1), qa_rows)
    summary.append(result)

    with open("comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved. All 8 strategies now present.")

# Print the full final table
def avg(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None

print(f"\n{'Strategy':<12} {'HitRate':>8} {'MRR':>8} {'CiteAcc':>8} {'Faith':>8} {'Relev':>8}")
for s in summary:
    results = s["per_query"]
    hit_rate = avg(results, "hit_rate")
    mrr = avg(results, "mrr")
    cite = avg(results, "citation_accuracy")
    faith = avg(results, "faithfulness")
    relev = avg(results, "answer_relevancy")
    fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
    print(f"{s['strategy']:<12} {fmt(hit_rate):>8} {fmt(mrr):>8} {fmt(cite):>8} {fmt(faith):>8} {fmt(relev):>8}")
