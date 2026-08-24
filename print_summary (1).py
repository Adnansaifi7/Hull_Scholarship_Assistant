import json

summary = json.load(open("comparison_results.json"))

def avg(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None

print(f"{'Strategy':<10} {'HitRate':>8} {'MRR':>8} {'CiteAcc':>8} {'Faith':>8} {'Relev':>8}")
for s in summary:
    results = s["per_query"]
    hit_rate = avg(results, "hit_rate")
    mrr = avg(results, "mrr")
    cite = avg(results, "citation_accuracy")
    faith = avg(results, "faithfulness")
    relev = avg(results, "answer_relevancy")
    fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
    print(f"{s['strategy']:<10} {fmt(hit_rate):>8} {fmt(mrr):>8} {fmt(cite):>8} {fmt(faith):>8} {fmt(relev):>8}")
