"""
Diagnose why parent_window's citation accuracy dropped after the corpus
expansion - finds its worst-scoring queries and shows exactly what happened.
"""
import json

summary = json.load(open("comparison_results.json"))

parent_window = next(s for s in summary if s["strategy"] == "parent_window")
results = parent_window["per_query"]

# Sort by citation_accuracy ascending - worst first
worst = sorted(results, key=lambda r: r["citation_accuracy"])[:8]

print("=== Worst 8 citation_accuracy cases for parent_window ===\n")
for r in worst:
    print(f"Query: {r['query']}")
    print(f"  citation_accuracy: {r['citation_accuracy']:.3f}")
    print(f"  faithfulness:      {r['faithfulness']:.3f}")
    print(f"  hit_rate:          {r['hit_rate']}")
    print()

# Also show the overall distribution to see how many are near-zero vs middling
scores = [r["citation_accuracy"] for r in results]
zero_count = sum(1 for s in scores if s < 0.1)
print(f"Total queries: {len(scores)}")
print(f"Near-zero citation_accuracy (<0.1): {zero_count}")
print(f"Mean: {sum(scores)/len(scores):.3f}")
