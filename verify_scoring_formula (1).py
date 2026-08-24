"""
Take the exact answer/citations we just saw for "What is the Humber Grant?"
and manually run it through evaluate.py's real citation_accuracy() function
to check if the SCORING FORMULA itself is the bug, not just retrieval luck.
"""
from parent_retrieve import ParentWindowRetriever
from generate import generate_answer
from evaluate import citation_accuracy, content_words

retriever = ParentWindowRetriever(window=1)
query = "What is the Humber Grant?"
chunks = retriever.search(query, top_k=4)
result = generate_answer(query, chunks)

print("Answer:", result["answer"])
print("Citations:", result["citations"])
print()

score = citation_accuracy(result["answer"], result["citations"], chunks)
print(f"citation_accuracy score: {score:.3f}")
print()

# Show the overlap math for each citation individually
answer_words = content_words(result["answer"])
print(f"Answer content words ({len(answer_words)}): {answer_words}")
print()

chunk_lookup = {c["chunk_id"]: c["text"] for c in chunks}
for cid in result["citations"]:
    chunk_text = chunk_lookup.get(cid)
    if chunk_text is None:
        print(f"'{cid}': NOT FOUND in retrieved chunks")
        continue
    chunk_words = content_words(chunk_text)
    overlap = chunk_words & answer_words
    fraction = len(overlap) / max(len(answer_words), 1)
    print(f"'{cid}':")
    print(f"  overlap words: {overlap}")
    print(f"  overlap fraction: {fraction:.3f}  (needs > 0.15 to count as correct)")
    print()
