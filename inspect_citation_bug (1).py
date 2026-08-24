"""
Live diagnostic: run one of the zero-citation-accuracy queries through the
real parent_window pipeline and print exactly what chunk_ids were retrieved
vs what the model actually cited, to find the mismatch.
"""
from parent_retrieve import ParentWindowRetriever
from generate import generate_answer

retriever = ParentWindowRetriever(window=1)
query = "What is the Humber Grant?"

chunks = retriever.search(query, top_k=4)
print("=== Retrieved chunk_ids (available for citation) ===")
for c in chunks:
    print(f"  {c['chunk_id']}  (score={c['score']:.3f})")
    print(f"    original chunk text: {c.get('original_chunk_text', '')[:80]}")
    print(f"    windowed text sent to model: {c['text'][:150]}")
    print()

result = generate_answer(query, chunks)
print("=== Model's answer ===")
print(result["answer"])
print()
print("=== Model's citations (raw, as returned) ===")
print(result["citations"])
print()

available_ids = {c["chunk_id"] for c in chunks}
print("=== Match check ===")
for cid in result["citations"]:
    match = cid in available_ids
    print(f"  '{cid}' -> {'MATCH' if match else 'NO MATCH (this is the bug)'}")
