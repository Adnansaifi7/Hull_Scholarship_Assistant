"""
Component A, step 3: embed every chunk and store a flat vector index.

Uses Ollama's /api/embeddings endpoint (works with e.g. `nomic-embed-text`,
which you should `ollama pull nomic-embed-text` before running this).
Stored as plain numpy + JSON metadata — no vector DB needed at dissertation
scale (a few hundred to few thousand chunks). If your corpus grows much
larger, swap this file's save/load for a FAISS index; the retrieval API in
retrieve.py is written so that swap wouldn't touch calling code.
"""

import json
import os
import numpy as np
import requests

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "chunks.jsonl")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "index")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def main():
    if not os.path.exists(CHUNKS_PATH):
        raise SystemExit(f"No chunks found at {CHUNKS_PATH}. Run chunk_corpus.py first.")

    chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
    if not chunks:
        raise SystemExit("chunks.jsonl is empty.")

    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embed_text(chunk["text"])
        vectors.append(vec)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"embedded {i + 1}/{len(chunks)}")

    matrix = np.vstack(vectors)
    # L2-normalise once so retrieval can use a plain dot product as cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    matrix = matrix / norms

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), matrix)
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump(chunks, f)

    print(f"\nIndex built: {matrix.shape[0]} vectors x {matrix.shape[1]} dims")
    print(f"Saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
