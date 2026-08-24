"""
Component A, step 5: generation.

Builds a grounded prompt from retrieved chunks and calls a small local model
via Ollama's /api/generate, asking for strict structured JSON output:
{ "answer": ..., "citations": [chunk_id, ...], "confidence": "high|medium|low" }

Forcing structured output with explicit chunk_id citations is what lets you
measure "citation accuracy" in evaluate.py — you can check whether the cited
chunk actually contains the claim, rather than trusting the model's prose.
"""

import json
import os
import re
from typing import List

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "phi3")

SYSTEM_PROMPT = """You are a university assistant answering questions about \
scholarships and finance using ONLY the provided context. Rules:
1. Answer only using facts present in the context below. Do not use outside knowledge.
2. If the context does not contain the answer, say so explicitly rather than guessing.
3. Every factual claim must be traceable to a specific context chunk.
4. Respond with ONLY a JSON object, no other text, in this exact schema:
{"answer": "<your answer>", "citations": ["<chunk_id>", ...], "confidence": "high|medium|low"}
"""


def build_context_block(chunks: List[dict]) -> str:
    parts = []
    for c in chunks:
        # window_chunk_ids (set by ParentWindowRetriever) lists every real
        # chunk_id blended into this text block. Fall back to just chunk_id
        # for retrievers that don't expand windows (dense, sparse, etc.) -
        # only one real ID exists in that case anyway.
        ids = c.get("window_chunk_ids", [c["chunk_id"]])
        ids_label = ", ".join(ids)
        parts.append(f"[valid chunk_ids for this passage: {ids_label}] (source: {c['source_file']})\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, retrieved_chunks: List[dict], _retry: bool = True) -> dict:
    context_block = build_context_block(retrieved_chunks)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"JSON RESPONSE:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        if _retry:
            # Under sustained batch load a single request can stall - one retry
            # is usually enough rather than aborting the whole comparison run.
            return generate_answer(query, retrieved_chunks, _retry=False)
        return {
            "answer": f"[generation failed: {e}]",
            "citations": [],
            "confidence": "low",
            "parse_error": True,
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        }

    raw_output = resp.json()["response"].strip()

    parsed = parse_model_json(raw_output)
    parsed["retrieved_chunk_ids"] = [c["chunk_id"] for c in retrieved_chunks]
    return parsed


def parse_model_json(raw_output: str) -> dict:
    """SLMs frequently wrap JSON in prose or code fences despite instructions.
    This defensively extracts the first {...} block rather than assuming
    clean output — expect to harden this further once you see real failures
    from your chosen model, and report the failure rate in your dissertation.
    """
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {
            "answer": raw_output,
            "citations": [],
            "confidence": "low",
            "parse_error": True,
        }
    try:
        parsed = json.loads(match.group(0))
        parsed.setdefault("citations", [])
        # Models sometimes copy the "chunk_id: " label from the prompt's
        # context formatting into the citation string itself. Strip it so
        # citation matching (evaluate.py, compare_pipelines.py) works
        # against the real chunk_id rather than silently failing to match.
        parsed["citations"] = [
            c.split("chunk_id:", 1)[-1].strip() if "chunk_id:" in c else c.strip()
            for c in parsed["citations"]
        ]
        parsed.setdefault("confidence", "low")
        parsed["parse_error"] = False
        return parsed
    except json.JSONDecodeError:
        return {
            "answer": raw_output,
            "citations": [],
            "confidence": "low",
            "parse_error": True,
        }
