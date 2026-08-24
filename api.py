"""
HTTP entrypoint for the whole integrated pipeline. Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"query": "Am I eligible for the Chancellor Scholarship?"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import HullScholarshipRAG

app = FastAPI(title="Hull Scholarships & Finance RAG Assistant")
pipeline = None  # lazy-loaded so the API can start even before index/model exist

# Allows the webpage (index.html, opened directly as a local file or served
# separately) to call this API across origins - needed because the browser
# treats file:// or a different port as a different origin than localhost:8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    top_k: int = 4


@app.on_event("startup")
def load_pipeline():
    global pipeline
    try:
        pipeline = HullScholarshipRAG()
    except FileNotFoundError as e:
        # Don't crash the server if setup steps haven't been run yet —
        # surface a clear error on first request instead.
        print(f"WARNING: pipeline not ready — {e}")


@app.post("/ask")
def ask(req: AskRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Run chunk_corpus.py, build_index.py, "
                   "and train_intent_classifier.py first, then restart the API.",
        )
    return pipeline.answer(req.query, top_k=req.top_k)


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}
