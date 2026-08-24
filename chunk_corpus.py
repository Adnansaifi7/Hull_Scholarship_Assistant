"""
Component A, step 2: parse the raw corpus (HTML/PDF) into clean text chunks
with source metadata, ready for embedding.

Chunking strategy: paragraph-aware sliding window. This is a deliberate,
citable methodology choice for your dissertation — not just "Dify's default"
— because the corpus mixes dense policy PDFs (need smaller, precise chunks
so citations are tight) with marketing-style HTML pages (can tolerate larger
chunks). Justify chunk_size/overlap in your methodology chapter; the values
below are a reasonable starting point, not gospel.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List

from bs4 import BeautifulSoup
from pypdf import PdfReader

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "raw")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "chunks.jsonl")

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40

# The "authority level" flag matters for your evaluation: policy PDFs make
# binding claims (eligibility, deadlines, appeals) and should be weighted
# higher / cited more strictly than marketing HTML copy.
SOURCE_METADATA = {
    "ug_scholarships_awards.html": {"doc_type": "landing_page", "authority": "marketing"},
    "pgt_scholarships_awards.html": {"doc_type": "landing_page", "authority": "marketing"},
    "pgr_fees_funding.html": {"doc_type": "landing_page", "authority": "marketing"},
    "chancellors_scholarship_policy.pdf": {"doc_type": "formal_policy", "authority": "binding"},
    "fees_and_finance_hub.html": {"doc_type": "landing_page", "authority": "marketing"},
    "finance_information_and_policies.html": {"doc_type": "policy_index", "authority": "binding"},
    "financial_advice_and_support.html": {"doc_type": "support_faq", "authority": "operational"},
    "fees_funding_faqs.html": {"doc_type": "faq", "authority": "operational"},
    "policies_and_information_index.html": {"doc_type": "policy_index", "authority": "binding"},
}


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    doc_type: str
    authority: str
    text: str
    chunk_index: int


def extract_text_html(path: str) -> str:
    with open(path, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def extract_text_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = " ".join(pages)
    return re.sub(r"\s+", " ", text).strip()


def sliding_window_chunks(text: str, size: int, overlap: int) -> List[str]:
    words = text.split(" ")
    if len(words) <= size:
        return [text] if text else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if len(window) < 20:  # drop tiny trailing fragments
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def main():
    if not os.path.isdir(RAW_DIR):
        raise SystemExit(f"No corpus found at {RAW_DIR}. Run collect_corpus.py first.")

    all_chunks: List[Chunk] = []
    for filename in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, filename)
        meta = SOURCE_METADATA.get(filename)
        if meta is None:
            if filename.startswith("faq_"):
                # Individually-fetched FAQ answer sub-pages (see collect_faq_subpages.py)
                meta = {"doc_type": "faq", "authority": "operational"}
            else:
                meta = {"doc_type": "unknown", "authority": "unknown"}

        if filename.endswith(".pdf"):
            text = extract_text_pdf(path)
        elif filename.endswith(".html"):
            text = extract_text_html(path)
        else:
            continue

        pieces = sliding_window_chunks(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(
                chunk_id=f"{filename}::{i}",
                source_file=filename,
                doc_type=meta["doc_type"],
                authority=meta["authority"],
                text=piece,
                chunk_index=i,
            ))
        print(f"{filename}: {len(pieces)} chunks")

    with open(CHUNKS_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
