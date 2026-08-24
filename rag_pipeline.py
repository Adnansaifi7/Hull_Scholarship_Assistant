"""
The integration point your dissertation is really about: Component B's
classifier output changes how Component A retrieves and how much you trust
the answer, rather than sitting next to the RAG pipeline unused.

Flow:
  query -> [Component B: intent classifier] -> retrieval_tags, confidence
        -> [Component A: retriever]         -> top-k chunks (filtered by tag)
        -> [Component A: generator]         -> grounded JSON answer
        -> low_confidence routing           -> flagged for human review if
                                                either the classifier OR the
                                                generator was unsure
"""

import os
import sys

import joblib

from parent_retrieve import ParentWindowRetriever
from generate import generate_answer

SKLEARN_SERVICE_DIR = os.path.join(os.path.dirname(__file__), "..", "sklearn-service")
sys.path.insert(0, SKLEARN_SERVICE_DIR)

INTENT_TO_DOC_TYPES = {
    "eligibility": ["formal_policy", "faq"],
    "deadline": ["formal_policy", "faq"],
    "amount": ["formal_policy", "support_faq"],
    "appeal": ["formal_policy"],
    "process": ["faq", "landing_page"],
    "hardship": ["support_faq"],
}

CLASSIFIER_CONFIDENCE_THRESHOLD = 0.5

# Below this, the classifier's prediction is barely better than random guessing
# across 6 classes (~0.167 baseline) and should not be trusted enough to
# FILTER retrieval - doing so risks excluding the actually-correct chunk
# entirely, as opposed to just ranking it lower. Discovered via a real case:
# "What is student finance?" was classified as "hardship" at 19.9% confidence,
# which filtered out the correct faq-tagged chunk (ranked #1 unfiltered at
# 0.839 similarity) in favour of only support_faq chunks.
ROUTING_CONFIDENCE_THRESHOLD = 0.3


class HullScholarshipRAG:
    def __init__(self):
        # ParentWindowRetriever chosen based on the 8-strategy comparison
        # (compare_pipelines.py): it matched dense retrieval's best-in-class
        # hit rate/MRR while winning on citation accuracy, faithfulness, and
        # relevancy, by giving the generator neighbouring-chunk context.
        self.retriever = ParentWindowRetriever(window=1)
        model_path = os.path.join(SKLEARN_SERVICE_DIR, "intent_classifier.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained classifier at {model_path}. "
                "Run `python train_intent_classifier.py` inside sklearn-service/ first."
            )
        self.classifier = joblib.load(model_path)

    def answer(self, query: str, top_k: int = 4) -> dict:
        # --- Component B ---
        intent = self.classifier.predict([query])[0]
        proba = self.classifier.predict_proba([query])[0]
        intent_confidence = float(max(proba))

        # Only trust the classifier's routing when it's meaningfully confident.
        # A low-confidence prediction filtering retrieval does more harm than
        # good - better to search the full corpus than wrongly narrow it.
        if intent_confidence >= ROUTING_CONFIDENCE_THRESHOLD:
            doc_type_filter = INTENT_TO_DOC_TYPES.get(intent)
        else:
            doc_type_filter = None

        # --- Component A: retrieval (routed by Component B) ---
        chunks = self.retriever.search(query, top_k=top_k, doc_type_filter=doc_type_filter)

        # --- Component A: generation ---
        result = generate_answer(query, chunks)

        # --- Combined confidence / human-review flag ---
        classifier_uncertain = intent_confidence < CLASSIFIER_CONFIDENCE_THRESHOLD
        generator_uncertain = result.get("confidence") in ("low",) or result.get("parse_error")
        needs_human_review = classifier_uncertain or generator_uncertain

        return {
            "query": query,
            "intent": intent,
            "intent_confidence": round(intent_confidence, 3),
            "answer": result["answer"],
            "citations": result["citations"],
            "generation_confidence": result["confidence"],
            "needs_human_review": needs_human_review,
            "retrieved_chunks": chunks,
        }


if __name__ == "__main__":
    # Manual smoke test — requires Ollama running + index built + classifier trained
    pipeline = HullScholarshipRAG()
    out = pipeline.answer("Am I eligible for the Chancellor's Scholarship?")
    import json
    print(json.dumps(out, indent=2))
