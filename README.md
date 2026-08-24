# Grounded AI for University Workflows

**A retrieval-augmented small language model assistant for university scholarship and finance queries, with classical ML intent routing — MSc research project, University of Hull.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

---

## Overview

University scholarship and finance information is accurate but institutionally fragmented — scattered across policy PDFs, FAQ pages, and support pages that rarely use the words a student would. This project builds and rigorously evaluates a two-component system to answer real scholarship and finance queries:

- **Component A** — a retrieval-augmented generation (RAG) pipeline grounding a small language model's answers in retrieved document evidence, with citations to source text.
- **Component B** — a classical ML intent classifier (TF-IDF + logistic regression) routing queries to the appropriate document subset, gated on prediction confidence.

Rather than adopting a single retrieval technique by default, this project implements and compares **eleven retrieval strategies** under an identical evaluation harness, validated with a proper tuning/held-out split, grid search, and paired statistical significance testing — reporting a genuine null result honestly where the evidence didn't support a claim, rather than only publishing the flattering number.

## Key Results

| | |
|---|---|
| **Best validated citation accuracy** | 0.857 (SD 0.012, 3 trials) |
| **Classifier improvement** | macro-F1 0.476 → 0.745 (56% relative) |
| **Corpus recovered via bug fix** | +114 documents (8 → 122) |
| **System defects found & fixed** | 2, via targeted adversarial testing |
| **Retrieval strategies compared** | 11, under one evaluation harness |

Full methodology, all 11 strategies' results, and the statistical validation are in [`report/`](./report).

## Architecture

```
Query → Component B (Intent Classifier) → Component A (Retrieval) → Generation → Cited Answer
                    │                              │
              confidence-gated                parent-window
                routing                        expansion
```

An incoming query is classified into one of six intent categories; the predicted intent and confidence score determine whether retrieval searches the full corpus or a document-type-filtered subset. Retrieved chunks are expanded to include neighbouring context (parent-window retrieval, the strongest single strategy evaluated), then passed to the generation model with instructions to answer only from provided context and cite specific source chunks.

## What Was Actually Tested

Eight baseline retrieval strategies plus three combined variants:

`Dense` · `Sparse (BM25)` · `Hybrid` · `HyDE` · `LLM Rerank` · `RRF` · `Multi-query` · `Parent-window` · `Hybrid+Parent-window (3 configurations)`

Evaluated on a 50-question, manually verified ground-truth set across five metrics (hit rate, MRR, citation accuracy, faithfulness, answer relevancy), then validated with:
- A 35/14-question tuning/held-out split
- A 9-point grid search over retrieval hyperparameters
- A paired t-test on the held-out result (reported honestly as **not statistically significant**, p=0.618, given the sample size)
- A separate, statistically validated generation-model comparison (phi3 vs. Llama 3.1) across 3 independent trials

## Two Real Defects Found and Fixed

Not found by watching aggregate metrics — found by deliberately testing individual adversarial queries:

1. **Confidence-gated routing defect** — a 19.9%-confidence classifier prediction was silently excluding the correct retrieval candidate. Fixed by only applying routing filters above a 0.3 confidence threshold.
2. **Citation window-matching defect** — a correctly-cited neighbouring chunk was being scored as hallucinated due to a labelling gap in expanded context windows. Fixed by exposing every real chunk ID to the scoring function; citation accuracy improved from 0.667 to 0.718 as a direct result.

## Tech Stack

- **Generation:** phi3 (3.8B) / Llama 3.1 (8B) via Ollama
- **Embeddings:** nomic-embed-text
- **Classifier:** scikit-learn (TF-IDF + logistic regression)
- **API:** FastAPI
- **Infrastructure:** University of Hull Viper HPC cluster, NVIDIA A40 GPU, Apptainer containerisation
- **Evaluation:** custom harness (hit rate, MRR, citation accuracy, faithfulness, answer relevancy)

## Repository Structure

```
├── rag-service/          # Component A: retrieval strategies, generation, evaluation harness
├── sklearn-service/       # Component B: intent classifier
├── webapp/                 # Browser-based demo interface
├── notebook/                # Full evaluation notebook (all 11 strategies, grid search, significance tests)
├── report/                    # Full written report (methodology, results, discussion)
└── README.md
```

## Running This Yourself

This project was developed and evaluated on a university HPC cluster with GPU access via Ollama. To reproduce:

1. Install [Ollama](https://ollama.com) and pull `phi3` and `nomic-embed-text`
2. `pip install -r requirements.txt`
3. Build the corpus index: `python rag-service/build_index.py`
4. Train the classifier: `python sklearn-service/train_intent_classifier.py`
5. Run the evaluation harness: see `notebook/` for the full, cell-by-cell evaluation pipeline

**Note:** the original corpus (scraped University of Hull public documents) is not included in this repository — only the code that builds it. This avoids republishing third-party institutional content while keeping the methodology fully reproducible against any similar document set.

## Limitations

Stated explicitly rather than glossed over: evaluation metrics are lexical-overlap heuristics rather than semantic judgement; the 14-question held-out set limits statistical power; only one alternative generation model was tested; wall-clock latency and cost were not empirically benchmarked. Full discussion in the report.

## Author

**Mohd Adnan Saifi** — MSc Data Science, AI and Modelling, University of Hull
Supervised by Dr Temitayo Matthew Fagbola

---

*This is an academic research project, not an official University of Hull service.*
