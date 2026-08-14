# KnowledgeAI

## Overview

KnowledgeAI is a document assistant built around retrieval-augmented generation.
Users upload PDFs and ask questions in natural language; answers are generated
strictly from passages retrieved out of those documents, with citations pointing
back to the source page.

Everything runs locally against Ollama, so no document content leaves the machine.

## Why it exists

The project's focus is **retrieval quality, measured rather than assumed**. Each
retrieval decision was benchmarked on BEIR NFCorpus before shipping, including
the ones that failed — a cross-encoder reranker was implemented, measured, found
to make results 8.4% worse, and left disabled with the evidence recorded.

Full results and methodology are in the [README](README.md).

## Target users

Businesses, students, researchers, legal firms, and healthcare organisations —
anyone who needs answers from their own documents with sources they can verify.

## What is built

- PDF upload with signature-based validation and size limits
- Page-aware text extraction, normalization, and overlapping chunking
- Embedding via Ollama, stored in ChromaDB with citation metadata
- **Hybrid retrieval**: dense embeddings fused with BM25 by reciprocal rank fusion
- Cross-encoder reranking — implemented, benchmarked, disabled on the evidence
- Grounded answer generation with model-emitted citations validated against the
  passages actually retrieved
- A web console: document scoping, conversation history, and a chunk inspector
- A BEIR evaluation harness, plus 38 tests that run with no services and no network

## What is not built

These are deliberately out of scope so far, not oversights:

- User authentication and per-user document isolation
- Server-side persistence — documents and conversations are held in the browser,
  and PostgreSQL is provisioned in Docker but unused
- Streaming responses and background indexing with progress
- DOCX support and OCR for scanned PDFs

## Technology

FastAPI · ChromaDB · Ollama (`qwen2.5:7b`, `nomic-embed-text`) · pypdf · pytest · Docker

The front end is a single self-contained page served by FastAPI. The embedding
model, chat model, vector store, and reranker each sit behind a small adapter, so
any of them can be replaced without touching the retrieval logic.
