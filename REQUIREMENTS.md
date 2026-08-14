# Requirements Specification

Status is tracked per requirement. **Done** means implemented and covered by
tests; **Not built** means deliberately out of scope so far, not overlooked.

---

## Functional requirements

### Document management

| # | Requirement | Status |
|---|---|---|
| 1.1 | Upload PDF documents | **Done** |
| 1.2 | Reject unsupported file types | **Done** — validated by `%PDF-` signature, since filename and MIME type are user-controlled |
| 1.3 | Reject oversized uploads | **Done** — 25 MB limit |
| 1.4 | Reject encrypted and image-only PDFs | **Done** — returns 422 with a specific reason |
| 1.5 | List uploaded documents | **Partial** — the browser keeps the list; the API has no endpoint for it |
| 1.6 | Delete uploaded documents | **Not built** |

### Document processing

| # | Requirement | Status |
|---|---|---|
| 2.1 | Extract text while retaining page numbers | **Done** — page numbers are what make citations verifiable |
| 2.2 | Normalize extraction noise | **Done** — control characters stripped, paragraph breaks kept as chunk boundaries |
| 2.3 | Split text into overlapping chunks | **Done** — paragraph → sentence → token-window fallback |
| 2.4 | Generate embeddings | **Done** — batched through Ollama |
| 2.5 | Store vectors with citation metadata | **Done** — re-indexing removes prior vectors, so no stale chunks survive |

### Retrieval

| # | Requirement | Status |
|---|---|---|
| 3.1 | Retrieve semantically relevant passages | **Done** |
| 3.2 | Scope retrieval to a single document | **Done** |
| 3.3 | Measure retrieval quality against a public benchmark | **Done** — BEIR NFCorpus, nDCG@10 |
| 3.4 | Improve recall beyond dense-only search | **Done** — hybrid BM25 + RRF, 0.3399 → 0.3518 |
| 3.5 | Rerank retrieved passages | **Rejected on evidence** — measured 8.4% worse; kept behind `RERANKING_ENABLED=false` |
| 3.6 | Discard passages below a relevance threshold | **Deferred** — the benchmark-derived cutoff did not transfer to real PDFs |

### AI chat

| # | Requirement | Status |
|---|---|---|
| 4.1 | Answer questions about uploaded documents | **Done** |
| 4.2 | Answer only from retrieved passages | **Done** — enforced by the system prompt |
| 4.3 | Cite sources for factual claims | **Done** — citations the model emits are validated against the passages actually supplied, so it cannot cite a source that was never retrieved |
| 4.4 | Say so when the documents do not contain the answer | **Done** |
| 4.5 | Handle conversational messages without citing | **Done** |
| 4.6 | Stream responses as they generate | **Not built** |

### Chat history

| # | Requirement | Status |
|---|---|---|
| 5.1 | Save previous conversations | **Partial** — browser storage only |
| 5.2 | Display and reopen conversation history | **Partial** — same limitation |
| 5.3 | Persist conversations server side | **Not built** — the intended use for PostgreSQL |

### Authentication

| # | Requirement | Status |
|---|---|---|
| 6.1 | Create an account, log in, log out | **Not built** |
| 6.2 | Restrict documents to their owner | **Not built** — every document is currently visible to every caller |

---

## Non-functional requirements

### Performance

| Requirement | Status |
|---|---|
| Index documents at a workable rate | **Done** — ~46 pages/second embedding throughput |
| Answer within a few seconds | **Partial** — retrieval is fast; generation is bound by local model speed |
| Index in the background with progress | **Not built** — indexing blocks the upload response |

### Security

| Requirement | Status |
|---|---|
| Validate uploaded files before parsing | **Done** |
| Treat document text as data, never as markup | **Done** — the console builds text nodes, so PDF content cannot inject HTML |
| Hash passwords, authenticate with JWT | **Not built** — no authentication yet |
| Prevent unauthorised document access | **Not built** |

### Maintainability

| Requirement | Status |
|---|---|
| Modular architecture with clear boundaries | **Done** — `documents/`, `rag/`, `api/` |
| Replaceable LLM and vector store | **Done** — each sits behind a small adapter |
| Tests that run without external services | **Done** — 38 tests, no network, no ChromaDB |
| Dockerised deployment | **Partial** — Compose runs, but provisions an unused PostgreSQL |

---

## Scope

**In scope and delivered:** PDF ingestion, hybrid retrieval, grounded answers with
verified citations, a measured evaluation harness, and a web console.

**Deliberately deferred:** authentication, server-side persistence, streaming, and
background indexing.

**Future:** DOCX support, OCR for scanned PDFs, multi-tenant organisations,
role-based permissions, and a cloud LLM adapter so the app runs without a local GPU.
