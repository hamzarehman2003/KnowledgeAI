# KnowledgeAI

An AI document assistant: upload PDFs, ask questions in natural language, get answers
grounded strictly in the retrieved text with verifiable citations. Runs entirely locally
on Ollama — no data leaves the machine.

The interesting part of this project is not that it does RAG. It is that **every retrieval
decision in it was measured**, on a standard benchmark, including the ones that turned out
to be wrong.

---

## Retrieval quality

Measured on [BEIR NFCorpus](https://github.com/beir-cellar/beir) — 3,633 documents,
323 test queries, human-graded relevance judgements. Primary metric is nDCG@10.

| retrieval strategy | nDCG@10 | recall@10 | recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 0.3056 | 0.1464 | 0.2366 | 0.5068 |
| dense embeddings (baseline) | 0.3399 | 0.1659 | 0.3047 | 0.5343 |
| dense + cross-encoder reranking | 0.3112 ✗ | 0.1412 | 0.3047 | 0.5322 |
| **hybrid dense + BM25 (RRF)** | **0.3518** ✓ | **0.1710** | **0.3145** | **0.5580** |

Reproduce with:

```bash
python -m eval.beir_index ../data/nfcorpus     # embed and store the corpus
python -m eval.beir_eval  ../data/nfcorpus --mode hybrid
```

### Why reranking was rejected

A cross-encoder reranker (`BAAI/bge-reranker-base`) is the standard next move after dense
retrieval, so it was the first thing tried. It **made results worse** — nDCG@10 fell 8.4%
— while costing 1.8 s/query on a GPU.

That was not an implementation bug: an inverted sort would have collapsed MRR@10, and MRR
barely moved. Per-query analysis showed some queries improving sharply and others
degrading.

The diagnosis mattered more than the result. `recall@100` was 0.3047 and **identical**
before and after reranking, because reranking only reorders what search already found.
Roughly 70% of relevant documents never entered the candidate pool at all — in a sampled
10 queries, 4 had *zero* relevant documents anywhere in their top 50.

**The bottleneck was recall, not ranking.** That is what pointed to hybrid retrieval, which
attacks the candidate pool directly, and it is the change that finally moved the number.

Reranking remains in the codebase behind `RERANKING_ENABLED=false` — a measured, rejected
option rather than deleted work.

### Method notes

- **BM25 is written out** in [`app/rag/lexical.py`](backend/app/rag/lexical.py) rather than
  imported, so the scoring is inspectable. Standalone it scores 0.3056 against the
  published BEIR baseline of ~0.32, which is the correctness check.
- **RRF's damping constant was tuned on the dev split and reported on test.** Tuning it on
  test would have inflated the result — `k=5` wins on test, `k=10` on dev, so `k=10` ships.
- **nDCG@10 is the primary metric, not recall@5.** NFCorpus has a median of 16 relevant
  documents per query, so recall@5 is capped near 30% even for a perfect retriever and
  would read as failure. nDCG@10 handles graded relevance and rank position correctly.
- **The distance threshold is disabled by default.** The sweep suggested a 0.50 cosine
  cutoff, but real uploaded PDFs score 0.53–0.61, so shipping that value would have made
  the app refuse every question. Benchmark-derived thresholds do not transfer across
  corpora.

---

## How it works

```
PDF ──▶ validate ──▶ extract ──▶ normalize ──▶ chunk ──▶ embed ──▶ ChromaDB
                                                                      │
                    question ──┬──▶ embed ──▶ vector search ──┐       │
                               │                              ├─ RRF ─┴──▶ LLM ──▶ answer
                               └──▶ BM25 lexical search ──────┘              + citations
```

| stage | module | notes |
|---|---|---|
| Validation | `documents/validation.py` | Checks the `%PDF-` signature, not the filename or MIME type — both are user-controlled |
| Extraction | `documents/extraction.py` | Page-aware, so citations can name a page. Rejects encrypted and image-only PDFs |
| Normalization | `documents/normalization.py` | Strips control characters, keeps paragraph breaks as chunk boundaries |
| Chunking | `rag/chunking.py` | Paragraph → sentence → token-window fallback, with overlap |
| Embedding | `rag/embeddings.py` | Batched Ollama calls behind a small adapter |
| Vector store | `rag/vector_store.py` | Cosine space; re-indexing deletes prior vectors so no stale chunks survive |
| Lexical index | `rag/lexical.py` | BM25 + RRF, **derived from the vector store** so it cannot drift out of sync |
| Generation | `rag/generation.py` | Grounded prompt; **model-emitted citations are validated against the sources actually retrieved**, so it cannot cite something that was never supplied |

---

## Running it

Requires [Ollama](https://ollama.com/):

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Start ChromaDB and the API:

```bash
chroma run --host localhost --port 8001 --path backend/chroma_data
```

```bash
cd backend && uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the console, or `/docs` for the OpenAPI schema.
`docker compose up --build` also works if you prefer containers.

## Tests

```bash
cd backend && pytest --basetemp=/tmp/pytest
```

38 tests, no network and no running services required — the vector store and embedding
adapters are stubbed at the seam, so the suite is fast and hermetic.

## Configuration

| variable | default | purpose |
|---|---|---|
| `HYBRID_RETRIEVAL_ENABLED` | `true` | Fuse BM25 with embedding search |
| `RRF_K` | `10` | RRF damping; chosen on the dev split |
| `RERANKING_ENABLED` | `false` | Cross-encoder reranking — measured, rejected |
| `RETRIEVAL_DISTANCE_THRESHOLD` | unset | Cosine cutoff; unset because it does not transfer across corpora |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b` | Swappable behind the adapter |

## Stack

FastAPI · ChromaDB · Ollama (`qwen2.5:7b`, `nomic-embed-text`) · pypdf · pytest · Docker

## Roadmap

- Authentication and per-user vector filtering
- Persisted documents and chat history in PostgreSQL
- Background indexing with progress, and streaming answers
- A cloud LLM adapter so the demo runs without a local GPU
