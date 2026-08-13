# KnowledgeAI

An AI-first document assistant: ingest PDFs, retrieve grounded context, and answer with citations.

## Current foundation

- FastAPI backend with a clear `app/rag` domain boundary
- ChromaDB for vectors and PostgreSQL reserved for application metadata
- Ollama configuration for local chat and embedding models
- A testable, dependency-free initial text chunker

## Run the backend stack

1. Install and start [Ollama](https://ollama.com/), then pull the configured models:

   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```

2. Copy `.env.example` to `.env` and change the PostgreSQL password.
3. Run `docker compose up --build`.
4. Visit `http://localhost:8000/health` and `http://localhost:8000/docs`.

## AI/RAG roadmap

1. PDF extraction and page-aware normalization
2. Embedding and Chroma upsert with user/document metadata filters
3. Retrieval, reranking, prompt grounding, and citations
4. Retrieval/evaluation dataset and quality metrics
