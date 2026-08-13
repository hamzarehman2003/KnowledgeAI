"""Standalone single-endpoint RAG generator.

Note: this duplicates the maintained pipeline in ``backend/app`` (see
``app/rag/generation.py`` and ``app/api/chat.py``), which additionally validates
model-emitted citations and covers the flow with tests. Prefer that one; this
file is kept only as a minimal self-contained example.
"""

import os

import chromadb
from fastapi import FastAPI, HTTPException
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

app = FastAPI()


class GenerateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    n_results: int = Field(default=5, ge=1, le=20)


def _chroma_collection():
    """Connect lazily; HttpClient probes the server as soon as it is built."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT).get_collection(
        name=COLLECTION_NAME
    )


def _ollama() -> OllamaClient:
    return OllamaClient(host=OLLAMA_BASE_URL)


@app.post("/generate")
def generate_answer(request: GenerateRequest) -> dict[str, object]:
    try:
        # The collection was written with nomic-embed-text vectors, so the query
        # must be embedded by the same model rather than Chroma's default.
        embedding = _ollama().embed(model=EMBEDDING_MODEL, input=[request.query])
        query_embedding = list(embedding.embeddings[0])

        result = _chroma_collection().query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            include=["documents", "metadatas"],
        )
        # Chroma nests one list per submitted query; we submit exactly one.
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        if not documents:
            return {
                "answer": "I don't have enough information in the uploaded documents.",
                "citations": [],
            }

        context = "\n\n".join(
            f"[S{index}] {text}" for index, text in enumerate(documents, start=1)
        )
        prompt = f"Query: {request.query}\n\nContext:\n{context}\n\nAnswer:"

        response = _ollama().generate(model=CHAT_MODEL, prompt=prompt)
        return {
            "answer": response.response.strip(),
            "citations": [dict(metadata) for metadata in metadatas],
        }
    except Exception as error:
        raise HTTPException(
            status_code=503, detail="Retrieval or generation service is unavailable."
        ) from error
