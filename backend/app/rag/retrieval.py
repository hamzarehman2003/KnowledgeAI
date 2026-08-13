from dataclasses import dataclass
from uuid import UUID

from app.rag.embeddings import OllamaEmbeddingService
from app.rag.vector_store import ChromaVectorStore, RetrievedChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: RetrievedChunk


class RetrievalService:
    """Embed a question and retrieve the closest document chunks."""

    def __init__(self, *, embedder: OllamaEmbeddingService, vector_store: ChromaVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self, *, question: str, top_k: int, document_id: UUID | None = None
    ) -> list[RetrievalResult]:
        normalized_question = " ".join(question.split())
        query_embedding = self.embedder.embed_texts([normalized_question])[0]
        chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=str(document_id) if document_id else None,
        )
        return [RetrievalResult(chunk=chunk) for chunk in chunks]
