from dataclasses import dataclass
from uuid import UUID

from app.rag.embeddings import OllamaEmbeddingService
from app.rag.reranking import Reranker
from app.rag.vector_store import ChromaVectorStore, RetrievedChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: RetrievedChunk
    rerank_score: float | None = None


class RetrievalService:
    """Embed a question and retrieve the closest document chunks."""

    def __init__(
        self,
        *,
        embedder: OllamaEmbeddingService,
        vector_store: ChromaVectorStore,
        distance_threshold: float | None = None,
        reranker: Reranker | None = None,
        candidate_depth: int = 50,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.distance_threshold = distance_threshold
        self.reranker = reranker
        self.candidate_depth = candidate_depth

    def search(
        self, *, question: str, top_k: int, document_id: UUID | None = None
    ) -> list[RetrievalResult]:
        normalized_question = " ".join(question.split())
        query_embedding = self.embedder.embed_texts([normalized_question])[0]
        # A reranker can only promote chunks it is given, so fetch a deeper
        # shortlist than the caller asked for and let the cross-encoder cut it down.
        depth = max(self.candidate_depth, top_k) if self.reranker else top_k
        chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=depth,
            document_id=str(document_id) if document_id else None,
        )
        candidates = self._within_threshold(chunks)

        if self.reranker is None:
            return [RetrievalResult(chunk=chunk) for chunk in candidates[:top_k]]

        reranked = self.reranker.rerank(question=normalized_question, chunks=candidates)
        return [
            RetrievalResult(chunk=candidate.chunk, rerank_score=candidate.score)
            for candidate in reranked[:top_k]
        ]

    def _within_threshold(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop chunks the embedding model considers too distant to be evidence.

        Filtering here rather than in the vector store keeps the store a thin
        adapter and makes the policy testable without a running ChromaDB.
        """
        if self.distance_threshold is None:
            return chunks
        return [chunk for chunk in chunks if chunk.distance <= self.distance_threshold]
