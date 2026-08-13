from functools import lru_cache

from app.core.config import get_settings
from app.rag.embeddings import OllamaEmbeddingService
from app.rag.reranking import CrossEncoderReranker
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import ChromaVectorStore


@lru_cache(maxsize=1)
def _cached_reranker(model_name: str, device: str | None, batch_size: int) -> CrossEncoderReranker:
    """Reuse one reranker across requests; the cross-encoder weights are ~1GB.

    The instance is cheap to build and loads its model on first use, so caching
    it here keeps model loading off the request path after the first question.
    """
    return CrossEncoderReranker(model_name=model_name, device=device, batch_size=batch_size)


def build_retrieval_service() -> RetrievalService:
    settings = get_settings()
    reranker = (
        _cached_reranker(
            settings.reranker_model,
            settings.reranker_device,
            settings.reranker_batch_size,
        )
        if settings.reranking_enabled
        else None
    )
    return RetrievalService(
        embedder=OllamaEmbeddingService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            batch_size=settings.embedding_batch_size,
        ),
        vector_store=ChromaVectorStore(
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection_name=settings.chroma_collection_name,
        ),
        distance_threshold=settings.retrieval_distance_threshold,
        reranker=reranker,
        candidate_depth=settings.rerank_candidate_depth,
    )
