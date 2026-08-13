from functools import lru_cache

from app.core.config import get_settings
from app.rag.embeddings import OllamaEmbeddingService
from app.rag.lexical import ChunkLexicalIndex
from app.rag.reranking import CrossEncoderReranker
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import ChromaVectorStore

_lexical_cache: dict[str, object] = {}


def build_lexical_index(vector_store: ChromaVectorStore) -> ChunkLexicalIndex:
    """Build the BM25 index from the vector store, reusing it while unchanged.

    Chunk count is a cheap staleness signal that catches uploads and deletions.
    It cannot see an edit that preserves the count, so `reset_lexical_index()`
    is called from the indexing endpoint, which is the only writer.
    """
    chunks = vector_store.all_chunks()
    if _lexical_cache.get("count") != len(chunks) or "index" not in _lexical_cache:
        _lexical_cache["index"] = ChunkLexicalIndex(chunks)
        _lexical_cache["count"] = len(chunks)
    return _lexical_cache["index"]


def reset_lexical_index() -> None:
    """Drop the cached index after a write, so the next search rebuilds it."""
    _lexical_cache.clear()


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
    vector_store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection_name,
    )
    return RetrievalService(
        embedder=OllamaEmbeddingService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            batch_size=settings.embedding_batch_size,
        ),
        vector_store=vector_store,
        distance_threshold=settings.retrieval_distance_threshold,
        reranker=reranker,
        candidate_depth=settings.rerank_candidate_depth,
        lexical_index_provider=(
            (lambda: build_lexical_index(vector_store))
            if settings.hybrid_retrieval_enabled
            else None
        ),
        rrf_k=settings.rrf_k,
    )
