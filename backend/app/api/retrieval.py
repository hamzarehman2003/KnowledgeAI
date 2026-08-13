from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingService
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None


class RetrievedChunkResponse(BaseModel):
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    distance: float


class SearchResponse(BaseModel):
    question: str
    result_count: int
    results: list[RetrievedChunkResponse]


@router.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Return the most semantically relevant chunks; no LLM generation yet."""
    settings = get_settings()
    service = RetrievalService(
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
    )
    try:
        results = service.search(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding or vector search service is unavailable.",
        ) from error

    return SearchResponse(
        question=request.question,
        result_count=len(results),
        results=[
            RetrievedChunkResponse(
                document_id=result.chunk.document_id,
                page_number=result.chunk.page_number,
                chunk_index=result.chunk.chunk_index,
                text=result.chunk.text,
                distance=result.chunk.distance,
            )
            for result in results
        ],
    )
