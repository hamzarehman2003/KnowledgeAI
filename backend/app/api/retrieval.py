from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.dependencies import build_retrieval_service
from app.rag.embeddings import EmbeddingError
from app.rag.reranking import RerankError
from app.rag.vector_store import VectorStoreError

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
    # Null when BM25 surfaced the chunk and embedding search did not, so there
    # is no cosine distance to report.
    distance: float | None


class SearchResponse(BaseModel):
    question: str
    result_count: int
    results: list[RetrievedChunkResponse]


@router.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Return the most semantically relevant chunks; no LLM generation yet."""
    try:
        results = build_retrieval_service().search(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except (EmbeddingError, VectorStoreError, RerankError) as error:
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
