from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingService
from app.rag.generation import GeneratedAnswer, GenerationError, OllamaChatService
from app.rag.retrieval import RetrievalResult, RetrievalService
from app.rag.vector_store import ChromaVectorStore, VectorStoreError

router = APIRouter(prefix="/chat", tags=["chat"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=10)
    document_id: UUID | None = None


class CitationResponse(BaseModel):
    source_id: str
    document_id: str
    page_number: int
    chunk_index: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunk_count: int


def _build_retrieval_service() -> RetrievalService:
    settings = get_settings()
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
    )


def _build_chat_service() -> OllamaChatService:
    settings = get_settings()
    return OllamaChatService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.chat_temperature,
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve evidence and generate a citation-grounded answer."""
    try:
        sources = _build_retrieval_service().search(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
        if not sources:
            return AskResponse(
                answer="I don't have enough information in the uploaded documents.",
                citations=[],
                retrieved_chunk_count=0,
            )
        generated = _build_chat_service().answer(question=request.question, sources=sources)
    except (EmbeddingError, VectorStoreError, GenerationError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval or answer-generation service is unavailable.",
        ) from error

    return AskResponse(
        answer=generated.text,
        citations=_cited_sources(generated, sources),
        retrieved_chunk_count=len(sources),
    )


def _cited_sources(
    generated: GeneratedAnswer, sources: list[RetrievalResult]
) -> list[CitationResponse]:
    source_map = {f"S{index}": source for index, source in enumerate(sources, start=1)}
    return [
        CitationResponse(
            source_id=source_id,
            document_id=source_map[source_id].chunk.document_id,
            page_number=source_map[source_id].chunk.page_number,
            chunk_index=source_map[source_id].chunk.chunk_index,
        )
        for source_id in generated.cited_source_ids
    ]
