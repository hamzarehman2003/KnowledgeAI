from uuid import uuid4

from app.rag.retrieval import RetrievalService
from app.rag.vector_store import RetrievedChunk


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["What is the refund policy?"]
        return [[0.1, 0.2]]


class FakeVectorStore:
    def search(self, **kwargs) -> list[RetrievedChunk]:
        assert kwargs["query_embedding"] == [0.1, 0.2]
        assert kwargs["top_k"] == 3
        return [
            RetrievedChunk(
                document_id="document-1",
                page_number=2,
                chunk_index=0,
                text="Refunds are available within 30 days.",
                distance=0.12,
            )
        ]


def test_retrieval_embeds_question_then_searches() -> None:
    service = RetrievalService(embedder=FakeEmbedder(), vector_store=FakeVectorStore())

    result = service.search(question="What is the refund policy?", top_k=3, document_id=uuid4())

    assert result[0].chunk.page_number == 2
