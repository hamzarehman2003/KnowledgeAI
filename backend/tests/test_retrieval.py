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


class StubEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


class DistanceStubStore:
    """Returns one chunk per supplied distance, nearest first."""

    def __init__(self, *distances: float) -> None:
        self.distances = distances

    def search(self, **_kwargs) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                document_id="document-1",
                page_number=1,
                chunk_index=index,
                text=f"chunk {index}",
                distance=distance,
            )
            for index, distance in enumerate(self.distances)
        ]


def test_retrieval_embeds_question_then_searches() -> None:
    service = RetrievalService(embedder=FakeEmbedder(), vector_store=FakeVectorStore())

    result = service.search(question="What is the refund policy?", top_k=3, document_id=uuid4())

    assert result[0].chunk.page_number == 2


def test_retrieval_drops_chunks_beyond_distance_threshold() -> None:
    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=DistanceStubStore(0.20, 0.45, 0.80),
        distance_threshold=0.50,
    )

    results = service.search(question="anything", top_k=3)

    assert [result.chunk.distance for result in results] == [0.20, 0.45]


def test_retrieval_keeps_every_chunk_when_threshold_is_unset() -> None:
    service = RetrievalService(
        embedder=StubEmbedder(), vector_store=DistanceStubStore(0.20, 0.80, 0.99)
    )

    results = service.search(question="anything", top_k=3)

    assert len(results) == 3
