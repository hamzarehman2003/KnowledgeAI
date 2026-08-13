import pytest

from app.rag.reranking import CrossEncoderReranker, RerankedChunk, RerankError
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import RetrievedChunk


def chunk(chunk_index: int, distance: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="document-1",
        page_number=1,
        chunk_index=chunk_index,
        text=f"chunk {chunk_index}",
        distance=distance,
    )


class FakeCrossEncoder:
    """Scores by a lookup keyed on chunk text, mimicking CrossEncoder.predict."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size: int = 32):
        self.seen_pairs = list(pairs)
        return [self.scores[text] for _question, text in self.seen_pairs]


class StubEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


class DepthRecordingStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.requested_top_k: int | None = None

    def search(self, **kwargs) -> list[RetrievedChunk]:
        self.requested_top_k = kwargs["top_k"]
        return self.chunks


class ReverseReranker:
    """Ranks later chunks first, so reordering is unambiguous in assertions."""

    def rerank(self, *, question: str, chunks) -> list[RerankedChunk]:
        return [
            RerankedChunk(chunk=chunk, score=float(index))
            for index, chunk in enumerate(reversed(list(chunks)))
        ]


def test_cross_encoder_reranker_orders_by_descending_score() -> None:
    reranker = CrossEncoderReranker(model_name="fake-model")
    reranker._model = FakeCrossEncoder({"chunk 0": 0.1, "chunk 1": 0.9, "chunk 2": 0.5})

    reranked = reranker.rerank(question="q", chunks=[chunk(0), chunk(1), chunk(2)])

    assert [candidate.chunk.chunk_index for candidate in reranked] == [1, 2, 0]
    assert reranked[0].score == pytest.approx(0.9)


def test_cross_encoder_reranker_pairs_question_with_each_chunk() -> None:
    reranker = CrossEncoderReranker(model_name="fake-model")
    fake = FakeCrossEncoder({"chunk 0": 0.1, "chunk 1": 0.2})
    reranker._model = fake

    reranker.rerank(question="what is the policy?", chunks=[chunk(0), chunk(1)])

    assert fake.seen_pairs == [
        ("what is the policy?", "chunk 0"),
        ("what is the policy?", "chunk 1"),
    ]


def test_cross_encoder_reranker_rejects_mismatched_score_count() -> None:
    reranker = CrossEncoderReranker(model_name="fake-model")
    reranker._model = type("Truncating", (), {"predict": lambda _s, pairs, batch_size=32: [0.5]})()

    with pytest.raises(RerankError):
        reranker.rerank(question="q", chunks=[chunk(0), chunk(1)])


def test_cross_encoder_reranker_returns_empty_without_loading_model() -> None:
    reranker = CrossEncoderReranker(model_name="model-that-does-not-exist")

    assert reranker.rerank(question="q", chunks=[]) == []


def test_retrieval_fetches_deeper_shortlist_when_reranking() -> None:
    store = DepthRecordingStore([chunk(index) for index in range(4)])
    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=store,
        reranker=ReverseReranker(),
        candidate_depth=50,
    )

    service.search(question="anything", top_k=3)

    assert store.requested_top_k == 50


def test_retrieval_reranks_then_truncates_to_top_k() -> None:
    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=DepthRecordingStore([chunk(index) for index in range(4)]),
        reranker=ReverseReranker(),
        candidate_depth=50,
    )

    results = service.search(question="anything", top_k=2)

    assert [result.chunk.chunk_index for result in results] == [3, 2]
    assert results[0].rerank_score == pytest.approx(0.0)


def test_retrieval_applies_threshold_before_reranking() -> None:
    store = DepthRecordingStore([chunk(0, distance=0.2), chunk(1, distance=0.9)])
    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=store,
        distance_threshold=0.5,
        reranker=ReverseReranker(),
    )

    results = service.search(question="anything", top_k=5)

    assert [result.chunk.chunk_index for result in results] == [0]
