from app.rag.lexical import BM25Index, ChunkLexicalIndex, reciprocal_rank_fusion
from app.rag.retrieval import RetrievalService
from app.rag.vector_store import RetrievedChunk


def chunk(index: int, text: str, distance: float | None = 0.1, document_id: str = "doc-1"):
    return RetrievedChunk(
        document_id=document_id,
        page_number=1,
        chunk_index=index,
        text=text,
        distance=distance,
    )


class StubEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


class StubStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def search(self, **_kwargs) -> list[RetrievedChunk]:
        return self.chunks


def test_bm25_ranks_exact_term_matches_first() -> None:
    index = BM25Index(
        ids=["a", "b", "c"],
        texts=[
            "the cat sat on the mat",
            "warfarin interacts with vitamin k",
            "a dog barked loudly",
        ],
    )

    assert index.search("warfarin", top_k=1) == ["b"]


def test_bm25_filters_by_owner() -> None:
    index = BM25Index(
        ids=["a", "b"],
        texts=["shared keyword here", "shared keyword here"],
        owners=["doc-1", "doc-2"],
    )

    assert index.search("keyword", top_k=5, owner="doc-2") == ["b"]


def test_bm25_returns_nothing_for_unknown_terms() -> None:
    index = BM25Index(ids=["a"], texts=["hello world"])

    assert index.search("nonexistentterm", top_k=5) == []


def test_rrf_promotes_documents_ranked_well_by_both_lists() -> None:
    # "b" is 2nd and 2nd; "a" and "c" each top one list but are absent from the other.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=10, top_k=3)

    assert fused[0] == "b"


def test_hybrid_recovers_a_chunk_dense_search_missed() -> None:
    dense_only = chunk(0, "general discussion of blood thinners")
    lexical_only = chunk(1, "warfarin dosage guidance", distance=None)
    index = ChunkLexicalIndex([dense_only, lexical_only])

    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=StubStore([dense_only]),
        lexical_index_provider=lambda: index,
    )

    results = service.search(question="warfarin", top_k=5)

    assert lexical_only.chunk_id in {result.chunk.chunk_id for result in results}


def test_hybrid_keeps_the_dense_record_so_distance_survives_fusion() -> None:
    shared = chunk(0, "warfarin dosage guidance", distance=0.22)
    index = ChunkLexicalIndex([chunk(0, "warfarin dosage guidance", distance=None)])

    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=StubStore([shared]),
        lexical_index_provider=lambda: index,
    )

    results = service.search(question="warfarin", top_k=5)

    assert results[0].chunk.distance == 0.22


def test_threshold_never_drops_lexical_only_chunks() -> None:
    lexical_only = chunk(1, "warfarin dosage guidance", distance=None)
    service = RetrievalService(
        embedder=StubEmbedder(),
        vector_store=StubStore([]),
        distance_threshold=0.3,
        lexical_index_provider=lambda: ChunkLexicalIndex([lexical_only]),
    )

    results = service.search(question="warfarin", top_k=5)

    assert [result.chunk.chunk_id for result in results] == [lexical_only.chunk_id]


def test_dense_only_service_ignores_lexical_path() -> None:
    only = chunk(0, "warfarin dosage guidance", distance=0.4)
    service = RetrievalService(embedder=StubEmbedder(), vector_store=StubStore([only]))

    results = service.search(question="warfarin", top_k=5)

    assert len(results) == 1
    assert results[0].rerank_score is None
