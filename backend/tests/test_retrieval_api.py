from fastapi.testclient import TestClient

from app.main import app
from app.rag.retrieval import RetrievalResult
from app.rag.vector_store import RetrievedChunk


def test_search_endpoint_returns_sources(monkeypatch) -> None:
    fake_retriever = type(
        "FakeRetriever",
        (),
        {
            "search": lambda _self, **_kwargs: [
                RetrievalResult(
                    chunk=RetrievedChunk(
                        document_id="document-1",
                        page_number=1,
                        chunk_index=0,
                        text="Relevant text.",
                        distance=0.05,
                    )
                )
            ]
        },
    )()
    monkeypatch.setattr("app.api.retrieval.build_retrieval_service", lambda: fake_retriever)
    client = TestClient(app)

    response = client.post("/api/v1/retrieval/search", json={"question": "Find relevant text"})

    assert response.status_code == 200
    assert response.json()["results"][0]["page_number"] == 1
