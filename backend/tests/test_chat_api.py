from fastapi.testclient import TestClient

from app.main import app
from app.rag.generation import GeneratedAnswer
from app.rag.retrieval import RetrievalResult
from app.rag.vector_store import RetrievedChunk


def test_ask_returns_model_answer_and_citation(monkeypatch) -> None:
    source = RetrievalResult(
        chunk=RetrievedChunk(
            document_id="document-1",
            page_number=3,
            chunk_index=1,
            text="Relevant policy text.",
            distance=0.1,
        )
    )
    fake_retriever = type("FakeRetriever", (), {"search": lambda _self, **_kwargs: [source]})()
    fake_chat = type(
        "FakeChat",
        (),
        {
            "answer": lambda _self, **_kwargs: GeneratedAnswer(
                text="The answer is here. [S1]", cited_source_ids=["S1"]
            )
        },
    )()
    monkeypatch.setattr("app.api.chat.build_retrieval_service", lambda: fake_retriever)
    monkeypatch.setattr("app.api.chat._build_chat_service", lambda: fake_chat)
    client = TestClient(app)

    response = client.post("/api/v1/chat/ask", json={"question": "What is the policy?"})

    assert response.status_code == 200
    assert response.json()["citations"] == [
        {"source_id": "S1", "document_id": "document-1", "page_number": 3, "chunk_index": 1}
    ]
