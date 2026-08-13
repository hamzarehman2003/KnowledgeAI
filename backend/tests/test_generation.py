from app.rag.generation import OllamaChatService
from app.rag.retrieval import RetrievalResult
from app.rag.vector_store import RetrievedChunk


class FakeChatClient:
    def chat(self, **_kwargs):
        return {"message": {"content": "Refunds are allowed within 30 days. [S1]"}}


def test_generated_answer_keeps_only_valid_citations() -> None:
    service = OllamaChatService(base_url="http://unused", model="test")
    service.client = FakeChatClient()
    sources = [
        RetrievalResult(
            chunk=RetrievedChunk(
                document_id="document-1",
                page_number=2,
                chunk_index=0,
                text="Refunds are allowed within 30 days.",
                distance=0.1,
            )
        )
    ]

    result = service.answer(question="What is the refund period?", sources=sources)

    assert result.cited_source_ids == ["S1"]
