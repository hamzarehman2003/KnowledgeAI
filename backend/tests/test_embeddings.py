from app.rag.embeddings import OllamaEmbeddingService


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, *, model: str, input: list[str]) -> dict[str, list[list[float]]]:
        self.calls.append(input)
        return {"embeddings": [[float(len(text)), 1.0] for text in input]}


def test_embeddings_are_batched_and_keep_order() -> None:
    service = OllamaEmbeddingService(base_url="http://unused", model="test", batch_size=2)
    client = FakeOllamaClient()
    service.client = client

    vectors = service.embed_texts(["one", "two", "three"])

    assert client.calls == [["one", "two"], ["three"]]
    assert vectors == [[3.0, 1.0], [3.0, 1.0], [5.0, 1.0]]
