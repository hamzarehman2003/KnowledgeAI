from collections.abc import Sequence
from typing import Protocol

from ollama import Client


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider returns an invalid response."""


class EmbeddingClient(Protocol):
    def embed(self, *, model: str, input: list[str]) -> object: ...


class OllamaEmbeddingService:
    """Small adapter around Ollama's embedding API with batching."""

    def __init__(self, *, base_url: str, model: str, batch_size: int = 64) -> None:
        self.client: EmbeddingClient = Client(host=base_url)
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = self.client.embed(model=self.model, input=batch)
            except Exception as error:
                raise EmbeddingError("Could not generate embeddings through Ollama.") from error
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise EmbeddingError("Ollama returned an unexpected number of embeddings.")
            vectors.extend([list(vector) for vector in embeddings])

        vector_dimensions = {len(vector) for vector in vectors}
        if len(vector_dimensions) != 1 or 0 in vector_dimensions:
            raise EmbeddingError("Ollama returned embeddings with invalid dimensions.")
        return vectors
