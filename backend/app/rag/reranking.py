from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.rag.vector_store import RetrievedChunk


class RerankError(RuntimeError):
    """Raised when the reranking model cannot score candidates."""


@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk
    score: float


class Reranker(Protocol):
    def rerank(
        self, *, question: str, chunks: Sequence[RetrievedChunk]
    ) -> list[RerankedChunk]: ...


class CrossEncoderReranker:
    """Reorder retrieved candidates with a cross-encoder.

    Embedding search scores the question and a chunk independently, so it never
    sees how the two interact. A cross-encoder reads both together and is far
    more accurate per pair, but it is far too slow to run across a whole corpus
    -- hence it runs only over the shortlist that vector search already narrowed
    down.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: object | None = None

    def _load_model(self) -> object:
        """Import and load lazily; torch is heavy and reranking is optional."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:  # pragma: no cover - depends on install extras
                raise RerankError(
                    "Reranking requires the sentence-transformers extra to be installed."
                ) from error
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as error:
                raise RerankError(f"Could not load reranking model {self.model_name!r}.") from error
        return self._model

    def rerank(self, *, question: str, chunks: Sequence[RetrievedChunk]) -> list[RerankedChunk]:
        if not chunks:
            return []

        model = self._load_model()
        try:
            scores = model.predict(
                [(question, chunk.text) for chunk in chunks],
                batch_size=self.batch_size,
            )
        except Exception as error:
            raise RerankError("Could not score candidates with the reranking model.") from error

        if len(scores) != len(chunks):
            raise RerankError("The reranking model returned an unexpected number of scores.")

        reranked = [
            RerankedChunk(chunk=chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        reranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return reranked
