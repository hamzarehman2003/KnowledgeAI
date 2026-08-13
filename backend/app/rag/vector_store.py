from collections.abc import Sequence
from dataclasses import dataclass

import chromadb

from app.rag.chunking import TextChunk


class VectorStoreError(RuntimeError):
    """Raised when vectors cannot be written to ChromaDB."""


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    # None when a chunk was found lexically rather than by embedding distance.
    distance: float | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.page_number}:{self.chunk_index}"


class ChromaVectorStore:
    """Store document vectors and the metadata required for later citations."""

    def __init__(self, *, host: str, port: int, collection_name: str) -> None:
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
        except Exception as error:
            raise VectorStoreError("Could not connect to ChromaDB.") from error
        self.collection_name = collection_name

    def replace_document_chunks(
        self,
        *,
        document_id: str,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Every chunk must have exactly one embedding.")

        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # Re-indexing must not leave stale vectors from an older chunking run.
            collection.delete(where={"document_id": document_id})
            collection.upsert(
                ids=[f"{document_id}:{chunk.page_number}:{chunk.chunk_index}" for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=[list(vector) for vector in embeddings],
                metadatas=[
                    {
                        "document_id": document_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "embedding_model": embedding_model,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as error:
            raise VectorStoreError("Could not write vectors to ChromaDB.") from error

    def all_chunks(self) -> list[RetrievedChunk]:
        """Read every stored chunk, for building the lexical index.

        The vector store is the single source of truth for chunk text, so the
        BM25 index is derived from it rather than persisted separately; it
        therefore cannot drift out of sync with what is actually searchable.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            result = collection.get(include=["documents", "metadatas"])
        except Exception as error:
            raise VectorStoreError("Could not read chunks from ChromaDB.") from error

        return [
            RetrievedChunk(
                document_id=str(metadata["document_id"]),
                page_number=int(metadata["page_number"]),
                chunk_index=int(metadata["chunk_index"]),
                text=str(text),
                distance=None,
            )
            for text, metadata in zip(
                result.get("documents") or [], result.get("metadatas") or [], strict=True
            )
        ]

    def search(
        self, *, query_embedding: Sequence[float], top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Find chunks closest to a question embedding by cosine distance."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            query_arguments: dict[str, object] = {
                "query_embeddings": [list(query_embedding)],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if document_id:
                query_arguments["where"] = {"document_id": document_id}
            result = collection.query(**query_arguments)

            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            return [
                RetrievedChunk(
                    document_id=str(metadata["document_id"]),
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=str(text),
                    distance=float(distance),
                )
                for text, metadata, distance in zip(documents, metadatas, distances, strict=True)
            ]
        except Exception as error:
            raise VectorStoreError("Could not search ChromaDB.") from error
