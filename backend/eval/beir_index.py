"""Index a BEIR dataset into its own Chroma collection.

Eval vectors are kept in a separate collection from ``document_chunks`` so that
benchmarking never pollutes real uploaded documents. Embeddings go through the
production ``OllamaEmbeddingService`` on purpose: the point is to measure the
pipeline we actually ship, not an idealised one.
"""

import argparse
import json
import time
from pathlib import Path

import chromadb

from app.core.config import get_settings
from app.rag.embeddings import OllamaEmbeddingService

UPSERT_BATCH = 500


def load_corpus(dataset_dir: Path, limit: int | None) -> list[dict[str, str]]:
    """Read BEIR ``corpus.jsonl`` into id/text records."""
    documents: list[dict[str, str]] = []
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            # BEIR convention: title and body are concatenated for retrieval.
            title = (record.get("title") or "").strip()
            body = (record.get("text") or "").strip()
            text = f"{title}\n{body}".strip() if title else body
            if not text:
                continue
            documents.append({"id": record["_id"], "text": text})
            if limit and len(documents) >= limit:
                break
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="folder holding corpus.jsonl")
    parser.add_argument("--collection", default="nfcorpus_eval")
    parser.add_argument("--limit", type=int, default=None, help="index only the first N docs")
    arguments = parser.parse_args()

    settings = get_settings()
    documents = load_corpus(arguments.dataset_dir, arguments.limit)
    print(f"loaded {len(documents)} documents from {arguments.dataset_dir}")

    embedder = OllamaEmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    # A stale collection would silently mix runs with different chunking or models.
    try:
        client.delete_collection(arguments.collection)
        print(f"dropped existing collection {arguments.collection!r}")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=arguments.collection, metadata={"hnsw:space": "cosine"}
    )

    started = time.perf_counter()
    for start in range(0, len(documents), UPSERT_BATCH):
        batch = documents[start : start + UPSERT_BATCH]
        vectors = embedder.embed_texts([document["text"] for document in batch])
        collection.upsert(
            ids=[document["id"] for document in batch],
            documents=[document["text"] for document in batch],
            embeddings=vectors,
            metadatas=[{"corpus_id": document["id"]} for document in batch],
        )
        done = start + len(batch)
        elapsed = time.perf_counter() - started
        rate = done / elapsed
        remaining = (len(documents) - done) / rate if rate else 0
        print(
            f"  {done}/{len(documents)} indexed "
            f"({rate:.1f} docs/s, ~{remaining/60:.1f} min left)",
            flush=True,
        )

    total = time.perf_counter() - started
    print(f"\ndone: {collection.count()} vectors in {arguments.collection!r} ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
