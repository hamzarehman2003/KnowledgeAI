from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.rag.chunking import TextChunk


def test_index_document_embeds_and_stores_chunks(tmp_path, monkeypatch) -> None:
    document_id = uuid4()
    (tmp_path / f"{document_id}.pdf").write_bytes(b"%PDF-1.7 placeholder")
    monkeypatch.setattr("app.api.documents.get_settings", lambda: type("Settings", (), {
        "upload_directory": str(tmp_path),
        "chunk_size_tokens": 450,
        "chunk_overlap_tokens": 75,
        "ollama_base_url": "http://unused",
        "ollama_embedding_model": "nomic-embed-text",
        "embedding_batch_size": 64,
        "chroma_host": "unused",
        "chroma_port": 8000,
        "chroma_collection_name": "test_chunks",
    })())
    chunks = [TextChunk(page_number=1, chunk_index=0, text="policy text", token_count=2)]
    monkeypatch.setattr("app.api.documents.build_document_chunks", lambda *_args, **_kwargs: chunks)
    monkeypatch.setattr(
        "app.api.documents.OllamaEmbeddingService.embed_texts", lambda _self, _texts: [[0.1, 0.2]]
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "app.api.documents.ChromaVectorStore.replace_document_chunks",
        lambda _self, **kwargs: captured.update(kwargs),
    )
    client = TestClient(app)

    response = client.post(f"/api/v1/documents/{document_id}/index")

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert captured["embeddings"] == [[0.1, 0.2]]
