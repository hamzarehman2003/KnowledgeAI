from uuid import uuid4

from fastapi.testclient import TestClient

from app.rag.chunking import TextChunk
from app.main import app


def test_chunk_document_returns_page_metadata(tmp_path, monkeypatch) -> None:
    document_id = uuid4()
    (tmp_path / f"{document_id}.pdf").write_bytes(b"%PDF-1.7 placeholder")
    monkeypatch.setattr("app.api.documents.get_settings", lambda: type("Settings", (), {
        "upload_directory": str(tmp_path),
        "chunk_size_tokens": 4,
        "chunk_overlap_tokens": 1,
    })())
    monkeypatch.setattr(
        "app.api.documents.build_document_chunks",
        lambda *_args, **_kwargs: [
            TextChunk(page_number=2, chunk_index=0, text="one two three four", token_count=4),
            TextChunk(page_number=2, chunk_index=1, text="four five", token_count=2),
        ],
    )
    client = TestClient(app)

    response = client.get(f"/api/v1/documents/{document_id}/chunks")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2
    assert response.json()["chunks"][0]["page_number"] == 2
