from fastapi.testclient import TestClient

from app.main import app


def test_upload_pdf_returns_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.api.documents.get_settings", lambda: type("Settings", (), {
        "max_upload_size_mb": 25,
        "upload_directory": str(tmp_path),
    })())
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.pdf", b"%PDF-1.7 sample", "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "policy.pdf"
    assert data["status"] == "uploaded"
    assert (tmp_path / f"{data['document_id']}.pdf").exists()


def test_upload_rejects_non_pdf() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
