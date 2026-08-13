from uuid import uuid4

from fastapi.testclient import TestClient

from app.documents.extraction import ExtractedPage
from app.main import app


def test_extract_document_returns_page_aware_text(tmp_path, monkeypatch) -> None:
    document_id = uuid4()
    (tmp_path / f"{document_id}.pdf").write_bytes(b"%PDF-1.7 placeholder")
    monkeypatch.setattr("app.api.documents.get_settings", lambda: type("Settings", (), {
        "upload_directory": str(tmp_path),
    })())
    monkeypatch.setattr(
        "app.api.documents.extract_pdf_pages",
        lambda _: [
            ExtractedPage(page_number=1, text="First page text."),
            ExtractedPage(page_number=3, text="Third page text."),
        ],
    )
    client = TestClient(app)

    response = client.get(f"/api/v1/documents/{document_id}/extract")

    assert response.status_code == 200
    assert response.json()["extracted_page_count"] == 2
    assert response.json()["extracted_pages"][1]["page_number"] == 3


def test_extract_returns_404_for_missing_document(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.api.documents.get_settings", lambda: type("Settings", (), {
        "upload_directory": str(tmp_path),
    })())
    client = TestClient(app)

    response = client.get(f"/api/v1/documents/{uuid4()}/extract")

    assert response.status_code == 404
