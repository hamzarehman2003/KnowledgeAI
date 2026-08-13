import pytest

from app.documents.validation import InvalidDocumentError, validate_pdf_upload


def test_valid_pdf_is_accepted() -> None:
    validated = validate_pdf_upload(
        filename="policy.pdf",
        content_type="application/pdf",
        content=b"%PDF-1.7 minimal test file",
        max_size_mb=25,
    )

    assert validated.filename == "policy.pdf"


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("notes.txt", "text/plain", b"%PDF-1.7"),
        ("empty.pdf", "application/pdf", b""),
        ("fake.pdf", "application/pdf", b"not a PDF"),
    ],
)
def test_invalid_pdf_is_rejected(
    filename: str, content_type: str, content: bytes
) -> None:
    with pytest.raises(InvalidDocumentError):
        validate_pdf_upload(
            filename=filename,
            content_type=content_type,
            content=content,
            max_size_mb=25,
        )
