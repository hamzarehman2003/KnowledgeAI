from dataclasses import dataclass

PDF_SIGNATURE = b"%PDF-"


class InvalidDocumentError(ValueError):
    """Raised when an uploaded file is not an acceptable PDF."""


@dataclass(frozen=True)
class ValidatedPdf:
    """Safe, validated input for the next ingestion stage."""

    filename: str
    content: bytes


def validate_pdf_upload(
    *, filename: str | None, content_type: str | None, content: bytes, max_size_mb: int
) -> ValidatedPdf:
    """Apply inexpensive checks before a PDF is saved or parsed.

    MIME types and filenames are user-controlled, so the PDF file signature is
    the authoritative format check at this stage.
    """
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidDocumentError("Only files with a .pdf extension are accepted.")

    if content_type and content_type not in {"application/pdf", "application/x-pdf"}:
        raise InvalidDocumentError("The uploaded file must have a PDF content type.")

    max_size_bytes = max_size_mb * 1024 * 1024
    if not content:
        raise InvalidDocumentError("The uploaded PDF is empty.")
    if len(content) > max_size_bytes:
        raise InvalidDocumentError(f"PDF exceeds the {max_size_mb} MB upload limit.")
    if not content.startswith(PDF_SIGNATURE):
        raise InvalidDocumentError("The file content is not a valid PDF.")

    return ValidatedPdf(filename=filename, content=content)
