from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfExtractionError(ValueError):
    """Raised when a PDF cannot be parsed or contains no usable text."""


class EncryptedPdfError(PdfExtractionError):
    """Raised when a password-protected PDF is uploaded."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[ExtractedPage]:
    """Extract non-empty text from each PDF page while retaining page numbers."""
    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as error:
        raise PdfExtractionError("The uploaded file could not be read as a PDF.") from error

    if reader.is_encrypted:
        raise EncryptedPdfError("Password-protected PDFs are not supported.")

    extracted_pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as error:
            raise PdfExtractionError(f"Text extraction failed on page {page_number}.") from error

        if text:
            extracted_pages.append(ExtractedPage(page_number=page_number, text=text))

    if not extracted_pages:
        raise PdfExtractionError(
            "No extractable text was found. This may be a scanned or image-only PDF."
        )

    return extracted_pages
