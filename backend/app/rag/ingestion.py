from pathlib import Path

from app.documents.extraction import extract_pdf_pages
from app.documents.normalization import normalize_extracted_text
from app.rag.chunking import TextChunk, chunk_page_text


def build_document_chunks(
    pdf_path: Path, *, max_tokens: int, overlap_tokens: int
) -> list[TextChunk]:
    """Transform an uploaded PDF into normalized, page-aware chunks."""
    return [
        chunk
        for page in extract_pdf_pages(pdf_path)
        for chunk in chunk_page_text(
            normalize_extracted_text(page.text),
            page_number=page.page_number,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]
