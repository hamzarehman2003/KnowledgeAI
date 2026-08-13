from dataclasses import dataclass


@dataclass(frozen=True)
class SourceChunk:
    """A retrievable unit of document knowledge with citation metadata."""

    content: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
