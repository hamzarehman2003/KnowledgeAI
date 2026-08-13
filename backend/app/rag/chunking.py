import re
from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class TextChunk:
    """A page-aware text unit ready for embedding in a later stage."""

    page_number: int
    chunk_index: int
    text: str
    token_count: int


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _split_long_unit(text: str, max_tokens: int) -> list[str]:
    """Prefer sentence boundaries; use token windows only as a final fallback."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        sentence_tokens = _tokens(sentence)
        if not sentence_tokens:
            continue
        if len(sentence_tokens) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(
                " ".join(sentence_tokens[start : start + max_tokens])
                for start in range(0, len(sentence_tokens), max_tokens)
            )
        elif len(_tokens(" ".join(current))) + len(sentence_tokens) > max_tokens and current:
            pieces.append(" ".join(current))
            current = sentence_tokens
        else:
            current.extend(sentence_tokens)

    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(text: str, max_tokens: int) -> Iterable[str]:
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(_tokens(paragraph)) <= max_tokens:
            yield paragraph
        else:
            yield from _split_long_unit(paragraph, max_tokens)


def chunk_page_text(
    text: str, *, page_number: int, max_tokens: int = 450, overlap_tokens: int = 75
) -> list[TextChunk]:
    """Create paragraph-aware, overlapping chunks for one extracted PDF page.

    Token counts are whitespace-token estimates. This is a transparent baseline;
    later we can replace it with the embedding model's exact tokenizer if tests
    show a material benefit.
    """
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("max_tokens must be positive and overlap must be smaller than max_tokens")

    chunks: list[TextChunk] = []
    current_tokens: list[str] = []

    for unit in _units(text, max_tokens):
        remaining_tokens = _tokens(unit)
        while remaining_tokens:
            available_tokens = max_tokens - len(current_tokens)
            current_tokens.extend(remaining_tokens[:available_tokens])
            remaining_tokens = remaining_tokens[available_tokens:]

            if not remaining_tokens:
                break
            chunks.append(
                TextChunk(
                    page_number=page_number,
                    chunk_index=len(chunks),
                    text=" ".join(current_tokens),
                    token_count=len(current_tokens),
                )
            )
            current_tokens = current_tokens[-overlap_tokens:] if overlap_tokens else []

    if current_tokens:
        chunks.append(
            TextChunk(
                page_number=page_number,
                chunk_index=len(chunks),
                text=" ".join(current_tokens),
                token_count=len(current_tokens),
            )
        )
    return chunks
