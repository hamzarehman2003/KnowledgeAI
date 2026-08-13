import pytest

from app.rag.chunking import chunk_page_text


def test_chunks_preserve_page_and_overlap() -> None:
    chunks = chunk_page_text(
        "one two three four five six seven", page_number=4, max_tokens=4, overlap_tokens=2
    )

    assert [chunk.text for chunk in chunks] == ["one two three four", "three four five six", "five six seven"]
    assert [chunk.page_number for chunk in chunks] == [4, 4, 4]


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (10, -1), (10, 10)])
def test_invalid_chunk_parameters_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_page_text("text", page_number=1, max_tokens=chunk_size, overlap_tokens=overlap)
