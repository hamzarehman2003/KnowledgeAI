from app.documents.normalization import normalize_extracted_text


def test_normalization_preserves_paragraphs_and_removes_extra_spacing() -> None:
    text = "Refunds   are\n available.\x00\n\n\nContact   support."

    assert normalize_extracted_text(text) == "Refunds are\navailable.\n\nContact support."
