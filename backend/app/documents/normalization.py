import re
import unicodedata


def normalize_extracted_text(text: str) -> str:
    """Remove extraction noise without deleting meaningful document content.

    Paragraph breaks are retained because they are useful semantic boundaries for
    chunking. Punctuation, numbers, headings, and list markers are preserved.
    """
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character for character in normalized if character == "\n" or not unicodedata.category(character).startswith("C")
    )
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
