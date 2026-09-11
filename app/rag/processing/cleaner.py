import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text.
    """

    if not text:
        return ""

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove extra spaces
    text = text.strip()

    return text