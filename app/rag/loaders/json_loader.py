import json
from pathlib import Path


def load_json_documents(file_path: str):
    """
    Load document metadata from a JSON file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("official_source_catalog", [])


def extract_document_info(document: dict):
    """
    Extract important information required
    for the RAG ingestion pipeline.
    """

    return {
        "document_id": document.get("document_id"),
        "document_title": document.get("document_title"),
        "document_type": document.get("document_type"),
        "document_url": document.get("document_url"),
        "source_page": document.get("source_page"),
        "publisher": document.get("publisher"),
        "scope": document.get("scope"),
    }