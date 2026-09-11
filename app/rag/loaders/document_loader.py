from app.rag.loaders.json_loader import (
    load_json_documents,
    extract_document_info,
)


def get_document_source_type(document: dict):
    """
    Identify which type of source should be used
    to process the document.
    """

    document_type = document.get(
        "document_type",
        ""
    ).lower()

    document_url = document.get(
        "document_url",
        ""
    ).lower()

    # PDF document
    if document_url.endswith(".pdf"):
        return "pdf"

    # Open-data dataset
    if "open-data" in document_type:
        return "dataset"

    # Web page or document reference
    if "web" in document_type:
        return "web"

    return "unknown"