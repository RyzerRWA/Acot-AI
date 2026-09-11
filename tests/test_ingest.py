from app.rag.ingestion.ingest import (
    ingest_documents
)


file_path = "data/raw/documents.json"


chunks = ingest_documents(
    file_path
)


print("\nFIRST CHUNK:\n")


if chunks:

    print(
        chunks[0]["text"][:1000]
    )

    print("\nMETADATA:\n")

    print(
        chunks[0]["metadata"]
    )