from app.rag.ingestion.source_detector import SourceDetector
from app.rag.ingestion.url_fetcher import URLFetcher
from app.rag.ingestion.content_extractor import ContentExtractor
from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.ingestion.chunker import TextChunker
from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.ingestion.supabase_embedding_writer import SupabaseEmbeddingWriter


URL = (
    "https://pf-ae-documents.s3.ap-southeast-1.amazonaws.com/"
    "new-project/brochure-994e4dad.pdf"
)

PROJECT_NAME = "Sky Edition at Seahaven"


def main():

    print("=" * 60)
    print("SUPABASE BROCHURE INGESTION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Detect source
    # ---------------------------------------------------------

    source_type = SourceDetector.detect(URL)

    print(f"\nSource Type: {source_type}")

    # ---------------------------------------------------------
    # 2. Fetch
    # ---------------------------------------------------------

    fetcher = URLFetcher()

    response = fetcher.fetch(URL)

    print(f"HTTP Status: {response['status_code']}")
    print(f"Content Type: {response['content_type']}")

    # ---------------------------------------------------------
    # 3. Extract
    # ---------------------------------------------------------

    extractor = ContentExtractor()

    text = extractor.extract(
        content=response["content"],
        content_type=response["content_type"],
        source_type=source_type
    )

    # ---------------------------------------------------------
    # 4. Clean
    # ---------------------------------------------------------

    cleaner = TextCleaner()

    cleaned_text = cleaner.clean(text)

    print(f"\nCleaned characters: {len(cleaned_text)}")

    # ---------------------------------------------------------
    # 5. Chunk
    # ---------------------------------------------------------

    chunker = TextChunker(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = chunker.chunk(cleaned_text)

    print(f"Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 6. Generate embeddings
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    print("Generating embeddings...")

    embeddings = embedding_model.embed_documents(
        chunks
    )

    print(
        f"Generated embeddings: {len(embeddings)}"
    )

    print(
        f"Embedding dimensions: "
        f"{len(embeddings[0])}"
    )

    # ---------------------------------------------------------
    # 7. Connect Supabase
    # ---------------------------------------------------------

    print("\nConnecting to Supabase...")

    writer = SupabaseEmbeddingWriter()

    # ---------------------------------------------------------
    # 8. Remove previous brochure chunks
    # ---------------------------------------------------------

    print("\nRemoving previous test ingestion...")

    delete_response = (
        writer.supabase
        .table("document_chunks")
        .delete()
        .eq("document_url", URL)
        .execute()
    )

    print("Previous chunks removed.")

    # ---------------------------------------------------------
    # 9. Insert chunks
    # ---------------------------------------------------------

    print("\nInserting chunks into Supabase...")

    inserted = 0

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings),
        start=1
    ):

        writer.insert_document_chunk(
            document_id="sky-edition-at-seahaven",
            document_title=PROJECT_NAME,
            document_url=URL,
            document_type="pdf",
            publisher="Sobha",
            chunk_id=index,
            content=chunk,
            embedding=embedding
        )

        inserted += 1

        print(
            f"Inserted chunk {index}/{len(chunks)}"
        )

    # ---------------------------------------------------------
    # 10. Verify
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("INGESTION RESULT")
    print("=" * 60)

    print(f"Chunks generated: {len(chunks)}")
    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Chunks inserted: {inserted}")

    # Query database
    result = (
        writer.supabase
        .table("document_chunks")
        .select(
            "id, document_id, document_title, "
            "document_url, document_type, publisher, "
            "chunk_id, content"
        )
        .eq(
            "document_id",
            "sky-edition-at-seahaven"
        )
        .order("chunk_id")
        .execute()
    )

    rows = result.data or []

    print(
        f"Chunks found in Supabase: {len(rows)}"
    )

    if rows:

        print("\nFirst stored chunk:")
        print("-" * 60)
        print(rows[0]["content"][:1000])
        print("-" * 60)

    print("\n" + "=" * 60)
    print("SUPABASE INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()