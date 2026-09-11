from app.rag.ingestion.source_detector import SourceDetector
from app.rag.ingestion.url_fetcher import URLFetcher
from app.rag.ingestion.content_extractor import ContentExtractor
from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.ingestion.chunker import TextChunker
from app.rag.embeddings.embedding_model import EmbeddingModel


URL = (
    "https://pf-ae-documents.s3.ap-southeast-1.amazonaws.com/"
    "new-project/brochure-994e4dad.pdf"
)


def main():

    print("=" * 60)
    print("BROCHURE EMBEDDING TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Fetch
    # ---------------------------------------------------------

    source_type = SourceDetector.detect(URL)

    fetcher = URLFetcher()
    response = fetcher.fetch(URL)

    # ---------------------------------------------------------
    # Extract
    # ---------------------------------------------------------

    extractor = ContentExtractor()

    text = extractor.extract(
        content=response["content"],
        content_type=response["content_type"],
        source_type=source_type
    )

    # ---------------------------------------------------------
    # Clean
    # ---------------------------------------------------------

    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(text)

    # ---------------------------------------------------------
    # Chunk
    # ---------------------------------------------------------

    chunker = TextChunker(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = chunker.chunk(cleaned_text)

    print(f"\nChunks created: {len(chunks)}")

    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    print("Generating embeddings...")

    embeddings = embedding_model.embed_documents(
        chunks
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EMBEDDING RESULT")
    print("=" * 60)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")

    if len(embeddings) > 0:

        print(
            f"Embedding dimensions: "
            f"{len(embeddings[0])}"
        )

        print(
            f"First embedding type: "
            f"{type(embeddings[0])}"
        )

        print("\nFirst 10 values:")

        print(
            embeddings[0][:10]
        )

    print("=" * 60)
    print("EMBEDDING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()