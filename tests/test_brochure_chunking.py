from app.rag.ingestion.source_detector import SourceDetector
from app.rag.ingestion.url_fetcher import URLFetcher
from app.rag.ingestion.content_extractor import ContentExtractor
from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.ingestion.chunker import TextChunker


URL = (
    "https://pf-ae-documents.s3.ap-southeast-1.amazonaws.com/"
    "new-project/brochure-994e4dad.pdf"
)


def main():

    print("=" * 60)
    print("BROCHURE CHUNKING TEST")
    print("=" * 60)

    # 1. Detect source
    source_type = SourceDetector.detect(URL)

    # 2. Download
    fetcher = URLFetcher()
    response = fetcher.fetch(URL)

    # 3. Extract
    extractor = ContentExtractor()

    text = extractor.extract(
        content=response["content"],
        content_type=response["content_type"],
        source_type=source_type
    )

    # 4. Clean
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(text)

    print(f"\nCleaned text: {len(cleaned_text)} characters")
    print(f"Words: {len(cleaned_text.split())}")

    # 5. Chunk
    chunker = TextChunker(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = chunker.chunk(cleaned_text)

    print(f"\nTotal chunks: {len(chunks)}")

    print("\n" + "=" * 60)
    print("FIRST 3 CHUNKS")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:3], start=1):

        print(f"\n--- CHUNK {i} ---")
        print(f"Characters: {len(chunk)}")
        print(chunk[:1000])

    print("\n" + "=" * 60)
    print("LAST CHUNK")
    print("=" * 60)

    last_chunk = chunks[-1]

    print(f"Characters: {len(last_chunk)}")
    print(last_chunk)

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()