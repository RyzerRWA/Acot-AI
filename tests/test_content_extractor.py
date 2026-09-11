from app.rag.ingestion.url_fetcher import URLFetcher
from app.rag.ingestion.content_extractor import ContentExtractor
from app.rag.ingestion.source_detector import SourceDetector


def main():

    url = "https://example.com"

    print("\n================================")
    print("CONTENT EXTRACTOR TEST")
    print("================================")

    # ---------------------------------------------------------
    # Detect source
    # ---------------------------------------------------------

    source_type = SourceDetector.detect(url)

    print(f"\nSource Type: {source_type}")

    # ---------------------------------------------------------
    # Fetch
    # ---------------------------------------------------------

    fetcher = URLFetcher()

    result = fetcher.fetch(url)

    # ---------------------------------------------------------
    # Extract
    # ---------------------------------------------------------

    extractor = ContentExtractor()

    text = extractor.extract(
        content=result["content"],
        content_type=result["content_type"],
        source_type=source_type
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print("\n================================")
    print("EXTRACTED TEXT")
    print("================================")

    print(text[:2000])

    print("\n================================")
    print("EXTRACTION COMPLETE")
    print("================================")

    print(
        f"Total extracted characters: "
        f"{len(text)}"
    )


if __name__ == "__main__":
    main()