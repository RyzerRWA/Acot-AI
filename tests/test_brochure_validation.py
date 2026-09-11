from app.rag.ingestion.source_detector import SourceDetector
from app.rag.ingestion.url_fetcher import URLFetcher
from app.rag.ingestion.content_extractor import ContentExtractor
from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.ingestion.document_validator import DocumentValidator


URL = (
    "https://pf-ae-documents.s3.ap-southeast-1.amazonaws.com/"
    "new-project/brochure-994e4dad.pdf"
)

PROJECT_NAME = "Sky Edition at Seahaven"


def main():

    print("=" * 60)
    print("BROCHURE VALIDATION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Detect source
    # ---------------------------------------------------------

    source_type = SourceDetector.detect(URL)

    print(f"\nSource Type: {source_type}")

    # ---------------------------------------------------------
    # 2. Download
    # ---------------------------------------------------------

    print("\nDownloading document...")

    fetcher = URLFetcher()

    response = fetcher.fetch(URL)

    print(f"HTTP Status: {response['status_code']}")
    print(f"Content Type: {response['content_type']}")
    print(f"Final URL: {response['final_url']}")

    # ---------------------------------------------------------
    # 3. Extract text
    # ---------------------------------------------------------

    print("\nExtracting text...")

    extractor = ContentExtractor()

    text = extractor.extract(
        content=response["content"],
        content_type=response["content_type"],
        source_type=source_type
    )

    print(f"Extracted Characters: {len(text)}")
    print(f"Extracted Words: {len(text.split())}")

    # ---------------------------------------------------------
    # 4. Clean text
    # ---------------------------------------------------------

    print("\nCleaning text...")

    cleaner = TextCleaner()

    cleaned_text = cleaner.clean(text)

    print(f"Cleaned Characters: {len(cleaned_text)}")
    print(f"Cleaned Words: {len(cleaned_text.split())}")

    # ---------------------------------------------------------
    # 5. Validate
    # ---------------------------------------------------------

    print("\nValidating document...")

    validation = DocumentValidator.validate(
        cleaned_text,
        project_name=PROJECT_NAME
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    for key, value in validation.items():
        print(f"{key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()