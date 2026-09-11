from app.rag.ingestion.source_detector import SourceDetector


def main():

    urls = [
        "https://example.com/document.pdf",

        "https://example.com/data.json",

        "https://example.com/api/projects",

        "https://api.example.com/projects",

        "https://www.propertyfinder.ae/en/new-projects/example"
    ]

    print("\n==============================")
    print("SOURCE DETECTOR TEST")
    print("==============================")

    for url in urls:

        source_type = SourceDetector.detect(url)

        print(f"\nURL:")
        print(url)

        print(f"Type:")
        print(source_type)


if __name__ == "__main__":
    main()