from app.rag.ingestion.url_fetcher import URLFetcher


def main():

    fetcher = URLFetcher()

    test_url = "https://example.com"

    result = fetcher.fetch(test_url)

    print("\n================================")
    print("FETCH RESULT")
    print("================================")

    print(f"Original URL:")
    print(result["url"])

    print(f"\nFinal URL:")
    print(result["final_url"])

    print(f"\nStatus Code:")
    print(result["status_code"])

    print(f"\nContent Type:")
    print(result["content_type"])

    print(f"\nContent Size:")
    print(len(result["content"]), "bytes")


if __name__ == "__main__":
    main()