from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.ingestion.chunker import TextChunker


def main():

    sample_text = """
    Dubai Marina is a waterfront community in Dubai.

    
    It contains residential developments, commercial areas,
    restaurants, amenities and transportation facilities.



        The community includes various real estate projects.
    """

    print("\n================================")
    print("CLEANER + CHUNKER TEST")
    print("================================")

    # =========================================================
    # CLEAN
    # =========================================================

    cleaner = TextCleaner()

    cleaned_text = cleaner.clean(
        sample_text
    )

    print("\nCLEANED TEXT:")
    print("----------------------------")
    print(cleaned_text)

    # =========================================================
    # CHUNK
    # =========================================================

    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = chunker.chunk(
        cleaned_text
    )

    print("\n================================")
    print("GENERATED CHUNKS")
    print("================================")

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nCHUNK {index}"
        )

        print("----------------------------")

        print(chunk)

        print(
            f"\nCharacters: {len(chunk)}"
        )

    print("\n================================")
    print("TEST COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()