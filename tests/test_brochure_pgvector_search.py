from app.retrieval.supabase.pgvector_retriever import (
    SupabasePGVectorRetriever
)


def main():

    print("=" * 60)
    print("BROCHURE PGVECTOR SEARCH TEST")
    print("=" * 60)

    retriever = SupabasePGVectorRetriever(
        match_threshold=0.3,
        match_count=5
    )

    questions = [
        "What security features are available at Sky Edition at Seahaven?",
        "What amenities are available at Sky Edition at Seahaven?",
        "What are the views from Sky Edition at Seahaven?",
        "What interior features are mentioned in the brochure?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        results = retriever.search_documents(
            question=question,
            match_threshold=0.3,
            match_count=5
        )

        print(f"\nResults found: {len(results)}")

        for i, item in enumerate(results, start=1):

            print("\n" + "-" * 60)
            print(f"RESULT {i}")
            print("-" * 60)

            print(
                f"Document: "
                f"{item.get('document_title')}"
            )

            print(
                f"Chunk ID: "
                f"{item.get('chunk_id')}"
            )

            print(
                f"Similarity: "
                f"{item.get('similarity')}"
            )

            print(
                f"URL: "
                f"{item.get('document_url')}"
            )

            print("\nContent:")
            print(
                item.get("content", "")[:1200]
            )

    print("\n" + "=" * 60)
    print("PGVECTOR SEARCH TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()