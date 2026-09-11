from app.rag.embeddings.embedding_model import EmbeddingModel


def main():

    print("\n================================")
    print("EMBEDDING MODEL TEST")
    print("================================")

    # =========================================================
    # LOAD MODEL
    # =========================================================

    print("\nLoading embedding model...")

    model = EmbeddingModel()

    print("Embedding model loaded.")

    # =========================================================
    # TEST DOCUMENT CHUNKS
    # =========================================================

    chunks = [
        "Dubai Marina is a waterfront community in Dubai.",
        "Sky Edition at Seahaven is a residential project by Sobha.",
        "The project offers apartments with 3 to 4 bedrooms."
    ]

    print("\nGenerating document embeddings...")

    embeddings = model.embed_documents(chunks)

    print("\n================================")
    print("DOCUMENT EMBEDDING RESULT")
    print("================================")

    print(f"Number of chunks: {len(chunks)}")

    print(
        f"Number of embeddings: "
        f"{len(embeddings)}"
    )

    for index, embedding in enumerate(
        embeddings,
        start=1
    ):

        print(f"\nEmbedding {index}")

        print(
            f"Dimension: {len(embedding)}"
        )

        print(
            "First 10 values:"
        )

        print(
            embedding[:10]
        )

    # =========================================================
    # TEST QUERY EMBEDDING
    # =========================================================

    query = "Tell me about projects in Dubai Marina."

    print("\n================================")
    print("QUERY EMBEDDING TEST")
    print("================================")

    print(f"Query: {query}")

    query_embedding = model.embed_query(
        query
    )

    print(
        f"\nQuery embedding dimension: "
        f"{len(query_embedding)}"
    )

    print(
        "First 10 values:"
    )

    print(
        query_embedding[:10]
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    print("\n================================")
    print("VALIDATION")
    print("================================")

    expected_dimension = 384

    all_correct = True

    for embedding in embeddings:

        if len(embedding) != expected_dimension:

            all_correct = False

    if len(query_embedding) != expected_dimension:

        all_correct = False

    if all_correct:

        print(
            "SUCCESS: All embeddings are "
            "384-dimensional."
        )

    else:

        print(
            "ERROR: Embedding dimension "
            "does not match 384."
        )

    print("\n================================")
    print("EMBEDDING TEST COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()