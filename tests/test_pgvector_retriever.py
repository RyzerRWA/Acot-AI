from app.retrieval.supabase.pgvector_retriever import (
    SupabasePGVectorRetriever
)


print("================================")
print("PGVECTOR RETRIEVER TEST")
print("================================")


# ============================================================
# CREATE RETRIEVER
# ============================================================

print("\nInitializing Supabase PGVector Retriever...")

retriever = SupabasePGVectorRetriever(
    match_threshold=0.5,
    match_count=5
)

print("Retriever initialized.")


# ============================================================
# TEST QUESTION
# ============================================================

question = "Tell me about Dubai Marina"

print("\nQuestion:")
print(question)


# ============================================================
# SEARCH
# ============================================================

print("\nSearching Supabase pgvector...")

results = retriever.search(question)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n================================")
print("RESULTS")
print("================================")


if not results:

    print("\nNo results found.")

else:

    print(f"\nFound {len(results)} results.\n")

    for index, result in enumerate(results, start=1):

        print(f"Result {index}")
        print("----------------------------")

        print("ID:")
        print(result.get("id"))

        print("Name:")
        print(result.get("name"))

        print("City:")
        print(result.get("city"))

        print("Similarity:")
        print(result.get("similarity"))

        print("Source:")
        print(result.get("source"))

        print()


print("================================")
print("TEST COMPLETED")
print("================================")