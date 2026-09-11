from app.retrieval.supabase.pgvector_retriever import (
    SupabasePGVectorRetriever
)

from app.retrieval.hybrid.context_builder import (
    HybridContextBuilder
)


print("================================")
print("PGVECTOR → CONTEXT BUILDER TEST")
print("================================")


# ==================================================
# 1. CREATE PGVECTOR RETRIEVER
# ==================================================

print("\nInitializing pgvector retriever...")

retriever = SupabasePGVectorRetriever(
    match_threshold=0.5,
    match_count=5
)


# ==================================================
# 2. QUESTION
# ==================================================

question = "Tell me about Dubai Marina"

print("\nQuestion:")
print(question)


# ==================================================
# 3. RETRIEVE FROM SUPABASE
# ==================================================

print("\nSearching Supabase pgvector...")

community_results = retriever.search(
    question
)

print(
    f"Retrieved {len(community_results)} community records."
)


# ==================================================
# 4. BUILD CONTEXT
# ==================================================

print("\nBuilding context...")

context_builder = HybridContextBuilder()

context = context_builder.build_context(
    structured_summary=None,
    document_results=[],
    community_results=community_results
)


# ==================================================
# 5. DISPLAY CONTEXT
# ==================================================

print("\n================================")
print("GENERATED CONTEXT")
print("================================")

print(context)


# ==================================================
# 6. FINAL
# ==================================================

print("\n================================")
print("TEST COMPLETED")
print("================================")