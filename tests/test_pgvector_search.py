from pathlib import Path

from dotenv import load_dotenv

from app.database.postgres_client import PostgresClient
from app.rag.embeddings.embedding_model import EmbeddingModel


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

print("================================")
print("POSTGRESQL PGVECTOR TEST")
print("================================")

print("\nEnvironment loaded successfully.")


# ============================================================
# 2. CREATE POSTGRES CLIENT
# ============================================================

print("\nConnecting to PostgreSQL...")

db = PostgresClient()

print("PostgreSQL connection initialized.")


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = EmbeddingModel()

print("Embedding model loaded.")


# ============================================================
# 4. TEST QUESTION
# ============================================================

question = "Tell me about Dubai Marina"

print("\nQuestion:")
print(question)


# ============================================================
# 5. CREATE QUERY EMBEDDING
# ============================================================

print("\nCreating query embedding...")

query_embedding = embedding_model.embed_query(question)


# IMPORTANT:
# The embedding model returns a NumPy ndarray.
# PostgreSQL RPC requires a normal Python list.
if hasattr(query_embedding, "tolist"):
    query_embedding = query_embedding.tolist()


print("\nEmbedding dimension:")
print(len(query_embedding))

print("Embedding type:")
print(type(query_embedding))


# ============================================================
# 6. VALIDATE EMBEDDING
# ============================================================

if len(query_embedding) != 384:
    raise ValueError(
        f"Expected embedding dimension 384, "
        f"but received {len(query_embedding)}"
    )

print("\nEmbedding dimension is correct: 384")


# ============================================================
# 7. SEARCH POSTGRES PGVECTOR
# ============================================================

print("\nSearching PostgreSQL pgvector...")

try:

    response = db.rpc(
        "match_communities",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": 5
        }
    ).execute()

except Exception as e:

    print("\n================================")
    print("PGVECTOR SEARCH ERROR")
    print("================================")

    print(type(e).__name__)
    print(str(e))

    raise


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

results = response.data

print("\n================================")
print("SEARCH RESULTS")
print("================================")


if not results:

    print("\nNo matching communities found.")

    print("\nPossible reasons:")
    print("1. communities.embedding contains no embeddings.")
    print("2. match_threshold is too high.")
    print("3. match_communities RPC has no matching records.")
    print("4. Existing embeddings use a different model.")

else:

    print(f"\nFound {len(results)} matching communities:\n")

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


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("================================")
print("TEST COMPLETED")
print("================================")