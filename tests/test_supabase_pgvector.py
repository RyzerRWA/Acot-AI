import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from app.rag.embeddings.embedding_model import EmbeddingModel


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")


if not SUPABASE_URL:
    raise ValueError(
        f"SUPABASE_URL is missing. Checked .env at:\n{ENV_FILE}"
    )

if not SUPABASE_KEY:
    raise ValueError(
        f"SUPABASE_SECRET_KEY is missing. Checked .env at:\n{ENV_FILE}"
    )


print("================================")
print("SUPABASE PGVECTOR TEST")
print("================================")

print("\nSupabase environment loaded successfully.")


# ============================================================
# 2. CREATE SUPABASE CLIENT
# ============================================================

print("\nConnecting to Supabase...")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("Supabase connection initialized.")


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
# Supabase RPC requires a normal Python list.
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
# 7. SEARCH SUPABASE PGVECTOR
# ============================================================

print("\nSearching Supabase pgvector...")

try:

    response = supabase.rpc(
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