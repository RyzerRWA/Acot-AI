from pathlib import Path

from dotenv import load_dotenv

from app.database.postgres_client import PostgresClient
from app.rag.embeddings.embedding_model import EmbeddingModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class SupabaseProjectVectorRetriever:

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_count: int = 5
    ):

        self.supabase = PostgresClient()

        self.embedding_model = EmbeddingModel()

        self.match_threshold = match_threshold
        self.match_count = match_count

    # =========================================================
    # PROJECT SEMANTIC SEARCH
    # =========================================================

    def search(
        self,
        question: str,
        match_threshold: float = None,
        match_count: int = None
    ):

        if not question:
            return []

        threshold = (
            match_threshold
            if match_threshold is not None
            else self.match_threshold
        )

        count = (
            match_count
            if match_count is not None
            else self.match_count
        )

        print("\n================================")
        print("SUPABASE PROJECT VECTOR SEARCH")
        print("================================")

        print(f"Question: {question}")
        print(f"Threshold: {threshold}")
        print(f"Match count: {count}")

        # -----------------------------------------------------
        # Generate query embedding
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_model.embed_query(
                question
            )
        )

        print(
            f"Query embedding dimension: "
            f"{len(query_embedding)}"
        )

        if len(query_embedding) != 384:

            raise ValueError(
                "Query embedding dimension must be 384."
            )

        # -----------------------------------------------------
        # Call Supabase RPC
        # -----------------------------------------------------

        response = self.supabase.rpc(
            "match_projects",
            {
                "query_embedding": query_embedding.tolist(),
                "match_threshold": threshold,
                "match_count": count
            }
        ).execute()

        results = response.data or []

        print(
            f"Projects retrieved: {len(results)}"
        )

        return results