from app.rag.embeddings.embedding_model import EmbeddingModel

from app.database.postgres import match


class PGVectorRetriever:
    """
    Semantic retriever over PostgreSQL (cosine similarity in SQL).

    Supports:
        1. Community semantic search
        2. Document chunk semantic search

    Embedding dimension:
        384

    PostgreSQL functions:
        match_communities
        match_document_chunks
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_count: int = 5
    ):

        self.embedding_model = EmbeddingModel()

        self.match_threshold = match_threshold
        self.match_count = match_count

    # =========================================================
    # CREATE QUERY EMBEDDING
    # =========================================================

    def create_query_embedding(
        self,
        question: str
    ):

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        embedding = self.embedding_model.embed_query(
            question
        )

        # -----------------------------------------------------
        # Convert NumPy array to Python list
        # -----------------------------------------------------

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # -----------------------------------------------------
        # Validate dimension
        # -----------------------------------------------------

        if len(embedding) != 384:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected 384, received {len(embedding)}."
            )

        return embedding

    # =========================================================
    # SEARCH COMMUNITIES
    # =========================================================

    def search(
        self,
        question: str,
        match_threshold: float = None,
        match_count: int = None
    ):

        if not question or not question.strip():
            return []

        # -----------------------------------------------------
        # Use supplied values or defaults
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Create query embedding
        # -----------------------------------------------------

        # Retired: old `communities` embeddings / match_communities RPC.
        _ = threshold, count, question
        return []

    # =========================================================
    # SEARCH DOCUMENT CHUNKS
    # =========================================================

    def search_documents(
        self,
        question: str,
        match_threshold: float = None,
        match_count: int = None
    ):
        """Retired: old `document_chunks` / match_document_chunks are not used."""

        _ = question, match_threshold, match_count
        return []