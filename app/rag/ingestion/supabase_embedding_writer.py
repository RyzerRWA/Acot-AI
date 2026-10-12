from app.database.postgres import execute


# =========================================================
# POSTGRES EMBEDDING WRITER
# =========================================================

class SupabaseEmbeddingWriter:

    # =========================================================
    # PROJECT
    # =========================================================

    def update_project_embedding(
        self,
        project_id: int,
        text: str,
        embedding
    ):

        if not text:
            raise ValueError(
                "Project text cannot be empty."
            )

        if embedding is None:
            raise ValueError(
                "Project embedding cannot be None."
            )

        embedding_list = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        if len(embedding_list) != 384:
            raise ValueError(
                f"Expected 384-dimensional embedding, "
                f"got {len(embedding_list)}."
            )

        execute(
            "update projects set knowledge_text = %s, embedding = %s::real[] where id = %s",
            (text, embedding_list, project_id),
        )

    # =========================================================
    # COMMUNITY
    # =========================================================

    def update_community_embedding(
        self,
        community_id: int,
        text: str,
        embedding
    ):

        if not text:
            raise ValueError(
                "Community text cannot be empty."
            )

        if embedding is None:
            raise ValueError(
                "Community embedding cannot be None."
            )

        embedding_list = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        if len(embedding_list) != 384:
            raise ValueError(
                f"Expected 384-dimensional embedding, "
                f"got {len(embedding_list)}."
            )

        execute(
            "update communities set knowledge_text = %s, embedding = %s::real[] where id = %s",
            (text, embedding_list, community_id),
        )

    # =========================================================
    # SUB-COMMUNITY
    # =========================================================

    def update_sub_community_embedding(
        self,
        sub_community_id: int,
        text: str,
        embedding
    ):

        if not text:
            raise ValueError(
                "Sub-community text cannot be empty."
            )

        if embedding is None:
            raise ValueError(
                "Sub-community embedding cannot be None."
            )

        embedding_list = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        if len(embedding_list) != 384:
            raise ValueError(
                f"Expected 384-dimensional embedding, "
                f"got {len(embedding_list)}."
            )

        execute(
            "update sub_communities set knowledge_text = %s, embedding = %s::real[] where id = %s",
            (text, embedding_list, sub_community_id),
        )

    # =========================================================
    # DOCUMENT CHUNK
    # =========================================================

    def insert_document_chunk(
        self,
        document_id,
        document_title,
        document_url,
        document_type,
        publisher,
        chunk_id,
        content,
        embedding
    ):

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        if not document_id:
            raise ValueError(
                "document_id is required."
            )

        if not document_url:
            raise ValueError(
                "document_url is required."
            )

        if not content:
            raise ValueError(
                "Document content cannot be empty."
            )

        if embedding is None:
            raise ValueError(
                "Document embedding cannot be None."
            )

        # -----------------------------------------------------
        # Convert embedding to Python list
        # -----------------------------------------------------

        embedding_list = (
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        # -----------------------------------------------------
        # Validate embedding dimension
        # -----------------------------------------------------

        if len(embedding_list) != 384:
            raise ValueError(
                f"Expected 384-dimensional embedding, "
                f"got {len(embedding_list)}."
            )

        # -----------------------------------------------------
        # Insert into Supabase document_chunks
        # -----------------------------------------------------

        execute(
            """
            insert into document_chunks
                (document_id, document_title, document_url, document_type,
                 publisher, chunk_id, content, embedding)
            values (%s, %s, %s, %s, %s, %s, %s, %s::real[])
            """,
            (
                document_id, document_title, document_url, document_type,
                publisher, chunk_id, content, embedding_list,
            ),
        )
