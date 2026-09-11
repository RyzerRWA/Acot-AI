import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


# =========================================================
# PROJECT ROOT / ENVIRONMENT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


# =========================================================
# SUPABASE EMBEDDING WRITER
# =========================================================

class SupabaseEmbeddingWriter:

    def __init__(self):

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SECRET_KEY")

        if not supabase_url:
            raise ValueError(
                "SUPABASE_URL is not configured."
            )

        if not supabase_key:
            raise ValueError(
                "SUPABASE_SECRET_KEY is not configured."
            )

        self.supabase = create_client(
            supabase_url,
            supabase_key
        )

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

        response = (
            self.supabase
            .table("projects")
            .update({
                "knowledge_text": text,
                "embedding": embedding_list
            })
            .eq("id", project_id)
            .execute()
        )

        return response

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

        response = (
            self.supabase
            .table("communities")
            .update({
                "knowledge_text": text,
                "embedding": embedding_list
            })
            .eq("id", community_id)
            .execute()
        )

        return response

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

        response = (
            self.supabase
            .table("sub_communities")
            .update({
                "knowledge_text": text,
                "embedding": embedding_list
            })
            .eq("id", sub_community_id)
            .execute()
        )

        return response

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

        response = (
            self.supabase
            .table("document_chunks")
            .insert({
                "document_id": document_id,
                "document_title": document_title,
                "document_url": document_url,
                "document_type": document_type,
                "publisher": publisher,
                "chunk_id": chunk_id,
                "content": content,
                "embedding": embedding_list
            })
            .execute()
        )

        return response