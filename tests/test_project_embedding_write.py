import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.ingestion.supabase_embedding_writer import (
    SupabaseEmbeddingWriter
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main():

    print("\n================================")
    print("PROJECT EMBEDDING WRITE TEST")
    print("================================")

    # =========================================================
    # SUPABASE CONNECTION
    # =========================================================

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    supabase = create_client(
        supabase_url,
        supabase_key
    )

    # =========================================================
    # GET ONE PROJECT
    # =========================================================

    project_id = 233000800

    print(
        f"\nFetching project ID: {project_id}"
    )

    response = (
        supabase
        .table("projects")
        .select("*")
        .eq("id", project_id)
        .single()
        .execute()
    )

    project = response.data

    if not project:

        print("Project not found.")

        return

    print(
        f"Project: {project.get('name')}"
    )

    print(
        f"Community: {project.get('community')}"
    )

    # =========================================================
    # BUILD TEXT
    # =========================================================

    text = f"""
Project Name: {project.get('name')}

Description:
{project.get('description')}

Developer:
{project.get('developer_name')}

City:
{project.get('city')}

Community:
{project.get('community')}

Sub-community:
{project.get('sub_community')}

Status:
{project.get('status')}

Project Status:
{project.get('project_status')}

Price:
AED {project.get('price')}

Minimum Size:
{project.get('size_min')} sqft

Maximum Size:
{project.get('size_max')} sqft

Minimum Bedrooms:
{project.get('bedroom_min')}

Maximum Bedrooms:
{project.get('bedroom_max')}

Property Types:
{project.get('property_types')}

Amenities:
{project.get('amenities')}

Handover:
{project.get('handover_time')}

Brochure URL:
{project.get('brochure_url')}

Property Finder URL:
{project.get('property_finder_url')}

Source:
{project.get('source')}
""".strip()

    print("\n================================")
    print("TEXT TO EMBED")
    print("================================")

    print(text[:2000])

    # =========================================================
    # EMBEDDING
    # =========================================================

    print("\nGenerating embedding...")

    embedding_model = EmbeddingModel()

    embedding = embedding_model.embed_documents(
        [text]
    )[0]

    print(
        f"Embedding dimension: "
        f"{len(embedding)}"
    )

    if len(embedding) != 384:

        raise ValueError(
            "Embedding dimension is not 384."
        )

    # =========================================================
    # WRITE
    # =========================================================

    print("\nWriting embedding to Supabase...")

    writer = SupabaseEmbeddingWriter()

    writer.update_project_embedding(
        project_id=project_id,
        text=text,
        embedding=embedding
    )

    print("\n================================")
    print("WRITE SUCCESS")
    print("================================")

    print(
        f"Project {project_id} now has "
        f"knowledge_text + embedding."
    )


if __name__ == "__main__":
    main()