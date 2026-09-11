import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from app.rag.ingestion.project_text_builder import (
    ProjectTextBuilder
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main():

    print("\n================================")
    print("PROJECT TEXT BUILDER TEST")
    print("================================")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    supabase = create_client(
        supabase_url,
        supabase_key
    )

    project_id = 233000800

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

    text = ProjectTextBuilder.build(
        project
    )

    print("\n================================")
    print("GENERATED PROJECT TEXT")
    print("================================")

    print(text)

    print("\n================================")
    print("TEXT BUILDER TEST COMPLETE")
    print("================================")

    print(
        f"Characters: {len(text)}"
    )


if __name__ == "__main__":
    main()