"""
Generate missing embeddings for projects, sub_communities, and listings.

Reuses the existing EmbeddingModel (all-MiniLM-L6-v2, 384 dimensions).
Only rows with `embedding IS NULL` are processed, so the script is safe
to re-run. Existing project and sub-community knowledge_text is embedded
as is. Listings get a knowledge_text built from their columns.

Usage:
    python -m app.rag.ingestion.populate_embeddings
    python -m app.rag.ingestion.populate_embeddings projects
    python -m app.rag.ingestion.populate_embeddings listings
"""

import sys

from app.database.postgres_client import PostgresClient
from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.ingestion.embedding_writer import (
    EmbeddingWriter
)


BATCH_SIZE = 50


def populate(table: str, writer, embedding_model) -> int:

    update = {
        "projects": writer.update_project_embedding,
        "sub_communities": writer.update_sub_community_embedding,
    }[table]

    id_argument = {
        "projects": "project_id",
        "sub_communities": "sub_community_id",
    }[table]

    records = (
        writer.db
        .table(table)
        .select("id, name, knowledge_text")
        .is_("embedding", "null")
        .execute()
        .data
    ) or []

    print(f"\n{table}: {len(records)} rows without embeddings")

    stored = 0

    for start in range(0, len(records), BATCH_SIZE):

        batch = [
            record
            for record in records[start:start + BATCH_SIZE]
            if (record.get("knowledge_text") or "").strip()
        ]

        skipped = len(records[start:start + BATCH_SIZE]) - len(batch)

        if skipped:
            print(f"Skipping {skipped} rows with empty knowledge_text")

        if not batch:
            continue

        embeddings = embedding_model.embed_documents(
            [record["knowledge_text"] for record in batch]
        )

        for record, embedding in zip(batch, embeddings):

            update(
                **{id_argument: record["id"]},
                text=record["knowledge_text"],
                embedding=embedding,
            )

            stored += 1

        print(f"{table}: stored {stored}/{len(records)}")

    return stored


LISTING_FIELDS = (
    "listing_id, title, purpose, property_type, category, beds, baths, "
    "size_sqft, price_aed, rent_frequency, furnishing, completion_status, "
    "community, sub_community, location, agency_name"
)


def build_listing_text(row: dict) -> str:

    beds = row.get("beds")
    if beds == 0:
        bedrooms = "studio"
    elif beds is not None:
        bedrooms = f"{beds} bedrooms"
    else:
        bedrooms = ""

    price = row.get("price_aed")
    price_text = ""
    if price is not None:
        price_text = f"AED {int(float(price))} {row.get('rent_frequency') or ''}".strip()

    size = row.get("size_sqft")
    fields = [
        ("Listing", row.get("title")),
        ("Purpose", row.get("purpose")),
        ("Type", row.get("property_type")),
        ("Category", row.get("category")),
        ("Bedrooms", bedrooms),
        ("Bathrooms", row.get("baths")),
        ("Size", f"{size} sqft" if size else ""),
        ("Rent", price_text),
        ("Furnishing", row.get("furnishing")),
        ("Status", row.get("completion_status")),
        ("Community", row.get("community")),
        ("Sub-community", row.get("sub_community")),
        ("Location", row.get("location")),
        ("Agency", row.get("agency_name")),
    ]

    lines = []
    for label, value in fields:
        text = "" if value is None else str(value).strip()
        if text:
            lines.append(f"{label}: {text}")

    return ". ".join(lines)


def ensure_listing_embedding_columns(client: PostgresClient) -> None:

    with client.pool.connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                ALTER TABLE public.listings
                    ADD COLUMN IF NOT EXISTS knowledge_text text;
                ALTER TABLE public.listings
                    ADD COLUMN IF NOT EXISTS embedding real[];

                CREATE OR REPLACE FUNCTION public.match_listings(
                    query_embedding real[],
                    match_threshold double precision DEFAULT 0.5,
                    match_count integer DEFAULT 12)
                RETURNS SETOF jsonb
                LANGUAGE sql STABLE
                AS $function$
                  select (to_jsonb(s) - 'embedding' - 'sim')
                         || jsonb_build_object('similarity', sim)
                  from (
                    select l.*, public.cosine_sim(l.embedding, query_embedding) as sim
                    from public.listings l
                    where l.embedding is not null
                  ) s
                  where sim > match_threshold
                  order by sim desc
                  limit match_count
                $function$;
                """
            )


def populate_listings(client: PostgresClient, embedding_model) -> int:

    ensure_listing_embedding_columns(client)

    records = (
        client
        .table("listings")
        .select(LISTING_FIELDS)
        .is_("embedding", "null")
        .execute()
        .data
    ) or []

    print(f"\nlistings: {len(records)} rows without embeddings")

    prepared = []
    for record in records:
        text = build_listing_text(record)
        if text:
            prepared.append((record["listing_id"], text))

    stored = 0

    for start in range(0, len(prepared), BATCH_SIZE):

        batch = prepared[start:start + BATCH_SIZE]
        embeddings = embedding_model.embed_documents(
            [text for _, text in batch]
        )

        rows = []
        for (listing_id, text), embedding in zip(batch, embeddings):
            vector = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )
            if len(vector) != 384:
                raise ValueError(
                    f"Expected 384-dimensional embedding, got {len(vector)}."
                )
            rows.append((text, vector, listing_id))

        with client.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    UPDATE public.listings
                    SET knowledge_text = %s,
                        embedding = %s::real[]
                    WHERE listing_id = %s
                    """,
                    rows,
                )

        stored += len(rows)
        print(f"listings: stored {stored}/{len(prepared)}")

    return stored


def main():

    tables = sys.argv[1:] or ["projects", "sub_communities"]

    for table in tables:
        if table not in ("projects", "sub_communities", "listings"):
            raise SystemExit(
                f"Unsupported table: {table}. "
                "Use projects, sub_communities, or listings."
            )

    print("\n================================")
    print("EMBEDDING INGESTION")
    print("================================")

    embedding_model = EmbeddingModel()
    writer = EmbeddingWriter()
    client = PostgresClient()

    for table in tables:
        if table == "listings":
            populate_listings(client, embedding_model)
        else:
            populate(table, writer, embedding_model)

    print("\n================================")
    print("EMBEDDING INGESTION COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()
