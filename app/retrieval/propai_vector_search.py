"""Nearest-name search over the Prop-AI pgvector columns.

Uses the same MiniLM model that filled the columns. Sales and rents are not searched here.
"""
from __future__ import annotations

from app.database.postgres import fetch
from app.rag.embeddings.embedding_model import EmbeddingModel

_SQL = {
    "places": """
        select level, label, area, master,
               round((1 - (embedding <=> %s::vector))::numeric, 3) as score
        from propai_places
        where embedding is not null
        order by embedding <=> %s::vector
        limit %s
    """,
    "raw": """
        select row_key,
               payload->>'project_name_en' as name,
               payload->>'area_name_label' as area,
               payload->>'developer_name' as developer,
               round((1 - (embedding <=> %s::vector))::numeric, 3) as score
        from propai_raw
        where dataset = 'projects' and embedding is not null
        order by embedding <=> %s::vector
        limit %s
    """,
    "status": """
        select project_number,
               payload->>'project_name_en' as name,
               payload->>'developer_name' as developer,
               payload->>'development_status' as status,
               round((1 - (embedding <=> %s::vector))::numeric, 3) as score
        from propai_project_status
        where embedding is not null
        order by embedding <=> %s::vector
        limit %s
    """,
    "brokers": """
        select brn, name_en, agency_name,
               round((1 - (embedding <=> %s::vector))::numeric, 3) as score
        from propai_brokers
        where embedding is not null
        order by embedding <=> %s::vector
        limit %s
    """,
}


def _vector(values) -> str:
    numbers = [float(value) for value in list(values)]
    if len(numbers) != 384:
        raise RuntimeError(f"Expected 384 dimensions, got {len(numbers)}.")
    return "[" + ",".join(f"{number:.7f}" for number in numbers) + "]"


def search_places(query: str, limit: int = 5) -> list[dict]:
    """Closest area or master-project names. Buildings are left out so a place search stays on the place."""
    text = (query or "").strip()
    if not text:
        return []
    embedded = EmbeddingModel().embed_query(text)
    vector = _vector(embedded)
    rows = fetch(
        """
        select level, label, area, master,
               round((1 - (embedding <=> %s::vector))::numeric, 3) as score
        from propai_places
        where embedding is not null and level in ('area', 'master')
        order by embedding <=> %s::vector
        limit %s
        """,
        (vector, vector, max(1, min(int(limit), 20))),
    )
    for row in rows:
        if row.get("score") is not None:
            row["score"] = float(row["score"])
    return rows


def search_propai(query: str, table: str = "places", limit: int = 5) -> list[dict]:
    """Return the closest rows in one embedded table: places, raw, status, or brokers."""
    text = (query or "").strip()
    if not text or table not in _SQL:
        return []
    embedded = EmbeddingModel().embed_query(text)
    vector = _vector(embedded)
    rows = fetch(_SQL[table], (vector, vector, max(1, min(int(limit), 20))))
    for row in rows:
        if row.get("score") is not None:
            row["score"] = float(row["score"])
    return rows
