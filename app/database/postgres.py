"""Single PostgreSQL access point for the ACOT AI service (PostgreSQL access layer)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is missing")
        _pool = ConnectionPool(
            url,
            min_size=1,
            max_size=int(os.getenv("DB_POOL_MAX", "10")),
            kwargs={"prepare_threshold": None},
            open=True,
        )
    return _pool


def fetch(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT; each row comes back as a plain JSON-style dict (numbers, ISO-string
    timestamps), the same shapes the previous REST client returned. Embeddings are dropped."""
    with _get_pool().connection() as conn:
        # no parameters -> no placeholder parsing, so a literal % (ILIKE '%x%') is safe
        return [row[0] for row in conn.execute(f"select to_jsonb(r) - 'embedding' from ({sql}) r", params or None)]


def execute(sql: str, params: tuple = ()) -> None:
    with _get_pool().connection() as conn:
        conn.execute(sql, params or None)


def match(function: str, embedding, threshold: float, count: int) -> list[dict]:
    """Vector search via match_communities / match_projects / match_document_chunks."""
    if function not in {"match_communities", "match_projects", "match_document_chunks"}:
        raise ValueError(f"Unknown match function: {function}")
    with _get_pool().connection() as conn:
        rows = conn.execute(
            f"select * from public.{function}(%s::real[], %s::float8, %s::int)",
            (list(embedding), threshold, count),
        )
        return [row[0] for row in rows]


def scalar(sql: str, params: tuple = ()):
    """First column of the first row (jsonb comes back as a dict), or None."""
    with _get_pool().connection() as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
