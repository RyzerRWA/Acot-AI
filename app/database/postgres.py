"""Single PostgreSQL access point for the ACOT AI service (PostgreSQL access layer)."""
import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg
import psycopg_pool
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
            kwargs={
                "prepare_threshold": None,
                # keep idle connections alive: the hosted database closes quiet ones, and the next
                # question then failed with "server closed the connection unexpectedly"
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
            },
            check=ConnectionPool.check_connection,  # test a connection before handing it out; replace it if dead
            max_idle=int(os.getenv("DB_POOL_MAX_IDLE", "120")),
            max_lifetime=int(os.getenv("DB_POOL_MAX_LIFETIME", "1800")),
            open=True,
        )
    return _pool


def _retry(work):
    """Run work(); if the connection died (server restart, idle timeout), try once more on a fresh one."""
    try:
        return work()
    except (psycopg.OperationalError, psycopg_pool.PoolTimeout):
        return work()


def fetch(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT; each row comes back as a plain JSON-style dict (numbers, ISO-string
    timestamps), the same shapes the previous REST client returned. Embeddings are dropped."""
    def work():
        with _get_pool().connection() as conn:
            # no parameters -> no placeholder parsing, so a literal % (ILIKE '%x%') is safe
            return [row[0] for row in conn.execute(f"select to_jsonb(r) - 'embedding' from ({sql}) r", params or None)]
    return _retry(work)


def execute(sql: str, params: tuple = ()) -> None:
    def work():
        with _get_pool().connection() as conn:
            conn.execute(sql, params or None)
    _retry(work)


def match(function: str, embedding, threshold: float, count: int) -> list[dict]:
    """Vector search via match_communities / match_projects / match_document_chunks."""
    if function not in {"match_communities", "match_projects", "match_document_chunks"}:
        raise ValueError(f"Unknown match function: {function}")

    def work():
        with _get_pool().connection() as conn:
            rows = conn.execute(
                f"select * from public.{function}(%s::real[], %s::float8, %s::int)",
                (list(embedding), threshold, count),
            )
            return [row[0] for row in rows]
    return _retry(work)


def scalar(sql: str, params: tuple = ()):
    """First column of the first row (jsonb comes back as a dict), or None."""
    def work():
        with _get_pool().connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None
    return _retry(work)