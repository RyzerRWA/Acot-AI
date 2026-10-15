from app.database.postgres_client import PostgresClient


# ============================================================
# POSTGRESQL CLIENT
# ============================================================
# The retrieval layer now talks to the new PostgreSQL database
# (DATABASE_URL) instead of Supabase. The module-level name
# `supabase` is kept so existing imports keep working.

supabase = PostgresClient()
