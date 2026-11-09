"""
Add an empty pgvector column to the Prop-AI tables that are searched by name or text.

Does not write any embedding values and does not touch propai_sales or propai_rents.
Those stay numeric tables. 384 dimensions matches all-MiniLM-L6-v2.

Requires: CREATE EXTENSION vector;  (a superuser, once per database)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.postgres import execute, fetch


TABLES = (
    "propai_places",
    "propai_raw",
    "propai_brokers",
    "propai_project_status",
)


def main():
    execute("create extension if not exists vector")
    for table in TABLES:
        execute(
            f"alter table public.{table} add column if not exists embedding vector(384)"
        )
    rows = fetch(
        """
        select c.relname as table, format_type(a.atttypid, a.atttypmod) as type
        from pg_attribute a
        join pg_class c on c.oid = a.attrelid
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = 'public' and a.attname = 'embedding' and not a.attisdropped
          and c.relname like 'propai_%'
        order by 1
        """
    )
    for row in rows:
        print(f"{row['table']}.embedding  {row['type']}")


if __name__ == "__main__":
    main()
