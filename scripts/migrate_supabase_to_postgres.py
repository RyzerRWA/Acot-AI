"""
One-off: copy ACOT AI data from Supabase into plain PostgreSQL (full refresh, re-runnable).

  set -a; . ./.env; set +a          # SUPABASE_URL, SUPABASE_SECRET_KEY
  NEW_DB_URL=postgresql://... python scripts/migrate_supabase_to_postgres.py
"""
import json
import os
import pathlib

import psycopg
import requests
from psycopg.types.json import Jsonb

TABLES = ["communities", "sub_communities", "projects", "document_chunks"]
JSONB = {"assigned_agents", "payload"}
PAGE = 500


def fetch(table):
    url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows, offset = [], 0
    while True:
        r = requests.get(f"{url}/rest/v1/{table}", headers=headers, timeout=60,
                         params={"select": "*", "order": "id", "limit": PAGE, "offset": offset})
        r.raise_for_status()
        page = r.json()
        rows += page
        if len(page) < PAGE:
            return rows
        offset += PAGE


def value(col, v):
    if col == "embedding" and isinstance(v, str):  # pgvector comes back as "[0.1,0.2,...]"
        v = json.loads(v)
    return Jsonb(v) if col in JSONB and v is not None else v


if __name__ == "__main__":
    schema = (pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql").read_text()
    with psycopg.connect(os.environ["NEW_DB_URL"]) as dst:
        dst.execute(schema)
        for t in TABLES:
            rows = fetch(t)
            dst.execute(f"truncate public.{t}")
            if rows:
                cols = list(rows[0])
                sql = f'insert into public.{t} ({",".join(cols)}) values ({",".join(["%s"] * len(cols))})'
                dst.cursor().executemany(sql, [[value(c, r[c]) for c in cols] for r in rows])
                dst.execute(f"select setval(pg_get_serial_sequence('public.{t}','id'), (select max(id) from public.{t}))") \
                    if t != "projects" else None
            dims = dst.execute(f"select count(*), count(embedding), min(array_length(embedding,1)), max(array_length(embedding,1)) from public.{t}").fetchone()
            print(f"{t}: fetched={len(rows)} rows/embeddings/dims(min,max)={dims}")
        dst.commit()
