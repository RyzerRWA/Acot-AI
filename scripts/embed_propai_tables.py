"""
Fill the empty pgvector columns. Run this on your PC, from the Acot-AI- folder.
It reads names, embeds them with the local MiniLM model, and writes vector(384) values back.
It does not touch propai_sales or propai_rents. Phone numbers and emails are not embedded.

    python scripts/embed_propai_tables.py places --limit 5
    python scripts/embed_propai_tables.py places
    python scripts/embed_propai_tables.py raw
    python scripts/embed_propai_tables.py status
    python scripts/embed_propai_tables.py brokers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.postgres import _get_pool, fetch
from app.rag.embeddings.embedding_model import EmbeddingModel

TABLES = ("places", "raw", "status", "brokers")


def _clean(value) -> str:
    return " ".join(str(value or "").split())


def _phrase(label: str, extras: list[str], kind: str) -> str:
    parts = []
    for value in [label, *extras]:
        text = _clean(value)
        if text and text.lower() not in {part.lower() for part in parts}:
            parts.append(text[:400])
    if not parts:
        return ""
    tail = f", in {', '.join(parts[1:])}" if len(parts) > 1 else ""
    return f"{parts[0]} ({kind}){tail}"


def _rows(kind: str, limit: int | None) -> tuple[str, list[tuple], list[str]]:
    """Return update SQL, key tuples, and the text to embed. Only rows with a null embedding."""
    cap = f" limit {int(limit)}" if limit else ""
    if kind == "places":
        sql = (
            "select level, location_key, label, area, master from propai_places "
            "where embedding is null and label is not null and btrim(label) <> '' "
            f"order by sales_count desc nulls last{cap}"
        )
        found = fetch(sql)
        keys, texts = [], []
        for row in found:
            text = _phrase(row.get("label"), [row.get("master"), row.get("area")], row.get("level") or "place")
            if text:
                keys.append((row["level"], row["location_key"]))
                texts.append(text)
        update = "update propai_places set embedding = %s::vector where level = %s and location_key = %s"
        return update, keys, texts

    if kind == "raw":
        sql = (
            "select dataset, row_key, payload->>'project_name_en' as name, "
            "payload->>'developer_name' as developer, payload->>'master_project_en' as master, "
            "payload->>'area_name_label' as area, payload->>'offplan_status' as status, "
            "left(payload->>'project_description_en', 400) as description "
            "from propai_raw where embedding is null and dataset = 'projects' "
            f"and coalesce(payload->>'project_name_en', '') <> '' order by row_key{cap}"
        )
        found = fetch(sql)
        keys, texts = [], []
        for row in found:
            text = _phrase(row.get("name"), [row.get("developer"), row.get("master"), row.get("area"),
                                              row.get("status"), row.get("description")], "project")
            if text:
                keys.append((row["dataset"], row["row_key"]))
                texts.append(text)
        update = "update propai_raw set embedding = %s::vector where dataset = %s and row_key = %s"
        return update, keys, texts

    if kind == "status":
        sql = (
            "select project_number, payload->>'project_name_en' as name, "
            "payload->>'developer_name' as developer, payload->>'development_status' as status, "
            "payload->>'percent_completed' as percent "
            "from propai_project_status where embedding is null "
            f"and coalesce(payload->>'project_name_en', '') <> '' order by project_number{cap}"
        )
        found = fetch(sql)
        keys, texts = [], []
        for row in found:
            percent = _clean(row.get("percent"))
            extras = [row.get("developer"), row.get("status")]
            if percent:
                extras.append(f"{percent} percent complete")
            text = _phrase(row.get("name"), extras, "project status")
            if text:
                keys.append((row["project_number"],))
                texts.append(text)
        update = "update propai_project_status set embedding = %s::vector where project_number = %s"
        return update, keys, texts

    sql = (
        "select brn, name_en, agency_name from propai_brokers "
        "where embedding is null and coalesce(name_en, '') <> '' "
        f"order by brn{cap}"
    )
    found = fetch(sql)
    keys, texts = [], []
    for row in found:
        text = _phrase(row.get("name_en"), [row.get("agency_name")], "broker")
        if text:
            keys.append((row["brn"],))
            texts.append(text)
    update = "update propai_brokers set embedding = %s::vector where brn = %s"
    return update, keys, texts


def _vector(values) -> str:
    numbers = [float(value) for value in values]
    if len(numbers) != 384:
        raise RuntimeError(f"Expected 384 dimensions, got {len(numbers)}.")
    return "[" + ",".join(f"{number:.7f}" for number in numbers) + "]"


def embed_table(kind: str, limit: int | None) -> None:
    update, keys, texts = _rows(kind, limit)
    print(f"{kind}: {len(texts)} rows to embed")
    if not texts:
        print(f"{kind}: nothing left")
        return
    model = EmbeddingModel()
    vectors = model.model.encode(texts, show_progress_bar=True, batch_size=64)
    wrote = 0
    with _get_pool().connection() as conn:
        for start in range(0, len(texts), 200):
            batch = [
                (_vector(vectors[index]), *keys[index])
                for index in range(start, min(start + 200, len(texts)))
            ]
            with conn.transaction():
                conn.cursor().executemany(update, batch)
            wrote += len(batch)
            print(f"{kind}: wrote {wrote}/{len(texts)}")
    print(f"{kind}: done, {wrote} embeddings saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table", choices=TABLES)
    parser.add_argument("--limit", type=int, default=0, help="Embed only this many rows, for a first test.")
    args = parser.parse_args()
    embed_table(args.table, args.limit or None)


if __name__ == "__main__":
    main()
