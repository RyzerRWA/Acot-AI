"""
Amenities from each developer's official site for projects Prop-AI has no amenity list for.
Uses the video collector (official sources only, never listing portals). Busiest projects first
(most sales in the last 12 months), resumable: projects already checked are skipped.

    python scripts/web_project_amenities.py [--limit N] [--workers 4]
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(here, "..", ".env"))
import psycopg  # noqa: E402
from app.Viedo_generation.collector import collect  # noqa: E402

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
DB = os.environ["DATABASE_URL"]

with psycopg.connect(DB, autocommit=True) as c:
    todo = c.execute(
        """
        select p.location_key, p.name, p.master_project, p.developer
        from propai_project_details d join propai_projects p on p.location_key = d.location_key
        left join lateral (
            select count(*) n from propai_sales s
            where s.location_key_project = p.location_key and s.instance_date >= now() - interval '12 months'
        ) s on true
        where d.web_checked_at is null and not (d.payload ? 'amenities_1')
        order by s.n desc, p.name
        """).fetchall()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} projects to look up", flush=True)

    def one(row):
        key, name, community, developer = row
        found = collect(key, name, community or "", developer or "")
        with psycopg.connect(DB, autocommit=True) as w:
            w.execute("update propai_project_details set web_amenities=%s, official_url=%s, web_checked_at=now() "
                      "where location_key=%s", (json.dumps(found["amenities"]), found["official_url"], key))
        return bool(found["amenities"])

    hits = 0
    with ThreadPoolExecutor(workers) as pool:
        for n, ok in enumerate(pool.map(one, todo), 1):
            hits += ok
            if n % 25 == 0:
                print(f"{n} checked, {hits} with amenities", flush=True)
    print(f"done: {len(todo)} checked, {hits} with amenities")
