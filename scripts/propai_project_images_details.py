"""
Fill photos for projects that have none, from Prop-AI's project details (`images_unclassified`).
The project list returns no photos for about 1,560 projects; the details call does for many of them.

    PROPAI_REQUEST_BUDGET=8000 python scripts/propai_project_images_details.py [--limit N]

One Prop-AI request per project; links expire in 1 hour so each project is copied to the bucket right away.
Resumable: projects that already have photos are skipped, and projects Prop-AI has no photos for are noted in
propai_project_images_checked so they are not requested again. Needs `gcloud auth login`. Stops on 401/403/429.
"""
import os, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

import httpx
import psycopg

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.dirname(here))
import propai_project_images as pi  # noqa: E402  (copy_one, ordered, token, AuthError)
from propai_project_images import ps  # noqa: E402

BASE, KEY = os.environ["PROPAI_BASE_URL"], os.environ["PROPAI_API_KEY"]
DB = os.environ.get("LISTINGS_DATABASE_URL") or os.environ["DATABASE_URL"]
BUDGET = int(os.getenv("PROPAI_REQUEST_BUDGET", "1000"))
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

with psycopg.connect(DB, autocommit=True) as c:
    c.execute("create table if not exists public.propai_project_images_checked "
              "(project_key text primary key, checked_at timestamptz default now(), found int)")
    todo = c.execute(
        "select p.key, p.location_key from propai_projects p where p.location_key is not null "
        "and not exists (select 1 from propai_project_images i where i.project_key = p.key) "
        "and not exists (select 1 from propai_project_images_checked k where k.project_key = p.key) "
        "order by p.is_plot, p.key").fetchall()
    if limit:
        todo = todo[:limit]
    used = c.execute("select requests_used from propai_sync_state where job='_budget'").fetchone()[0]
    print(f"{len(todo)} projects to check, {used}/{BUDGET} requests used so far", flush=True)
    pi.token()  # fail fast if gcloud is logged out
    got = 0
    try:
        with ThreadPoolExecutor(pi.WORKERS) as threads:
            for n, (key, lk) in enumerate(todo, 1):
                if used >= BUDGET:
                    print("budget reached, stopping"); break
                r = httpx.get(BASE.rstrip("/") + "/data/projects/details/" + urllib.parse.quote(lk, safe=""),
                              headers={"x-api-key": KEY}, timeout=90)
                used += 1
                c.execute("update propai_sync_state set requests_used=%s, updated_at=now() where job='_budget'", (used,))
                if r.status_code in (401, 403, 429):
                    print("stopped:", r.status_code, r.text[:200]); sys.exit(1)
                urls = []
                if r.status_code == 200:
                    d = r.json()
                    d = d.get("data", d) if isinstance(d, dict) else d
                    d = d[0] if isinstance(d, list) and d else d
                    if isinstance(d, dict):
                        urls = [u for u in (d.get("images_unclassified") or []) if isinstance(u, str)]
                results = [x for x in threads.map(pi.copy_one, [(key, i, u) for i, u in enumerate(pi.ordered(urls))]) if x]
                for row in results:
                    c.execute("insert into propai_project_images(project_key,position,object_name,bytes,content_type) "
                              "values (%s,%s,%s,%s,%s) on conflict (project_key,position) do update set "
                              "object_name=excluded.object_name, bytes=excluded.bytes, copied_at=now()", row)
                if len(results) == len(urls):  # all copied (or none exist): do not ask again
                    c.execute("insert into propai_project_images_checked(project_key, found) values (%s,%s) "
                              "on conflict (project_key) do update set found=excluded.found, checked_at=now()", (key, len(results)))
                got += bool(results)
                if n % 50 == 0 or limit:
                    print(f"{n}/{len(todo)} checked, {got} projects got photos, {used} requests used", flush=True)
                time.sleep(0.1)
    except pi.AuthError as e:
        print("STOPPED:", e); sys.exit(1)
    print(f"done: {got} projects got photos")
