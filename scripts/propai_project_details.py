"""
Save the full Prop-AI project details for every project into public.propai_project_details
(amenities, highlights, layouts with starting prices, payment plans, brochures). One request per
project, resumable: projects already saved are skipped. Counts requests in the shared _budget row
and stops at the budget or on 401/403/429.

    PROPAI_REQUEST_BUDGET=15000 python scripts/propai_project_details.py [--refresh-days N]
"""
import os, sys, json, time, urllib.parse
import psycopg, requests
from dotenv import dotenv_values

here = os.path.dirname(__file__)
env = {**dotenv_values(os.path.join(here, "..", ".env")), **os.environ}
BASE, KEY, DB = env["PROPAI_BASE_URL"], env["PROPAI_API_KEY"], env["DATABASE_URL"]
BUDGET = int(env.get("PROPAI_REQUEST_BUDGET", "1000"))
days = int(sys.argv[sys.argv.index("--refresh-days") + 1]) if "--refresh-days" in sys.argv else None

with psycopg.connect(DB, autocommit=True) as c:
    c.execute(open(os.path.join(here, "..", "migrations", "017_propai_project_details.sql")).read())
    stale = "and fetched_at > now() - make_interval(days => %s)" if days else ""
    keys = [r[0] for r in c.execute(
        "select location_key from propai_projects where not is_plot and location_key is not null "
        f"except select location_key from propai_project_details where true {stale} order by 1",
        (days,) if days else None)]
    used = c.execute("select requests_used from propai_sync_state where job='_budget'").fetchone()[0]
    print(f"{len(keys)} projects to load, {used}/{BUDGET} requests used so far", flush=True)
    done = 0
    for lk in keys:
        if used >= BUDGET:
            print("budget reached, stopping"); break
        try:
            r = requests.get(BASE + "/data/projects/details/" + urllib.parse.quote(lk, safe=""),
                             headers={"x-api-key": KEY}, timeout=90)
        except requests.RequestException as exc:
            print("skip", lk, exc, flush=True); time.sleep(5); continue
        used += 1
        c.execute("update propai_sync_state set requests_used=%s, updated_at=now() where job='_budget'", (used,))
        if r.status_code in (401, 403, 429):
            print("stopped:", r.status_code, r.text[:200]); sys.exit(1)
        d = {}
        if r.status_code != 404:
            if r.status_code >= 500:
                print("skip", lk, r.status_code, flush=True); continue
            r.raise_for_status()
            d = r.json()
            d = d.get("data", d) if isinstance(d, dict) else {}
        c.execute("insert into propai_project_details(location_key, payload) values (%s,%s) "
                  "on conflict (location_key) do update set payload=excluded.payload, fetched_at=now()",
                  (lk, json.dumps(d)))
        done += 1
        if done % 100 == 0: print(f"{done} loaded, {used} requests used", flush=True)
        time.sleep(0.1)
    print("done:", c.execute(
        "select count(*), count(*) filter (where payload ? 'amenities_1') from propai_project_details").fetchone(),
        "(rows, with amenities)")
