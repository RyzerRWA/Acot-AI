"""
Load payment plans for off-plan projects into public.propai_project_plans (one request per project, resumable).
Counts requests in the shared _budget row and stops at the budget or on 401/403/429.

    PROPAI_REQUEST_BUDGET=8000 python scripts/propai_project_plans.py
"""
import os, sys, json, time, urllib.parse
import psycopg, requests
from dotenv import dotenv_values

here = os.path.dirname(__file__)
env = {**dotenv_values(os.path.join(here, "..", ".env")), **os.environ}
BASE, KEY, DB = env["PROPAI_BASE_URL"], env["PROPAI_API_KEY"], env["DATABASE_URL"]
BUDGET = int(env.get("PROPAI_REQUEST_BUDGET", "1000"))

with psycopg.connect(DB, autocommit=True) as c:
    c.execute(open(os.path.join(here, "..", "migrations", "014_propai_project_plans.sql")).read())
    keys = [r[0] for r in c.execute(
        "select location_key from propai_projects where not is_plot and location_key is not null "
        "and offplan_status in ('On Sale','Upcoming','Presale (EOI)','Announced','Secondary Off-Plan') "
        "except select location_key from propai_project_plans order by 1")]
    used = c.execute("select requests_used from propai_sync_state where job='_budget'").fetchone()[0]
    print(f"{len(keys)} off-plan projects to load, {used}/{BUDGET} requests used so far", flush=True)
    done = 0
    for lk in keys:
        if used >= BUDGET:
            print("budget reached, stopping"); break
        r = requests.get(BASE + "/data/projects/details/" + urllib.parse.quote(lk, safe=""), headers={"x-api-key": KEY}, timeout=90)
        used += 1
        c.execute("update propai_sync_state set requests_used=%s, updated_at=now() where job='_budget'", (used,))
        if r.status_code in (401, 403, 429):
            print("stopped:", r.status_code, r.text[:200]); sys.exit(1)
        if r.status_code == 404:
            c.execute("insert into propai_project_plans(location_key) values (%s) on conflict do nothing", (lk,)); continue
        r.raise_for_status()
        d = r.json()
        d = d.get("data", d) if isinstance(d, dict) else {}
        plans = [p for p in (d.get("payment_plan_details") or []) if p.get("payment_plan") not in (None, "Not Found")]
        c.execute("insert into propai_project_plans(location_key, payment_plans, property_types) values (%s,%s,%s) "
                  "on conflict (location_key) do update set payment_plans=excluded.payment_plans, property_types=excluded.property_types, fetched_at=now()",
                  (lk, json.dumps(plans), d.get("property_types")))
        done += 1
        if done % 100 == 0: print(f"{done} loaded, {used} requests used", flush=True)
        time.sleep(0.1)
    print("done:", c.execute("select count(*), count(*) filter (where jsonb_array_length(coalesce(payment_plans,'[]'))>0) from propai_project_plans").fetchone(), "(rows, with a plan)")
