"""
Load DLD project status (progress, inspections, developer / management contacts) into public.propai_project_status.
Looks up every project number we hold, 20 per request. Counts requests in the shared _budget row.

    python scripts/propai_project_status.py
"""
import os, sys, json, time
import psycopg, requests
from dotenv import dotenv_values

env = {**dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env")), **os.environ}
BASE, KEY, DB = env["PROPAI_BASE_URL"], env["PROPAI_API_KEY"], env["DATABASE_URL"]
BATCH, BUDGET = 20, int(env.get("PROPAI_REQUEST_BUDGET", "1000"))

with psycopg.connect(DB, autocommit=True) as c:
    c.execute(open(os.path.join(os.path.dirname(__file__), "..", "migrations", "012_propai_project_status.sql")).read())
    nums = [r[0] for r in c.execute(
        "select distinct project_number from propai_projects where coalesce(project_number,'') <> '' "
        "except select project_number from propai_project_status order by 1")]
    used = c.execute("select requests_used from propai_sync_state where job='_budget'").fetchone()[0]
    print(f"{len(nums)} project numbers to look up, {used}/{BUDGET} requests used so far", flush=True)
    saved = 0
    for i in range(0, len(nums), BATCH):
        if used >= BUDGET:
            print("budget reached, stopping"); break
        chunk = nums[i:i + BATCH]
        r = requests.get(BASE + "/data/projects/status", headers={"x-api-key": KEY}, timeout=120,
                         params={"project_number": ",".join(chunk), "page_size": BATCH})
        used += 1
        c.execute("update propai_sync_state set requests_used=%s, updated_at=now() where job='_budget'", (used,))
        if r.status_code == 429 or r.status_code in (401, 403):
            print("stopped:", r.status_code, r.text[:200]); sys.exit(1)
        r.raise_for_status()
        for row in r.json().get("data", []):
            c.execute("insert into propai_project_status(project_number, payload) values (%s, %s) "
                      "on conflict (project_number) do update set payload=excluded.payload, fetched_at=now()",
                      (str(row["project_number"]), json.dumps(row)))
            saved += 1
        print(f"batch {i // BATCH + 1}: {saved} statuses saved, {used} requests used", flush=True)
        time.sleep(0.2)
    print("done:", c.execute("select count(*) from propai_project_status").fetchone()[0], "statuses stored")
