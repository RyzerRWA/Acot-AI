"""
Mirror Dubai data from the Prop-AI API into public.propai_raw, inside a hard request budget.

Three bulk streams (whole Dubai, no per-area loops) cost ~rows/1000 requests each; area/project metrics and
trends are computed from the mirrored rows in SQL instead of spending requests on them.

    python scripts/propai_sync.py --probe            # 1 request per stream, prints a sample row's keys
    python scripts/propai_sync.py [projects rents sales brokers]   # resumable; re-run to continue / refresh

Env: PROPAI_API_KEY, PROPAI_BASE_URL (default dev), PROPAI_REQUEST_BUDGET (default 1000, counted in the DB).
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database.postgres import _get_pool  # noqa: E402


def _hash_key(r):
    return hashlib.sha1(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()


# ponytail: key fields below are guesses for sales/projects until --probe shows the real names; the hash fallback
# dedupes identical rows only, so confirm the sales id before the first full run.
JOBS = {
    # typed tables (migration 005): jsonb_populate_recordset maps JSON keys to columns by name
    "rents": dict(path="/data/rents/regions/dubai", mode="cursor", table="propai_rents", pk=("pai_contract_id", "unit_seq"),
                  params={"per_page": 1000, "paging": "cursor", "outlier_removal": "false"}),
    "sales": dict(path="/data/sales/regions/dubai", mode="cursor", table="propai_sales", pk=("source", "transaction_id"),
                  params={"per_page": 1000, "paging": "cursor", "outlier_removal": "false"}),
    "projects": dict(path="/data/projects/basic-details", mode="page", size=100,
                     params={"region": "dubai", "page_size": 100},
                     key=lambda r: str(r.get("prop-ai_location_key_project") or r.get("project_id") or _hash_key(r))),
    "brokers": dict(path="/data/brokers/agents", mode="page", size=500, params={"page_size": 500},
                    key=lambda r: str(r.get("brn") or _hash_key(r))),
}
BUDGET = int(os.getenv("PROPAI_REQUEST_BUDGET", "1000"))


def upsert_sql(table, pk):
    """insert ... from jsonb_populate_recordset, deduplicated inside the batch, newest updated_at wins."""
    with _get_pool().connection() as c:
        cols = [r[0] for r in c.execute("select column_name from information_schema.columns "
                                        "where table_schema='public' and table_name=%s order by ordinal_position", (table,))]
    pks = ", ".join(pk)
    sets = ", ".join(f"{col}=excluded.{col}" for col in cols if col not in pk)
    return (f"insert into {table} select distinct on ({pks}) * from jsonb_populate_recordset(null::{table}, %s::jsonb) "
            f"order by {pks}, updated_at desc nulls last on conflict ({pks}) do update set {sets}")


def rows_of(body):
    if isinstance(body, list):
        return body
    for v in body.values():
        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
            return v
    return []


def state(conn, job):
    conn.execute("insert into propai_sync_state(job) values (%s) on conflict do nothing", (job,))
    return conn.execute("select cursor, page, since, done, requests_used from propai_sync_state where job=%s", (job,)).fetchone()


def run(job, client, probe=False):
    cfg = JOBS[job]
    pool = _get_pool()
    with pool.connection() as conn:
        cursor, page, since, done, _ = state(conn, job)
        if done:  # finished earlier: start an incremental pass from the last finish time
            cursor, page, done = None, 1, False
        used = state(conn, "_budget")[4]
    started = datetime.now(timezone.utc)
    upsert = upsert_sql(cfg["table"], cfg["pk"]) if cfg.get("table") else None
    while True:
        if used >= BUDGET:
            print(f"{job}: request budget {BUDGET} reached, stopping (re-run to resume)")
            return
        params = dict(cfg["params"])
        window = os.getenv(f"PROPAI_SINCE_{job.upper()}")
        if window:
            params["start_date"] = window
        if cfg["mode"] == "cursor":
            if cursor:
                params["cursor"] = cursor
            if since:
                params["updated_since"] = since.isoformat()
        else:
            params["page"] = page
        resp = client.get(cfg["path"], params=params)
        used += 1
        if resp.status_code == 429:
            print(f"{job}: rate limited (429), stopping; progress is saved, re-run later")
            return
        resp.raise_for_status()
        body = resp.json()
        rows = rows_of(body)
        if probe:
            print(job, "top-level keys:", list(body) if isinstance(body, dict) else "list", "| rows:", len(rows))
            print(json.dumps(rows[0], indent=1, default=str)[:1500] if rows else "no rows")
            return
        nxt = (body.get("pagination") or {}).get("next_cursor") if cfg["mode"] == "cursor" else None
        # the API keeps returning a cursor after the last page, so a short or empty page also ends a cursor stream
        finished = (not nxt or len(rows) < cfg["params"]["per_page"]) if cfg["mode"] == "cursor" else len(rows) < cfg["size"]
        with pool.connection() as conn:   # rows + resume point + budget commit together
            if cfg.get("table"):
                conn.execute(upsert, (json.dumps(rows, default=str),))
            else:
                conn.cursor().executemany(
                    "insert into propai_raw(dataset,row_key,payload) values (%s,%s,%s) "
                    "on conflict (dataset,row_key) do update set payload=excluded.payload, fetched_at=now()",
                    [(job, cfg["key"](r), json.dumps(r, default=str)) for r in rows])
            conn.execute("update propai_sync_state set requests_used=%s, updated_at=now() where job='_budget'", (used,))
            conn.execute("insert into propai_sync_state(job) values ('_budget') on conflict do nothing")
            conn.execute("update propai_sync_state set cursor=%s, page=%s, done=%s, since=%s, updated_at=now() where job=%s",
                         (nxt, page + 1, finished, started if finished else since, job))
        print(f"{job}: +{len(rows)} rows, {used}/{BUDGET} requests")
        if finished:
            return
        cursor, page = nxt, page + 1




if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")] or list(JOBS)
    with _get_pool().connection() as c:
        state(c, "_budget")
    with httpx.Client(base_url=os.getenv("PROPAI_BASE_URL", "https://dev-api.prop-ai.de"), timeout=120,
                      headers={"x-api-key": os.environ["PROPAI_API_KEY"], "Accept": "application/json"}) as cl:
        for j in args:
            run(j, cl, probe="--probe" in sys.argv)
            if j == "brokers" and "--probe" not in sys.argv:   # idempotent, so safe after a partial run too
                with _get_pool().connection() as c:
                    c.execute(REFRESH_BROKERS)
                    print("brokers table:", c.execute("select count(*) from propai_brokers").fetchone()[0], "rows")
