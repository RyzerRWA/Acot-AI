"""
Keep the Dubai market data current: pull new and revised sales, rents and projects from Prop-AI, then rebuild the
figures built from them (project prices, transaction-only projects, community stats).

It runs by itself every MARKET_REFRESH_EVERY_HOURS (default 48) and on demand from the admin panel. One run at a time:
a database lock stops a second one, even from another copy of the service. Each run may spend at most
MARKET_REFRESH_REQUESTS Prop-AI requests (default 3000), counted in the same total as the manual scripts.

Needs migration 015 (project_prices / derived_projects views) in the database. MARKET_REFRESH_ENABLED=false turns the
schedule off; the admin button still works.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import psycopg

from app.database.postgres import _get_pool, fetch

log = logging.getLogger("acot.refresh")

INTERVAL = timedelta(hours=float(os.getenv("MARKET_REFRESH_EVERY_HOURS", "48")))
ALLOWANCE = int(os.getenv("MARKET_REFRESH_REQUESTS", "3000"))
MANUAL_GAP = timedelta(minutes=10)  # a second manual run this soon is refused, so the button cannot burn the quota
JOBS = ("sales", "rents", "projects")
TABLE = {"sales": "propai_sales", "rents": "propai_rents", "projects": "propai_raw"}
VIEWS = ("propai_project_prices", "propai_derived_projects", "propai_area_stats", "propai_master_stats")
LOCK_KEY = 7_201_126
STALE_RUN = timedelta(hours=6)  # a "running" row older than this is a run that died with its process

_thread: threading.Thread | None = None
_guard = threading.Lock()

CREATE = """create table if not exists public.propai_refresh_runs (
  id bigserial primary key, started_at timestamptz not null default now(), finished_at timestamptz,
  trigger text not null, status text not null default 'running', requests int, added jsonb, error text)"""


def _sync():
    """The command-line sync script, loaded as a module so both use the same code."""
    scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import propai_sync
    return propai_sync


def _count(job: str) -> int:
    where = " where dataset = 'projects'" if job == "projects" else ""
    return fetch(f"select count(*)::int as n from {TABLE[job]}{where}")[0]["n"]


def _ensure_table():
    with _get_pool().connection() as c:
        c.execute(CREATE)


def _run(trigger: str, run_id: int, lock: psycopg.Connection):
    added: dict = {}
    status, error, spent = "ok", None, 0
    try:
        ps = _sync()
        with _get_pool().connection() as c:
            ps.state(c, "_budget")
            used0 = ps.state(c, "_budget")[4]
        ps.BUDGET = used0 + ALLOWANCE
        before = {j: _count(j) for j in JOBS}
        with httpx.Client(base_url=os.getenv("PROPAI_BASE_URL", "https://api.prop-ai.com"), timeout=120,
                          headers={"x-api-key": os.environ["PROPAI_API_KEY"], "Accept": "application/json"}) as client:
            for job in JOBS:
                ps.run(job, client)
        with _get_pool().connection() as c:
            unfinished = [r[0] for r in c.execute("select job from propai_sync_state where job = any(%s) and not done", (list(JOBS),))]
            used1 = ps.state(c, "_budget")[4]
        spent = used1 - used0
        added = {j: _count(j) - before[j] for j in JOBS}
        if unfinished:
            status = "partial"  # the request allowance or Prop-AI's rate limit stopped it; the next run carries on
        for view in VIEWS:  # the figures people see are built from the rows just loaded
            with _get_pool().connection() as c:
                c.execute(f"refresh materialized view public.{view}")
    except Exception as exc:  # noqa: BLE001 - recorded and shown in the admin, never silent
        log.exception("market refresh failed")
        status, error = "failed", str(exc)[:300]
    finally:
        try:
            with _get_pool().connection() as c:
                c.execute("update public.propai_refresh_runs set finished_at = now(), status = %s, requests = %s, added = %s, error = %s where id = %s",
                          (status, spent, json.dumps(added), error, run_id))
        finally:
            lock.close()  # releases the database lock
    log.info("market refresh %s (%s): %s requests, added %s", status, trigger, spent, added)


def start(trigger: str = "manual") -> dict:
    """Begin a refresh in the background. Returns what happened: started, or why not."""
    global _thread
    with _guard:
        _ensure_table()
        if _thread and _thread.is_alive():
            return {"started": False, "reason": "already_running"}
        if trigger == "manual":
            recent = fetch("select 1 from public.propai_refresh_runs where trigger = 'manual' and started_at > now() - %s::interval limit 1",
                           (f"{int(MANUAL_GAP.total_seconds())} seconds",))
            if recent:
                return {"started": False, "reason": "too_soon", "wait_minutes": int(MANUAL_GAP.total_seconds() // 60)}
        lock = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
        if not lock.execute("select pg_try_advisory_lock(%s)", (LOCK_KEY,)).fetchone()[0]:
            lock.close()
            return {"started": False, "reason": "already_running"}  # another copy of the service is running it
        with _get_pool().connection() as c:
            run_id = c.execute("insert into public.propai_refresh_runs(trigger) values (%s) returning id", (trigger,)).fetchone()[0]
        _thread = threading.Thread(target=_run, args=(trigger, run_id, lock), name="market-refresh", daemon=True)
        _thread.start()
        return {"started": True, "run_id": run_id}


def _last_done() -> datetime | None:
    rows = fetch("select max(finished_at) as at from public.propai_refresh_runs where status in ('ok', 'partial')")
    at = rows[0]["at"] if rows else None
    if not at:  # no run yet: the last manual script run counts
        rows = fetch("select max(updated_at) as at from propai_sync_state where job in ('sales', 'rents')")
        at = rows[0]["at"] if rows else None
    return datetime.fromisoformat(at) if at else None


def status() -> dict:
    _ensure_table()
    runs = fetch("select id, trigger, status, started_at, finished_at, requests, added, error from public.propai_refresh_runs order by id desc limit 5")
    running = bool(_thread and _thread.is_alive()) or any(
        r["status"] == "running" and datetime.fromisoformat(r["started_at"]) > datetime.now(timezone.utc) - STALE_RUN for r in runs)
    data = fetch(
        "select (select max(instance_date) from propai_sales where instance_date <= current_date)::text as latest_sale, "
        "(select max(contract_start_date) from propai_rents where contract_start_date <= current_date)::text as latest_rent, "
        "(select count(*) from propai_sales)::int as sales, (select count(*) from propai_rents)::int as rents, "
        "(select count(*) from propai_projects)::int as projects, "
        "(select requests_used from propai_sync_state where job = '_budget') as requests_used")[0]
    last = _last_done()
    return {
        "running": running, "runs": runs, "data": data,
        "every_hours": INTERVAL.total_seconds() / 3600, "request_allowance": ALLOWANCE,
        "schedule_on": os.getenv("MARKET_REFRESH_ENABLED", "true").lower() != "false",
        "last_refresh_at": last.isoformat() if last else None,
        "next_scheduled_at": (last + INTERVAL).isoformat() if last else None,
    }


def _due() -> bool:
    last = _last_done()
    return last is None or datetime.now(timezone.utc) - last >= INTERVAL


def _loop():
    time.sleep(120)  # let the service finish starting
    while True:
        try:
            if _due() and not (_thread and _thread.is_alive()):
                log.info("market refresh is due: starting")
                start("schedule")
        except Exception:  # noqa: BLE001 - a database blip must not end the schedule
            log.exception("market refresh scheduler check failed")
        time.sleep(900)


def start_schedule():
    """Called once at service start."""
    if os.getenv("MARKET_REFRESH_ENABLED", "true").lower() == "false":
        log.info("market refresh schedule is off")
        return
    threading.Thread(target=_loop, name="market-refresh-schedule", daemon=True).start()
