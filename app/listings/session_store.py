"""
Durable chat memory for the agent (conversation history and user profile), so a restart or the two-hour idle
cleanup does not make ACOT forget the user.

Off by default. Set ACOT_PERSIST_SESSIONS=1 to turn it on; the table is created the first time it is used.
Everything here is best effort: if the database is unreachable, the chat carries on with in-memory memory only.
"""
from __future__ import annotations

import json
import logging
import os

from app.database.postgres import execute, fetch

log = logging.getLogger("acot.session_store")

MAX_TURNS = 12                 # messages kept per session
MAX_AGE = "30 days"            # older sessions are ignored and removed

_ready = False


def enabled() -> bool:
    return os.getenv("ACOT_PERSIST_SESSIONS", "0").strip().lower() in ("1", "true", "yes", "on")


def _ensure_table() -> None:
    global _ready
    if _ready:
        return
    execute("create table if not exists acot_agent_sessions ("
            "session_id text primary key, history jsonb not null default '[]'::jsonb, "
            "profile jsonb not null default '{}'::jsonb, updated_at timestamptz not null default now())")
    _ready = True


def load(session_id: str) -> tuple[list, dict]:
    """(history, profile) saved for this session, or empty ones."""
    if not (enabled() and session_id):
        return [], {}
    try:
        _ensure_table()
        rows = fetch("select history, profile from acot_agent_sessions "
                     f"where session_id = %s and updated_at > now() - interval '{MAX_AGE}'", (session_id,))
    except Exception as exc:
        log.warning("could not load session memory: %s", exc)
        return [], {}
    if not rows:
        return [], {}
    history, profile = rows[0]["history"], rows[0]["profile"]
    return (history if isinstance(history, list) else []), (profile if isinstance(profile, dict) else {})


def save(session_id: str, history: list, profile: dict) -> None:
    if not (enabled() and session_id):
        return
    try:
        _ensure_table()
        execute("insert into acot_agent_sessions (session_id, history, profile, updated_at) "
                "values (%s, %s::jsonb, %s::jsonb, now()) on conflict (session_id) do update "
                "set history = excluded.history, profile = excluded.profile, updated_at = now()",
                (session_id, json.dumps(history[-MAX_TURNS:], default=str), json.dumps(profile, default=str)))
    except Exception as exc:
        log.warning("could not save session memory: %s", exc)


def delete(session_id: str) -> None:
    if not (enabled() and session_id):
        return
    try:
        _ensure_table()
        execute("delete from acot_agent_sessions where session_id = %s", (session_id,))
    except Exception as exc:
        log.warning("could not delete session memory: %s", exc)
