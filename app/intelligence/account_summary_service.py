import threading
import time
from datetime import datetime, timezone

from app.database.postgres_client import get_postgres_client
from app.intelligence.account_summary_llm import generate_account_profile
from app.intelligence.chat_pattern_analyzer import analyze_questions


db = get_postgres_client()


# ============================================================
# SCORE LIMITS
# ============================================================

SCORE_LIMITS = {
    "freeplan": 5.0,
    "professional": 7.0,
    "enterprise": 10.0,
}


# ============================================================
# CACHE
# ============================================================

CACHE_TTL_SECONDS = 15 * 60
CACHE_MAX_ENTRIES = 200

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
_inflight: dict[str, threading.Lock] = {}


def clear_summary_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ============================================================
# USER PLAN
# ============================================================

def _get_user_plan(user_id: str) -> str:
    try:
        result = (
            db
            .schema("acot")
            .table("users")
            .select("plan")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )

        row = (result.data or [None])[0]
        plan = str(
            (row or {}).get("plan") or "freeplan"
        ).strip().lower()

        return plan if plan in SCORE_LIMITS else "freeplan"

    except Exception as exc:
        print(
            f"Account plan lookup failed for {user_id}: {exc}"
        )
        return "free"


# ============================================================
# PUBLIC API
# ============================================================

def get_account_summary(
    user_id: str,
    refresh: bool = False
) -> dict:
    """
    Build the account AI summary for a user.

    The subscription plan controls the maximum ACOT score.
    """

    analysis = analyze_questions(user_id)

    if not _has_signal(analysis):
        return _empty_payload(user_id, analysis)

    key = _cache_key(user_id, analysis)

    if not refresh:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    with _lock_for(key):
        if not refresh:
            cached = _read_cache(key)
            if cached is not None:
                return cached

        payload = _build_payload(user_id, analysis)
        _write_cache(key, payload)

    return payload


# ============================================================
# PAYLOAD
# ============================================================

def _build_payload(user_id: str, analysis: dict) -> dict:
    plan = _get_user_plan(user_id)

    analysis_for_ai = dict(analysis)
    analysis_for_ai["subscription_plan"] = plan

    profile = generate_account_profile(
        analysis_for_ai
    )

    max_score = SCORE_LIMITS[plan]
    raw_score = profile.get("acot_score")

    if isinstance(raw_score, (int, float)):
        acot_score = round(
            max(0.0, min(max_score, float(raw_score))),
            1,
        )
    else:
        acot_score = None

    return {
        "user_id": user_id,
        "available": True,
        "summary": profile["summary"],
        "acot_score": acot_score,
        "key_insights": profile["key_insights"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_question_at": analysis.get("latest_question_at"),
    }


def _empty_payload(user_id: str, analysis: dict) -> dict:
    return {
        "user_id": user_id,
        "available": False,
        "summary": None,
        "acot_score": None,
        "key_insights": [],
        "generated_at": None,
        "last_question_at": analysis.get("latest_question_at"),
    }


def _has_signal(analysis: dict) -> bool:
    stats = analysis.get("statistics") or {}

    return int(
        stats.get("meaningful_questions") or 0
    ) > 0


# ============================================================
# CACHE INTERNALS
# ============================================================

def _cache_key(
    user_id: str,
    analysis: dict
) -> str:
    plan = _get_user_plan(user_id)

    return (
        f"{user_id}:"
        f"{plan}:"
        f"{analysis.get('latest_question_at') or 'none'}"
    )


def _read_cache(key: str) -> dict | None:
    now = time.monotonic()

    with _cache_lock:
        entry = _cache.get(key)

        if not entry:
            return None

        stored_at, payload = entry

        if now - stored_at > CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None

        _cache[key] = (
            stored_at,
            payload,
        )

    return dict(payload)


def _write_cache(
    key: str,
    payload: dict
) -> None:
    with _cache_lock:
        _cache[key] = (
            time.monotonic(),
            dict(payload),
        )

        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.pop(
                next(iter(_cache)),
                None,
            )


def _lock_for(key: str) -> threading.Lock:
    with _cache_lock:
        lock = _inflight.get(key)

        if lock is None:
            lock = threading.Lock()
            _inflight[key] = lock

        return lock


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import json
    import sys

    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    if not target:
        raise SystemExit(
            "Usage: python -m "
            "app.intelligence.account_summary_service "
            "<user_id>"
        )

    print(
        json.dumps(
            get_account_summary(
                target,
                refresh=True,
            ),
            indent=2,
            default=str,
        )
    )