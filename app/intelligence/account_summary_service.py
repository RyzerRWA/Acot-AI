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

# The key already changes whenever the user asks a new question or changes plan, so a long
# lifetime only avoids paying for the same summary again.
CACHE_TTL_SECONDS = 6 * 60 * 60
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

        if "free" in plan:
            return "freeplan"
        if plan in {"professional", "pro"} or "professional" in plan:
            return "professional"
        if plan == "enterprise" or "enterprise" in plan:
            return "enterprise"
        return plan if plan in SCORE_LIMITS else "freeplan"

    except Exception as exc:
        print(
            f"Account plan lookup failed for {user_id}: {exc}"
        )
        return "freeplan"


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

        try:
            payload = _build_payload(user_id, analysis)
        except Exception as exc:
            # ACOT QA R17: one Admin account summary failed to load. A slow or failed model call used to become
            # a 500; show the last good summary, or say plainly that it could not be generated.
            print(f"Account summary generation failed for {user_id}: {exc}")
            return _last_good(user_id) or {
                **_empty_payload(user_id, analysis),
                "error": "The AI summary could not be generated right now. Try again in a minute.",
            }
        _write_cache(key, payload)

    return payload


def _last_good(user_id: str) -> dict | None:
    """The newest cached summary for this user (an older question set or plan), marked stale."""
    with _cache_lock:
        entries = [entry for key, entry in _cache.items() if key.startswith(f"{user_id}:")]
    if not entries:
        return None
    _, payload = max(entries, key=lambda entry: entry[0])
    return {**payload, "stale": True}


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