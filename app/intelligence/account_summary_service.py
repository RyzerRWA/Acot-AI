import threading
import time
from datetime import datetime, timezone

from app.intelligence.account_summary_llm import generate_account_profile
from app.intelligence.chat_pattern_analyzer import analyze_questions


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
# PUBLIC API
# ============================================================

def get_account_summary(
    user_id: str,
    refresh: bool = False
) -> dict:
    """
    Build the account AI summary for a user.

    Only presentation fields are returned. Raw chat pattern data stays
    internal to this service.
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

    profile = generate_account_profile(analysis)

    return {
        "user_id": user_id,
        "available": True,
        "summary": profile["summary"],
        "acot_score": profile["acot_score"],
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

    return int(stats.get("meaningful_questions") or 0) > 0


# ============================================================
# CACHE INTERNALS
# ============================================================

def _cache_key(user_id: str, analysis: dict) -> str:

    # The newest question timestamp is part of the key so a summary is
    # regenerated automatically once new questions arrive.
    return f"{user_id}:{analysis.get('latest_question_at') or 'none'}"


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

        _cache[key] = (stored_at, payload)

    return dict(payload)


def _write_cache(key: str, payload: dict) -> None:

    with _cache_lock:
        _cache[key] = (time.monotonic(), dict(payload))

        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)), None)


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

    target = sys.argv[1] if len(sys.argv) > 1 else None

    if not target:
        raise SystemExit("Usage: python -m app.intelligence.account_summary_service <user_id>")

    print(
        json.dumps(
            get_account_summary(target, refresh=True),
            indent=2,
            default=str,
        )
    )