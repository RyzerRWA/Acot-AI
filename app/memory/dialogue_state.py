"""Session dialogue state for ACOT turns.

Stores the active task as slots (place, beds, action…), not sentence patterns.
Phase 1: create / update / read. Follow-up routing builds on this later.
"""
from __future__ import annotations

from typing import Any


ACTIONS = frozenset({
    "search",
    "project_count",
    "sales_count",
    "sales_list",
    "recommend",
    "agents",
    "clarify",
    "general",
})

GOALS = frozenset({
    "family",
    "budget",
    "invest",
    "ready",
    "compare",
    "open_buy",
    "afford",
})


# Recommend subtypes persisted so follow-ups ("studios" after a split) stay on path.
RECOMMEND_KINDS = frozenset({
    "filtered",
    "afford",
    "family_areas",
    "open_best",
    "cheapest",
    "shortlist",
    "split",
    "ready_offplan",
    "yield_areas",
})


def empty_dialogue_state() -> dict[str, Any]:
    return {
        "action": None,
        "goal": None,
        "kind": None,
        "place": None,
        "beds": None,
        "budget_aed": None,
        "month": None,
        "year": None,
        "ready_only": False,
        "prefer_cheaper": False,
        "exclude_names": [],
        "offset": 0,
        # Last-N for sales_list ("last 5 sales"); None → default page size.
        "limit": None,
        "last_result": None,
        "source_question": None,
        # Family-areas ranking flags (kids / avoid_towers / villas).
        "focus": None,
    }


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (OverflowError, TypeError, ValueError):
        return None


def normalize_dialogue_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return a full state dict with known keys only."""
    base = empty_dialogue_state()
    if not isinstance(raw, dict):
        return base
    action = _clean_text(raw.get("action"))
    if action in ACTIONS:
        base["action"] = action
    goal = _clean_text(raw.get("goal"))
    if goal in GOALS:
        base["goal"] = goal
    kind = _clean_text(raw.get("kind"))
    if kind in RECOMMEND_KINDS:
        base["kind"] = kind
    base["place"] = _clean_text(raw.get("place"))
    base["beds"] = _clean_int(raw.get("beds"))
    base["budget_aed"] = _clean_int(raw.get("budget_aed"))
    base["month"] = _clean_int(raw.get("month"))
    base["year"] = _clean_int(raw.get("year"))
    base["ready_only"] = bool(raw.get("ready_only"))
    base["prefer_cheaper"] = bool(raw.get("prefer_cheaper"))
    names = raw.get("exclude_names") or []
    if isinstance(names, (list, tuple)):
        base["exclude_names"] = [
            str(name).strip()
            for name in names
            if str(name or "").strip()
        ][:40]
    base["offset"] = max(0, _clean_int(raw.get("offset")) or 0)
    limit = _clean_int(raw.get("limit"))
    if limit is not None:
        base["limit"] = max(1, min(limit, 15))
    last = raw.get("last_result")
    base["last_result"] = dict(last) if isinstance(last, dict) else None
    base["source_question"] = _clean_text(raw.get("source_question"))
    focus = raw.get("focus")
    if isinstance(focus, dict):
        base["focus"] = {
            key: bool(focus.get(key))
            for key in (
                "kids",
                "avoid_towers",
                "want_villas",
                "not_studios",
                "want_3bed",
                "no_yield",
            )
            if key in focus
        } or None
    else:
        base["focus"] = None
    return base


def patch_dialogue_state(
    current: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge updates into current state. None values in updates leave old fields."""
    state = normalize_dialogue_state(current)
    if not isinstance(updates, dict):
        return state
    for key, value in updates.items():
        if key not in state:
            continue
        if value is None and key not in {
            "ready_only", "prefer_cheaper", "exclude_names", "offset", "limit",
            "last_result", "focus",
        }:
            # Explicit clear only when caller passes empty string for text slots.
            continue
        if key == "focus":
            if isinstance(value, dict):
                state["focus"] = {
                    k: bool(value.get(k))
                    for k in (
                        "kids",
                        "avoid_towers",
                        "want_villas",
                        "not_studios",
                        "want_3bed",
                        "no_yield",
                    )
                    if k in value
                } or None
            elif value is None:
                state["focus"] = None
            continue
        if key == "exclude_names":
            if isinstance(value, (list, tuple)):
                state["exclude_names"] = [
                    str(name).strip()
                    for name in value
                    if str(name or "").strip()
                ][:40]
            continue
        if key == "last_result":
            state["last_result"] = dict(value) if isinstance(value, dict) else None
            continue
        if key in {"ready_only", "prefer_cheaper"}:
            state[key] = bool(value)
            continue
        if key == "offset":
            state["offset"] = max(0, _clean_int(value) or 0)
            continue
        if key == "limit":
            cleaned = _clean_int(value)
            state["limit"] = max(1, min(cleaned, 15)) if cleaned is not None else None
            continue
        if key in {"beds", "budget_aed", "month", "year"}:
            state[key] = _clean_int(value)
            continue
        if key == "action":
            text = _clean_text(value)
            state["action"] = text if text in ACTIONS else state["action"]
            continue
        if key == "goal":
            text = _clean_text(value)
            state["goal"] = text if text in GOALS else state["goal"]
            continue
        if key == "kind":
            text = _clean_text(value)
            state["kind"] = text if text in RECOMMEND_KINDS else state["kind"]
            continue
        if key in {"place", "source_question"}:
            state[key] = _clean_text(value)
    return normalize_dialogue_state(state)


def state_from_sale_filters(
    *,
    action: str,
    place=None,
    beds=None,
    month=None,
    year=None,
    count=None,
    source_question=None,
    offset: int = 0,
    limit=None,
) -> dict[str, Any]:
    """Build state for sales count / sales list turns."""
    last_result = None
    if count is not None:
        last_result = {"count": _clean_int(count)}
    return normalize_dialogue_state({
        "action": action,
        "place": place,
        "beds": beds,
        "month": month,
        "year": year,
        "offset": offset,
        "limit": limit,
        "last_result": last_result,
        "source_question": source_question,
    })


def sale_filters_from_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Legacy-shaped sale_query dict for callers that still read sale_query.

    Only sales_list / sales_count tasks expose sale filters. A recommend/search
    place must not be treated as a prior sales query (that made transaction
    follow-ups jump to area-wide sales after a project shortlist).
    """
    current = normalize_dialogue_state(state)
    if current.get("action") not in {"sales_count", "sales_list"}:
        return None
    out = {
        "place": current.get("place"),
        "bedrooms": current.get("beds"),
        "year": current.get("year"),
        "month": current.get("month"),
        "source_question": current.get("source_question"),
        "action": current.get("action"),
        "limit": current.get("limit"),
    }
    if isinstance(current.get("last_result"), dict) and current["last_result"].get("count") is not None:
        out["count"] = current["last_result"]["count"]
    return out
