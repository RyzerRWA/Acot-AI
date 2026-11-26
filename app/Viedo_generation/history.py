"""
Load the projects and communities a buyer searched for.

Source: acot.messages.response_json (assistant answers), linked by
acot.conversations.user_id. No account summary is used.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.database.postgres import fetch

# Latest completed assistant turns to scan per user.
LOOKBACK = 30


def searched_projects(user_id: str, lookback: int = LOOKBACK) -> dict[str, Any]:
    """
    Return projects and communities from the buyer's AI search chat.

    Shape:
      {
        "user_id": str,
        "projects": [
          {
            "key": str | None,          # catalog key when present
            "name": str,
            "community": str | None,
            "developer": str | None,
            "photos": list[str],        # from the saved answer (up to ~4)
            "price": number | None,
            "price_label": str | None,
            "bedrooms_min": number | None,
            "bedrooms_max": number | None,
            "handover_time": str | None,
            "status": str | None,
            "appeared": int,            # how often this project showed up
            "last_seen": str | None,    # ISO timestamp of latest appearance
          },
          ...
        ],
        "communities": [
          {"name": str, "appeared": int, "last_seen": str | None},
          ...
        ],
      }

    Projects are sorted by appeared (desc), then last_seen (desc).
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return {"user_id": "", "projects": [], "communities": []}

    rows = fetch(
        """
        select
          m.response_json as "responseJson",
          m.created_at as "createdAt"
        from acot.conversations c
        join acot.messages m on m.conversation_id = c.id
        where c.user_id = %s
          and m.role = 'assistant'
          and m.status = 'ready'
          and m.response_json is not null
        order by m.created_at desc
        limit %s
        """,
        (user_id, max(1, int(lookback))),
    )

    projects_by_id: dict[str, dict[str, Any]] = {}
    communities_by_name: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"name": "", "appeared": 0, "last_seen": None}
    )

    for row in rows:
        payload = row.get("responseJson") or {}
        if not isinstance(payload, dict):
            continue
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            continue
        created_at = row.get("createdAt")
        seen_at = str(created_at) if created_at is not None else None

        for item in data.get("projects") or []:
            if isinstance(item, dict):
                _merge_project(projects_by_id, item, seen_at)

        # Comparison cards are also projects shown side-by-side.
        for item in data.get("comparison") or []:
            if isinstance(item, dict):
                _merge_project(projects_by_id, item, seen_at)

    # Communities: once per turn, from community cards and project.community.
    for row in rows:
        payload = row.get("responseJson") or {}
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        created_at = row.get("createdAt")
        seen_at = str(created_at) if created_at is not None else None
        names_this_turn: set[str] = set()

        for item in data.get("communities") or []:
            if isinstance(item, dict):
                name = _text(item.get("name") or item.get("official_area"))
                if name:
                    names_this_turn.add(name)

        for item in list(data.get("projects") or []) + list(data.get("comparison") or []):
            if isinstance(item, dict):
                name = _text(item.get("community"))
                if name:
                    names_this_turn.add(name)

        for name in names_this_turn:
            entry = communities_by_name[name.lower()]
            entry["name"] = name
            entry["appeared"] += 1
            if seen_at and (entry["last_seen"] is None or seen_at > entry["last_seen"]):
                entry["last_seen"] = seen_at

    projects = sorted(
        projects_by_id.values(),
        key=lambda p: (p["appeared"], p["last_seen"] or ""),
        reverse=True,
    )
    communities = sorted(
        communities_by_name.values(),
        key=lambda c: (c["appeared"], c["last_seen"] or ""),
        reverse=True,
    )

    return {
        "user_id": user_id,
        "projects": projects,
        "communities": communities,
    }


def _merge_project(
    store: dict[str, dict[str, Any]],
    item: dict[str, Any],
    seen_at: str | None,
) -> None:
    name = _text(item.get("name"))
    key = _text(item.get("key")) or None
    if not name and not key:
        return

    # Prefer catalog key; fall back to lowercased name.
    identity = key or name.lower()
    photos = _photos(item.get("photos"))
    beds = item.get("bedrooms") if isinstance(item.get("bedrooms"), dict) else {}

    existing = store.get(identity)
    if existing is None:
        store[identity] = {
            "key": key,
            "name": name or key or "Project",
            "community": _text(item.get("community")) or None,
            "developer": _text(item.get("developer")) or None,
            "photos": photos,
            "price": _number(item.get("price")),
            "price_label": _text(item.get("price_label")) or None,
            "bedrooms_min": _number(beds.get("min")),
            "bedrooms_max": _number(beds.get("max")),
            "handover_time": _text(item.get("handover_time")) or None,
            "status": _text(item.get("status")) or None,
            "appeared": 1,
            "last_seen": seen_at,
        }
        return

    existing["appeared"] += 1
    if seen_at and (existing["last_seen"] is None or seen_at > existing["last_seen"]):
        existing["last_seen"] = seen_at
    # Keep the richest copy of each field.
    if key and not existing["key"]:
        existing["key"] = key
    if name and (not existing["name"] or existing["name"] == existing["key"]):
        existing["name"] = name
    if photos and len(photos) > len(existing["photos"]):
        existing["photos"] = photos
    for field, value in (
        ("community", _text(item.get("community")) or None),
        ("developer", _text(item.get("developer")) or None),
        ("price_label", _text(item.get("price_label")) or None),
        ("handover_time", _text(item.get("handover_time")) or None),
        ("status", _text(item.get("status")) or None),
    ):
        if value and not existing.get(field):
            existing[field] = value
    price = _number(item.get("price"))
    if price is not None and existing.get("price") is None:
        existing["price"] = price
    if existing.get("bedrooms_min") is None:
        existing["bedrooms_min"] = _number(beds.get("min"))
    if existing.get("bedrooms_max") is None:
        existing["bedrooms_max"] = _number(beds.get("max"))


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        n = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return int(n) if n.is_integer() else n


def _photos(value: Any) -> list[str]:
    if isinstance(value, str) and value.startswith("http"):
        return [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        url = item.strip()
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


# Quick local check:
#   python -c "from app.Viedo_generation.history import searched_projects; import json; print(json.dumps(searched_projects('USER_ID'), indent=2)[:2000])"
