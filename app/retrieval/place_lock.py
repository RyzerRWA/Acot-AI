"""Keep a named place from swallowing numbered siblings (Arabian Ranches vs 2/3)."""
from __future__ import annotations

import re

_SPACE = re.compile(r"\s+")
_VERSION_SUFFIX = re.compile(r"[\s-]+(?:[0-9]+|II|III|IV)\s*$", re.I)
_VERSION_TAIL = re.compile(r"^(?:[0-9]+|II|III|IV)\b", re.I)


def norm_place(name: str) -> str:
    return _SPACE.sub(" ", (name or "").strip()).upper()


def is_numbered_sibling(asked: str, candidate: str) -> bool:
    """True when candidate is asked + a version number (Arabian Ranches 3)."""
    asked_n = norm_place(asked)
    cand = norm_place(candidate)
    if not asked_n or not cand or asked_n == cand:
        return False
    if cand.startswith(asked_n + " ") or cand.startswith(asked_n + "-"):
        rest = cand[len(asked_n) :].lstrip(" -")
        return bool(_VERSION_TAIL.match(rest))
    return False


def asked_keys(names) -> list[str]:
    keys: list[str] = []
    for name in names or []:
        key = norm_place(str(name))
        if key and key not in keys:
            keys.append(key)
    return keys


def drop_numbered_siblings(rows: list[dict], names, fields=("community", "sub_community", "name")) -> list[dict]:
    asked = [str(name) for name in (names or []) if name]
    if not asked or not rows:
        return rows
    kept = []
    for row in rows:
        if any(
            is_numbered_sibling(name, str(row.get(field) or ""))
            for name in asked
            for field in fields
        ):
            continue
        kept.append(row)
    return kept


def _sibling_patterns(keys: list[str]) -> list[str]:
    return [
        re.escape(key) + r"[[:space:][:punct:]]+(?:[0-9]+|II|III|IV)\b"
        for key in keys
        if not _VERSION_SUFFIX.search(key)
    ]


def contains_place_sql(columns: list[str], names: list[str]) -> tuple[str, list]:
    """ILIKE match that drops numbered master-plan siblings."""
    keys = asked_keys(names)
    if not keys or not columns:
        return "false", []
    likes = ["%" + key + "%" for key in keys]
    matched = " or ".join(f"{col} ILIKE ANY(%s::text[])" for col in columns)
    params: list = [likes] * len(columns)
    patterns = _sibling_patterns(keys)
    if not patterns:
        return f"({matched})", params
    excluded = " or ".join(
        f"upper(coalesce({col}, '')) ~ ANY(%s::text[])" for col in columns
    )
    params.extend([patterns] * len(columns))
    return f"(({matched}) and not ({excluded}))", params


def sales_place_sql(asked: list[str], areas: list[str] | None = None) -> tuple[str, list]:
    """Sales for this place only, even when a sibling shares a DLD area name."""
    master_keys = asked_keys(asked)
    area_keys = asked_keys(list(asked or []) + list(areas or []))
    if not master_keys:
        return "false", []
    master_norm = (
        "upper(regexp_replace(btrim(coalesce(master_project_prop_ai, '')), '\\s+', ' ', 'g'))"
    )
    area_norm = (
        "upper(regexp_replace(btrim(coalesce(area_name_prop_ai, '')), '\\s+', ' ', 'g'))"
    )
    sql = (
        f"({master_norm} = any(%s::text[])"
        f" or ({area_norm} = any(%s::text[])"
        f" and ({master_norm} = '' or {master_norm} = any(%s::text[]))))"
    )
    return sql, [master_keys, area_keys, master_keys]
