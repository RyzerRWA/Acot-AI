"""Keep project sale facts honest across cards and answer text.

Rules:
- Never claim recent_sale without a positive price (or clear sale date + kind).
- last_sold_on and last_sale stay in sync when either is present.
- Do not invent prices, dates, or bedroom counts.
"""

from __future__ import annotations


def _positive_price(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _sale_date_iso(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        return text[:10]
    text = str(value).strip()
    if not text:
        return None
    return text[:10]


def ground_project_row(record: dict | None) -> dict:
    """Normalize one project row so cards and chat cite the same grounded facts."""
    if not isinstance(record, dict):
        return {}

    row = dict(record)

    price = _positive_price(row.get("price"))
    row["price"] = int(round(price)) if price is not None else None

    sold_raw = row.get("last_sold_on") or row.get("last_sale")
    sold_iso = _sale_date_iso(sold_raw)
    if sold_iso:
        row["last_sold_on"] = sold_iso
        if row.get("last_sale") is None:
            row["last_sale"] = sold_raw
    else:
        row["last_sold_on"] = None
        row["last_sale"] = None

    kind = str(row.get("price_kind") or "").strip().lower()
    if row["price"] is None:
        if kind in {"recent_sale", "latest_sale"}:
            row["price_kind"] = None
    elif kind in {"", "none", "null"}:
        # Sale-backed Prop-AI rows with a price are recent sales, not list prices.
        if sold_iso or str(row.get("source") or "").lower() == "propai":
            row["price_kind"] = "recent_sale"
        else:
            row["price_kind"] = None
    elif kind in {"recent_sale", "latest_sale", "starting_price", "list_price", "asking"}:
        row["price_kind"] = kind
    else:
        row["price_kind"] = kind or None

    return row


def ground_project_rows(records) -> list[dict]:
    return [
        ground_project_row(record)
        for record in (records or [])
        if isinstance(record, dict)
    ]
