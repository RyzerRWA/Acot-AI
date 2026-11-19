"""
Read-only listing tools for the AI agent.

The model never writes SQL. It picks one of these tools and fills in validated arguments; every query is
parameterised and read-only, and every number the model quotes comes from here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.database.postgres import fetch
from app.retrieval.place_lock import contains_place_sql, is_numbered_sibling

MIN_SQFT = 100  # sizes outside this range are data errors (a studio claiming 4,000,000 sqft)
MAX_SQFT = 50_000

# People say "JVC" or "Marina"; the data says "Jumeirah Village Circle (JVC)" or "Dubai Marina".
ALIASES = {
    "jvc": "Jumeirah Village Circle",
    "jvt": "Jumeirah Village Triangle",
    "jlt": "Jumeirah Lake Towers",
    "jbr": "Jumeirah Beach Residence",
    "dso": "Dubai Silicon Oasis",
    "dip": "Dubai Investments Park",
    "bb": "Business Bay",
    "marina": "Dubai Marina",
    "downtown": "Downtown Dubai",
    "palm": "Palm Jumeirah",
    "dubai hills": "Dubai Hills Estate",
    "arabian ranches": "Arabian Ranches",
    "creek": "Dubai Creek Harbour",
    "barsha": "Al Barsha",
    "sports city": "Dubai Sports City",
}

SORTS = {
    "cheapest": "price_aed ASC, listing_id",
    "priciest": "price_aed DESC, listing_id",
    "largest": "size_sqft DESC NULLS LAST, listing_id",
    "best_value": "(price_aed / NULLIF(size_sqft, 0)) ASC NULLS LAST, listing_id",
    "newest": "listed_at DESC NULLS LAST, listing_id",
}
GROUPS = {"community": "community", "beds": "beds", "property_type": "property_type", "furnishing": "furnishing"}

COLUMNS = ("listing_id, title, property_type, price_aed, beds, baths, size_sqft, furnishing, "
           "community, sub_community, agency_name, stored_photos")


class ToolInputError(ValueError):
    """The model sent bad arguments. The message goes back to it so it can correct itself."""


@dataclass
class ToolResult:
    output: Any
    cards: list[dict] = field(default_factory=list)  # full cards for the app; never shown to the model


# ---------------------------------------------------------------- argument validation

def _obj(raw: Any) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ToolInputError("Arguments must be an object.")
    return raw


_BED_WORDS = {
    "studio": 0, "studios": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_BED_TOKEN = re.compile(
    r"\b(studio|studios|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*-?\s*"
    r"(?:bed(?:room)?s?|br)\b",
    re.I,
)
_BROWSE_NAME = re.compile(
    r"\b(?:show|find|list|tell|apartments?|bedrooms?|propert(?:y|ies)|projects?|"
    r"villas?|rentals?|listings?)\b",
    re.I,
)


def _num(args: dict, key: str, lo: float, hi: float):
    value = args.get(key)
    if value is None or value == "":
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{key} must be a number between {lo:g} and {hi:g}.") from None
    if n != n or n < lo or n > hi:
        raise ToolInputError(f"{key} must be a number between {lo:g} and {hi:g}.")
    return n


def _beds(args: dict, key: str, lo: float = 0, hi: float = 12):
    """Bedroom counts: 2, '2', 'two' / 'two bedrooms', or [2] from the model."""
    value = args.get(key)
    if value is None or value == "":
        return None
    # Models often send bedrooms as [2] or {"min": 2}.
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None or value == "":
            return None
    if isinstance(value, dict):
        value = (
            value.get("min")
            if value.get("min") is not None
            else value.get("max")
            if value.get("max") is not None
            else value.get("value")
            if value.get("value") is not None
            else value.get(key)
        )
        if value is None or value == "":
            return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _BED_WORDS:
            value = _BED_WORDS[text]
        else:
            match = _BED_TOKEN.search(text) or re.fullmatch(
                r"(studio|studios|one|two|three|four|five|six|seven|eight|nine|ten|\d+)",
                text,
            )
            if match:
                token = match.group(1).lower()
                value = _BED_WORDS.get(token, token)
    return _num({key: value}, key, lo, hi)


def _place_from_text(text: str) -> str | None:
    lowered = (text or "").lower()
    for key in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            return ALIASES[key]
    for alias in dict.fromkeys(ALIASES.values()):
        if alias.lower() in lowered:
            return alias
    return None


def _normalize_project_search(a: dict) -> dict:
    """Turn a browse phrase the model put in `name` into real filters."""
    a = dict(a)
    if a.get("bedrooms") is not None:
        beds = _beds(a, "bedrooms")
        if beds is not None:
            a["bedrooms"] = int(beds)
    # Models sometimes send area/community under rental-style keys.
    if not a.get("communities"):
        for key in ("areas", "community", "area", "place"):
            if a.get(key) in (None, "", []):
                continue
            places = _strings(a, key)
            if places:
                a["communities"] = places
                break
    name = str(a.get("name") or "").strip()
    if not name or not _BROWSE_NAME.search(name):
        return a
    if a.get("bedrooms") is None:
        match = _BED_TOKEN.search(name)
        if match:
            token = match.group(1).lower()
            a["bedrooms"] = _BED_WORDS[token] if token in _BED_WORDS else int(token)
    if not a.get("communities"):
        place = _place_from_text(name)
        if place:
            a["communities"] = [place]
    a["name"] = None
    return a


def _strings(args: dict, key: str):
    value = args.get(key)
    if value is None:
        return None
    items = [str(v).strip() for v in (value if isinstance(value, list) else [value]) if str(v).strip()]
    if len(items) > 8 or any(len(v) > 80 for v in items):
        raise ToolInputError(f"{key} accepts at most 8 values of up to 80 characters.")
    return items or None


def parse_filters(raw: Any) -> dict:
    """Validated filters. Monthly budgets are converted to yearly, because the data is stored per year."""
    a = _obj(raw)

    def monthly(key):
        v = _num(a, key, 0, 10_000_000)
        return None if v is None else v * 12

    furnishing = a.get("furnishing")
    furnishing = None if furnishing is None else str(furnishing).lower()
    if furnishing not in (None, "furnished", "unfurnished"):
        raise ToolInputError('furnishing must be "furnished" or "unfurnished".')

    def first(*values):
        return next((v for v in values if v is not None), None)

    f = {
        "areas": _strings(a, "areas"),
        "beds_min": _beds(a, "beds_min", 0, 10),
        "beds_max": _beds(a, "beds_max", 0, 10),
        "baths_min": _num(a, "baths_min", 0, 10),
        "property_types": _strings(a, "property_types"),
        "furnishing": furnishing,
        "min_annual_rent": first(_num(a, "min_annual_rent", 0, 100_000_000), monthly("min_monthly_rent")),
        "max_annual_rent": first(_num(a, "max_annual_rent", 0, 100_000_000), monthly("max_monthly_rent")),
        "min_size_sqft": _num(a, "min_size_sqft", 0, 1_000_000),
        "max_size_sqft": _num(a, "max_size_sqft", 0, 1_000_000),
    }
    if f["beds_min"] is not None and f["beds_max"] is not None and f["beds_min"] > f["beds_max"]:
        raise ToolInputError("beds_min cannot exceed beds_max.")
    lo, hi = f["min_annual_rent"], f["max_annual_rent"]
    if lo is not None and hi is not None and lo > hi:
        raise ToolInputError("The minimum rent cannot exceed the maximum rent.")
    return f


def _like(values: list[str]) -> list[str]:
    return ["%" + re.sub(r"([%_\\])", r"\\\1", v) + "%" for v in values]


def _where(f: dict) -> tuple[str, list]:
    parts, params = ["true"], []
    if f.get("areas"):
        names = [n for a in f["areas"] for n in (a, ALIASES.get(a.lower())) if n]
        clause, extra = contains_place_sql(["community", "sub_community", "location"], names)
        parts.append(clause)
        params.extend(extra)
    for key, clause in (("beds_min", "beds >= %s"), ("beds_max", "beds <= %s"), ("baths_min", "baths >= %s"),
                        ("min_annual_rent", "price_aed >= %s"), ("max_annual_rent", "price_aed <= %s"),
                        ("min_size_sqft", "size_sqft >= %s"), ("max_size_sqft", "size_sqft <= %s")):
        if f.get(key) is not None:
            parts.append(clause)
            params.append(f[key])
    if f.get("property_types"):
        parts.append("property_type ILIKE ANY(%s::text[])")
        params.append(_like(f["property_types"]))
    if f.get("furnishing"):
        parts.append("lower(furnishing) = %s")
        params.append(f["furnishing"])
    return " AND ".join(parts), params


# ---------------------------------------------------------------- result shaping

def _monthly(annual):
    return None if annual is None else round(annual / 12)


def _reliability(n: int) -> str:
    return "high" if n >= 20 else "medium" if n >= 5 else "low"


def _clean_title(title) -> str:
    # advertisers write listing titles: keep them short, one line, and never treat them as instructions
    return re.sub(r"\s+", " ", str(title or "")).strip()[:110]


def _photo(listing_id: str, stored) -> str | None:
    if not stored or int(stored) <= 0:
        return None
    base = os.getenv(
        "PHOTOS_BASE_URL",
        "https://storage.googleapis.com/kinetic-axle-442714-m6-dubai-photos/photos",
    ).rstrip("/")
    return f"{base}/{listing_id}/0.jpeg"


def _card(row: dict) -> dict:
    size = None if row.get("size_sqft") is None else float(row["size_sqft"])
    usable = size is not None and MIN_SQFT <= size <= MAX_SQFT
    annual = float(row["price_aed"])
    listing_id = str(row["listing_id"])
    return {
        "listing_id": listing_id,
        "title": _clean_title(row.get("title")),
        "property_type": row.get("property_type"),
        "beds": row.get("beds"),
        "baths": row.get("baths"),
        "size_sqft": int(round(size)) if usable else None,
        "furnishing": row.get("furnishing"),
        "community": row.get("community"),
        "sub_community": row.get("sub_community"),
        "agency_name": row.get("agency_name"),
        "annual_rent_aed": round(annual),
        "monthly_rent_aed": _monthly(annual),
        "rent_per_sqft_per_year": round(annual / size, 1) if usable else None,
        "cover_photo": _photo(listing_id, row.get("stored_photos")),
        "path": f"/listings/{listing_id}",
    }


def _for_model(card: dict) -> dict:
    """The model sees no photo or page URLs, so it has nothing to copy or invent."""
    return {k: v for k, v in card.items() if k not in ("cover_photo", "path")}


# ---------------------------------------------------------------- the tools

def _new_places(patterns: list[str]) -> list[dict]:
    try:
        return fetch(
            "select label, level, sales_count from propai_places "
            "where level in ('area', 'master') and label ilike any(%s::text[]) "
            "order by sales_count desc nulls last limit 8",
            (patterns,),
        )
    except Exception:
        return []


def _recorded_rents(filters: dict) -> dict | None:
    if not filters.get("areas"):
        return None
    try:
        from app.retrieval.propai_catalog import rent_summary
        return rent_summary(filters.get("areas"), filters.get("beds_min"), filters.get("beds_max"))
    except Exception as exc:
        print(f"Prop-AI rent summary unavailable: {exc}")
        return None


def _new_projects(filters: dict, limit: int, sort: str, skip_names=None) -> tuple[list[dict], int]:
    try:
        from app.retrieval.propai_catalog import agent_projects
        return agent_projects(filters, limit=limit, sort=sort, skip_names=skip_names)
    except Exception as exc:
        print(f"Prop-AI project search unavailable: {exc}")
        return [], 0


def find_areas(raw) -> ToolResult:
    """Place lookup via propai_places."""
    a = _obj(raw)
    text = str(a.get("text") or "").strip()[:80]
    if not text:
        raise ToolInputError("text is required.")
    names = [n for n in (text, ALIASES.get(text.lower())) if n]
    patterns = _like(names)
    places = [
        row for row in _new_places(patterns)
        if not is_numbered_sibling(text, row.get("label") or "")
    ]
    if places:
        return ToolResult({
            "matches": places,
            "new_places": places,
            "note": (
                f'Prop-AI places matching "{text}". '
                "Call search_projects for projects or search_listings for registered rents."
            ),
        })
    return ToolResult({
        "matches": [],
        "note": f'No Prop-AI place matching "{text}".',
    })


def search_listings(raw) -> ToolResult:
    """Browse registered rents from propai_rents (main Prop-AI rent data)."""
    from app.retrieval.propai_catalog import list_rents

    a = _obj(raw)
    f = parse_filters(a)
    # Newest rent contracts first unless the agent/user asked cheap/pricey.
    sort = str(a.get("sort") or "newest")
    if sort not in SORTS and sort not in {"newest"}:
        sort = ["newest","Cheapest"]
    limit = int(round(_num(a, "limit", 1, 10) or 5))
    found = list_rents(
        f.get("areas"),
        f.get("beds_min"),
        f.get("beds_max"),
        limit=limit,
        sort=sort,
    )
    rows = found.get("listings") or []
    cards = []
    for row in rows:
        # Shape expected by the agent card builder.
        cards.append({
            "listing_id": row.get("listing_id") or "",
            "title": row.get("title"),
            "property_type": row.get("property_type"),
            "beds": row.get("beds"),
            "baths": None,
            "size_sqft": row.get("size_sqft"),
            "furnishing": None,
            "community": row.get("community"),
            "sub_community": row.get("sub_community"),
            "agency_name": None,
            "annual_rent_aed": row.get("annual_rent_aed"),
            "monthly_rent_aed": row.get("monthly_rent_aed"),
            "cover_photo": None,
            "path": None,
            "kind": "listing",
            "row": row,
        })
    output = {
        "matching_listings": found.get("matching_listings") or 0,
        "showing": len(cards),
        "sorted_by": found.get("sorted_by") or sort,
        "listings": [
            {
                "listing_id": c.get("listing_id"),
                "title": c.get("title"),
                "property_type": c.get("property_type"),
                "beds": c.get("beds"),
                "size_sqft": c.get("size_sqft"),
                "community": c.get("community"),
                "annual_rent_aed": c.get("annual_rent_aed"),
                "monthly_rent_aed": c.get("monthly_rent_aed"),
                "contract_start": (c.get("row") or {}).get("contract_start"),
            }
            for c in cards
        ],
        "note": found.get("note"),
    }
    recorded = _recorded_rents(f)
    if recorded:
        output["recorded_rents"] = recorded
    return ToolResult(output, cards)


def area_stats(raw) -> ToolResult:
    """Rent aggregates from propai_rents."""
    from app.retrieval.propai_catalog import rent_stats

    a = _obj(raw)
    f = parse_filters(a)
    stats = rent_stats(f.get("areas"), f.get("beds_min"), f.get("beds_max"))
    if not stats:
        return ToolResult({
            "grouped_by": "none",
            "rows": [],
            "note": "No Prop-AI rent stats for that place. Name a Dubai area.",
        })
    row = {
        "listings": stats.get("listings"),
        "reliability": _reliability(int(stats.get("listings") or 0)),
        "avg_annual_rent_aed": stats.get("avg_annual_rent_aed"),
        "avg_monthly_rent_aed": _monthly(stats.get("avg_annual_rent_aed")),
        "median_annual_rent_aed": stats.get("median_annual_rent_aed"),
        "median_monthly_rent_aed": _monthly(stats.get("median_annual_rent_aed")),
        "typical_range_annual_aed": stats.get("typical_range_annual_aed"),
        "min_annual_rent_aed": stats.get("min_annual_rent_aed"),
        "max_annual_rent_aed": stats.get("max_annual_rent_aed"),
    }
    return ToolResult({
        "grouped_by": "none",
        "rows": [row],
        "note": stats.get("note") or "Registered Prop-AI rents.",
    })


def get_listing(raw) -> ToolResult:
    """One registered rent contract from propai_rents by pai_contract_id."""
    listing_id = str(_obj(raw).get("listing_id") or "").strip()
    if not listing_id:
        raise ToolInputError("listing_id must be the Prop-AI contract id.")
    rows = fetch(
        """
        select pai_contract_id, property_type, bedrooms, actual_area_sqft, annual_amount,
               contract_start_date, area_name_prop_ai, master_project_prop_ai,
               project_name_prop_ai, building_name_prop_ai
        from propai_rents
        where deleted_at is null and pai_contract_id = %s
        limit 1
        """,
        (listing_id,),
    )
    if not rows:
        return ToolResult({
            "found": False,
            "note": f"No Prop-AI rent contract with id {listing_id}.",
        })
    from app.retrieval.propai_catalog import _rent_row_public

    pub = _rent_row_public(rows[0])
    card = {
        "listing_id": pub.get("listing_id") or listing_id,
        "title": pub.get("title"),
        "property_type": pub.get("property_type"),
        "beds": pub.get("beds"),
        "size_sqft": pub.get("size_sqft"),
        "community": pub.get("community"),
        "annual_rent_aed": pub.get("annual_rent_aed"),
        "monthly_rent_aed": pub.get("monthly_rent_aed"),
        "kind": "listing",
        "row": pub,
    }
    return ToolResult({"found": True, "listing": pub}, [card])


def dataset_info(raw=None) -> ToolResult:
    """Prop-AI dataset counts (projects + registered rents)."""
    _ = raw
    try:
        rents = fetch(
            "select count(*)::int as n, "
            "round(percentile_cont(0.5) within group (order by annual_amount))::bigint as median "
            "from propai_rents where deleted_at is null and annual_amount > 0 "
            "and contract_start_date >= date '2025-09-01'"
        )[0]
        projects = fetch(
            "select count(*)::int as n from propai_projects where coalesce(name, '') <> ''"
        )[0]["n"]
        sales = fetch(
            "select count(*)::int as n from propai_sales "
            "where deleted_at is null and sale_price > 0 "
            "and instance_date >= current_date - interval '24 months'"
        )[0]["n"]
    except Exception:
        return ToolResult({"note": "Prop-AI dataset info unavailable."})
    places = _available_places()
    return ToolResult({
        "total_listings": rents.get("n") or 0,
        "projects": projects,
        "sales_24m": sales,
        "areas": places.get("communities_with_projects") or [],
        "median_annual_rent_aed": None if rents.get("median") is None else int(rents["median"]),
        "median_monthly_rent_aed": _monthly(rents.get("median")),
        "what_this_is": (
            "Prop-AI registered rents, projects, sales, brokers, places, "
            "project plans and status."
        ),
        "not_included": "Old Bayut advertised listings / communities / document_chunks.",
    })


# ---------------------------------------------------------------- project tools
# The projects table holds off-plan / new-launch projects. Its price is a STARTING price for the project,
# never a price for a particular unit. Cards for the app carry the raw database row (kind "project").

PROJECT_COLUMNS = (
    "id, slug, name, description, status, project_status, pool_type, developer_id, developer_name, city, "
    "community, sub_community, price, size_min, size_max, bedroom_min, bedroom_max, handover_time, photos, "
    "property_types, amenities, brochure_url, property_finder_url, source"
)
PROJECT_SORTS = {
    # A price of 0 means "not given" in this data, so it must never rank as the cheapest.
    "cheapest": "NULLIF(price, 0) ASC NULLS LAST, id",
    "priciest": "NULLIF(price, 0) DESC NULLS LAST, id",
    "earliest_handover": "handover_time ASC NULLS LAST, id",
    "latest_handover": "handover_time DESC NULLS LAST, id",
    "largest": "size_max DESC NULLS LAST, id",
    "more": "handover_time DESC NULLS LAST, id",
}
DEFAULT_PROJECT_SORT = "latest_handover"
PROJECT_GROUPS = {"community": "community", "developer": "btrim(developer_name)"}
_NAME_STOPWORDS = {"by", "at", "the", "of", "in", "and", "a"}
MAX_PROFILE_NOTE = 200


def _project_filters(a: dict) -> dict:
    f = {
        "communities": _strings(a, "communities"),
        "developers": _strings(a, "developers"),
        "name": (str(a.get("name") or "").strip()[:80] or None),
        "min_price": _num(a, "min_price_aed", 0, 10_000_000_000),
        "max_price": _num(a, "max_price_aed", 0, 10_000_000_000),
        "bedrooms": _beds(a, "bedrooms", 0, 12),
        "handover_year": _num(a, "handover_year", 2000, 2100),
        "property_types": _strings(a, "property_types"),
        "min_size": _num(a, "min_size_sqft", 0, 1_000_000),
        "max_size": _num(a, "max_size_sqft", 0, 1_000_000),
    }
    if f["min_price"] is not None and f["max_price"] is not None and f["min_price"] > f["max_price"]:
        raise ToolInputError("min_price_aed cannot exceed max_price_aed.")
    return f


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if t not in _NAME_STOPWORDS]


def _project_where(f: dict) -> tuple[str, list]:
    parts, params = ["true"], []
    if f.get("communities"):
        names = [n for c in f["communities"] for n in (c, ALIASES.get(c.lower())) if n]
        clause, extra = contains_place_sql(["community", "sub_community"], names)
        parts.append(clause)
        params.extend(extra)
    if f.get("developers"):
        parts.append("developer_name ILIKE ANY(%s::text[])")
        params.append(_like(f["developers"]))
    for token in _name_tokens(f.get("name") or ""):
        parts.append("name ILIKE %s")
        params.append(_like([token])[0])
    if f.get("min_price") is not None or f.get("max_price") is not None:
        parts.append("price > 0")  # price 0 = not given; it must not pass a budget filter
    if f.get("min_price") is not None:
        parts.append("price >= %s")
        params.append(f["min_price"])
    if f.get("max_price") is not None:
        parts.append("price <= %s")
        params.append(f["max_price"])
    if f.get("bedrooms") is not None:
        parts.append("bedroom_min <= %s AND bedroom_max >= %s")
        params += [f["bedrooms"]] * 2
    if f.get("min_size") is not None:  # some unit in the project is at least this big
        parts.append("size_max >= %s")
        params.append(f["min_size"])
    if f.get("max_size") is not None:  # some unit is no bigger than this
        parts.append("size_min > 0 AND size_min <= %s")
        params.append(f["max_size"])
    if f.get("handover_year") is not None:
        parts.append("extract(year from handover_time at time zone 'UTC') = %s")
        params.append(f["handover_year"])
    if f.get("property_types"):
        parts.append("property_types && %s::text[]")
        params.append([v.upper() for v in f["property_types"]])
    return " AND ".join(parts), params


def _project_for_model(row: dict) -> dict:
    """What the model may quote. No photos, URLs, amenity codes or marketing essay."""
    from app.answers.structured_answer import _format_bedroom_phrase, _format_handover_month
    def size(value):
        # sizes outside a believable range are data errors (a project "from 0 sqft"); show them as unknown
        return None if value is None or not MIN_SQFT <= float(value) <= MAX_SQFT else float(value)

    price = row.get("price")
    return {
        "id": str(row["id"]),
        "name": row.get("name"),
        "developer": (row.get("developer_name") or "").strip() or None,
        "community": row.get("community"),
        "sub_community": row.get("sub_community") or None,
        # 0 or empty means the price was not given: report it as unknown, never as AED 0
        "starting_price_aed": int(price) if price and price > 0 else None,
        "bedrooms": _format_bedroom_phrase(row),
        "size_sqft_min": size(row.get("size_min")),
        "size_sqft_max": size(row.get("size_max")),
        "handover": _format_handover_month(row.get("handover_time")),
        # Prop-AI rows often omit property_types; null must not be read as "not an apartment".
        "property_types": row.get("property_types") or None,
        "property_types_note": (
            None
            if row.get("property_types")
            else "Type not recorded; treat as a residential project match for apartment browses."
        ),
        "status": row.get("project_status") or row.get("status"),
        "has_brochure": bool(row.get("brochure_url")),
        "price_kind": row.get("price_kind") or "starting",
        # The key above says "starting" for both kinds; this is the wording to put before the price.
        "price_label": "recent sale price" if row.get("price_kind") == "recent_sale" else "from",
    }


def _project_card(row: dict) -> dict:
    return {"listing_id": str(row["id"]), "kind": "project", "row": row}


def _available_places() -> dict:
    """Shown when a search finds nothing, so the model can offer real alternatives instead of a dead end."""
    return {
        "communities_with_projects": fetch(
            "select coalesce(nullif(btrim(master_project), ''), area) as community, "
            "count(*)::int as projects from propai_projects "
            "where coalesce(name, '') <> '' "
            "and coalesce(nullif(btrim(master_project), ''), area) is not null "
            "group by 1 order by projects desc"
        ),
        "developers": fetch(
            "select btrim(developer) as developer, count(*)::int as projects "
            "from propai_projects where coalesce(developer, '') <> '' "
            "group by 1 order by projects desc"
        ),
    }


_known_names: set[str] | None = None


def mentions_known_name(text: str) -> bool:
    """True when the text names a community, project, developer or common short form that is in our data."""
    global _known_names
    if _known_names is None:
        try:
            rows = fetch("select lower(btrim(coalesce(nullif(btrim(master_project), ''), area))) as n "
                         "from propai_projects union "
                         "select lower(btrim(developer)) from propai_projects union "
                         "select lower(btrim(name)) from propai_projects")
            extra = fetch("select lower(btrim(label)) as n from propai_places "
                          "where level in ('area', 'master')")
            rows = list(rows) + list(extra)
        except Exception:
            return False  # no database: do not force anything
        _known_names = {r["n"] for r in rows if r.get("n") and len(r["n"]) >= 4}
        _known_names |= {alias for alias in ALIASES if len(alias) >= 3}
    lowered = (text or "").lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered) for name in _known_names)


_place_names: list[str] | None = None


def _closest_place(text: str) -> str | None:
    """A known community for a misspelt one ("arjun" -> "Arjan"), or None when nothing is close."""
    import difflib
    global _place_names
    if _place_names is None:
        try:
            rows = list(fetch("select distinct btrim(label) as n from propai_places where level in ('area', 'master')"))
        except Exception:
            return None  # no database: leave the name as typed
        _place_names = sorted({r["n"] for r in rows if r.get("n")})
    lowered = {name.lower(): name for name in _place_names}
    hit = difflib.get_close_matches((text or "").lower().strip(), list(lowered), n=1, cutoff=0.8)
    return lowered[hit[0]] if hit and hit[0] != (text or "").lower().strip() else None


def search_projects(raw) -> ToolResult:
    """Project search that also reads a misspelt place the way the user meant it."""
    a = _normalize_project_search(_obj(raw))
    result = _search_projects(a)
    if result.output.get("matching_projects"):
        return result
    # Nothing found: a typo'd community, or a place sent as a developer ("arjun projects").
    fixed, read_as = dict(a), {}
    for key in ("communities", "developers"):
        for value in _strings(a, key) or []:
            place = _closest_place(ALIASES.get(value.lower(), value))
            if place:
                read_as[value] = place
                fixed[key] = [v for v in (fixed.get(key) or []) if v != value]
                fixed["communities"] = [*(fixed.get("communities") or []), place]
    if not read_as:
        return result
    retry = _search_projects(fixed)
    if not retry.output.get("matching_projects"):
        return result
    retry.output["read_as"] = read_as
    retry.output["read_as_note"] = ("The user's place name matched nothing as typed. It was read as the place "
                                    "shown in read_as. This is a match, not a no-match: never say \"couldn't find\"; "
                                    "open with \"Showing results for Arjan:\" (the read_as place) and answer.")
    return retry


def _project_sort(raw) -> str:
    sort = str(raw or DEFAULT_PROJECT_SORT).strip().lower()
    if sort in PROJECT_SORTS:
        return sort
    if sort in {"newest", "new", "recent", "relevance", "best", "recommended"}:
        return DEFAULT_PROJECT_SORT
    if not re.fullmatch(r"[a-z_]+", sort or ""):
        raise ToolInputError(f"sort must be one of: {', '.join(PROJECT_SORTS)}.")
    return DEFAULT_PROJECT_SORT


def _project_limit(raw) -> int:
    if raw is None or raw == "":
        return 8
    try:
        return max(1, min(int(round(float(raw))), 10))
    except (TypeError, ValueError):
        return 8


def _search_projects(a: dict) -> ToolResult:
    f = _project_filters(a)
    sort = _project_sort(a.get("sort"))
    limit = _project_limit(a.get("limit"))
    raw_skip = a.get("exclude_names")
    if raw_skip is None:
        skip_names = []
    else:
        items = raw_skip if isinstance(raw_skip, list) else [raw_skip]
        skip_names = [str(name).strip() for name in items if str(name).strip()][:20]
    rows, total = _new_projects(f, limit, sort, skip_names=skip_names)
    projects_out = []
    for row in rows:
        shown = _project_for_model(row)
        if row.get("sales_count") is not None:
            shown["sales_count"] = int(row["sales_count"])
        projects_out.append(shown)
    out = {
        "matching_projects": total, "showing": len(rows), "sorted_by": sort,
        "source": "propai",
        "price_note": "price_kind recent_sale is the median sale price from the last 24 months in recorded sales data.",
        "projects": projects_out,
        "browse_note": (
            "These are recorded projects. List them now. Do not call search_projects again, "
            "and do not call search_listings unless the user asked for listings. "
            "Do NOT add Acot Recommend unless the user asked which is best, invest, buy, or recommend. "
            "Never say Prop-AI in the user-facing answer."
        ),
    }
    if total > 0 and f.get("bedrooms") is not None:
        out["bedroom_match_note"] = (
            f"matching_projects={total} means these projects include {int(f['bedrooms'])}-bedroom units "
            "(bedroom ranges like 1 to 4 or studios to 2 count as matches). "
            "Present them as normal results. Do not say there was no exact match."
        )
    cards = [_project_card(r) for r in rows]
    if total == 0:
        out["note"] = (
            "No project matches these filters. Say so. Offer to loosen a filter. "
            "Do not call search_listings and do not use the older project table. "
            "Only call a result 'over budget' if the user actually gave a budget. "
            "Never say Prop-AI in the user-facing answer."
        )
        closest = _closest_projects(f)
        if closest:
            out["closest_matches"] = [_project_for_model(r) | _gap(r, f) for r in closest]
            out["closest_matches_note"] = ("These are the nearest projects after relaxing: "
                                           f"{closest[0]['_relaxed']}. Present them as the closest options, "
                                           "say plainly that they do not fully match, and quote the gap given. "
                                           "Never say Prop-AI in the user-facing answer."
                                           + ("" if f.get("max_price") else
                                              " The user gave no budget, so do not say anything is over budget "
                                              "or compute a gap."))
            cards += [_project_card({k: v for k, v in r.items() if k != "_relaxed"}) for r in closest]
    return ToolResult(out, cards)


def _closest_projects(f: dict, limit: int = 3) -> list[dict]:
    """Nearest real projects when nothing matches: drop one filter at a time, in the order people give way."""
    attempts = (
        ("the budget", {"min_price": None, "max_price": None}),
        ("the size", {"min_size": None, "max_size": None}),
        ("the bedroom count", {"bedrooms": None}),
        ("the handover year", {"handover_year": None}),
        ("the property type", {"property_types": None}),
        ("the place", {"communities": None}),
    )
    loosened, dropped = dict(f), []
    for label, drop in attempts:
        if all(f.get(key) is None for key in drop):
            continue  # that filter was not used, so relaxing it changes nothing
        loosened.update(drop)  # each step keeps the earlier relaxations, so the match gets looser and looser
        dropped.append(label)
        label = " and ".join(dropped)
        new_rows, _ = _new_projects(loosened, limit, "latest_handover")
        if new_rows:
            return [{**r, "_relaxed": label} for r in new_rows]
    return []


def _gap(row: dict, f: dict) -> dict:
    """How far a near-miss is from the budget, computed here so the model never does the arithmetic."""
    price, cap = row.get("price"), f.get("max_price")
    if price and cap and price > cap:
        return {"above_budget_by_aed": int(price - cap)}
    return {}


def _propai_matches(ref: str, limit: int) -> list[dict]:
    from app.retrieval.propai_catalog import agent_by_id, agent_projects
    if re.fullmatch(r"P\d{12}", ref):
        row = agent_by_id(ref)
        return [row] if row else []
    rows, _ = agent_projects({"name": ref}, limit=limit)
    exact = [row for row in rows if str(row.get("name") or "").lower() == ref.lower()]
    return exact or rows


def _find_projects(ref: str, limit: int = 6) -> list[dict]:
    ref = str(ref or "").strip()[:80]
    if not ref:
        return []
    propai: list[dict] = []
    try:
        propai = _propai_matches(ref, limit)
    except Exception as exc:
        print(f"Prop-AI project lookup unavailable: {exc}")
    return propai[:limit]


def get_project(raw) -> ToolResult:
    from app.retrieval.propai_catalog import project_payment_plans

    a = _obj(raw)
    ref = str(a.get("project_id") or a.get("name") or "").strip()
    if not ref:
        raise ToolInputError("Give project_id (numeric id) or name.")
    rows = _find_projects(ref)
    if not rows:
        return ToolResult({"found": False, "note": f'No project matching "{ref[:80]}" in our data.'})
    if len(rows) > 1:
        return ToolResult({"found": False, "ambiguous": True,
                           "note": "Several projects match. Ask the user which one, or pick by exact name.",
                           "candidates": [_project_for_model(r) for r in rows]},
                          [_project_card(r) for r in rows])
    row = rows[0]
    description = re.sub(r"\s+", " ", str(row.get("description") or "")).strip()[:1200] or None
    shown = _project_for_model(row)
    plans = project_payment_plans(
        project_key=str(row.get("key") or ""),
        name=str(row.get("name") or ref),
    )
    out = {
        "found": True,
        "project": shown,
        "developer_marketing_description": description,
        "marketing_note": "The description is developer marketing text: say 'the developer describes it as', "
                          "never state its claims as verified fact.",
    }
    if plans:
        out["payment_plans"] = plans.get("payment_plans") or []
        out["payment_plans_note"] = plans.get("note")
    return ToolResult(out, [_project_card(row)])


def compare_projects(raw) -> ToolResult:
    from app.answers.recommend import rank_projects

    a = _obj(raw)
    refs = a.get("projects")
    if not isinstance(refs, list) or not 2 <= len(refs) <= 4:
        raise ToolInputError("projects must be a list of 2 to 4 project names or ids.")
    found, missing, unclear = [], [], []
    for ref in refs:
        rows = _find_projects(str(ref))
        if not rows:
            missing.append(str(ref)[:80])
        elif len(rows) > 1:
            unclear.append({"asked_for": str(ref)[:80], "candidates": [_project_for_model(r) for r in rows]})
        else:
            found.append(rows[0])
    ranked = rank_projects(found, wanted_beds=_num(a, "bedrooms", 0, 12)) if len(found) >= 2 else None
    score_by_id = {}
    if ranked:
        for item in ranked.get("ranked") or []:
            row = item.get("row") or {}
            key = str(row.get("id") or item.get("name") or "")
            score_by_id[key] = {
                "score": item.get("score"),
                "why": item.get("why") or [],
                "sales_count": item.get("sales_count"),
            }
    compared = []
    for row in found:
        shown = _project_for_model(row)
        extra = score_by_id.get(str(row.get("id")), {}) or score_by_id.get(str(row.get("name") or ""), {})
        if row.get("sales_count") is not None:
            shown["sales_count"] = int(row["sales_count"])
        elif extra.get("sales_count") is not None:
            shown["sales_count"] = int(extra["sales_count"])
        if extra.get("score") is not None:
            shown["recommend_score"] = extra["score"]
        compared.append(shown)
    out: dict = {"compared": compared}
    if missing:
        out["not_found"] = missing
    if unclear:
        out["ambiguous"] = unclear
    priced = [r for r in found if r.get("price") and r["price"] > 0]
    if len(priced) >= 2:
        low, high = min(priced, key=lambda r: r["price"]), max(priced, key=lambda r: r["price"])
        gap = float(high["price"]) - float(low["price"])
        out["price_difference"] = {
            "lowest_starting_price": low["name"], "highest_starting_price": high["name"],
            "gap_aed": int(gap), "highest_is_percent_more": round(100 * gap / float(low["price"]), 1),
            "note": "price_kind recent_sale is a median from recorded sales. Never say Prop-AI to the user.",
        }
    dated = [r for r in found if r.get("handover_time") is not None]
    if len(dated) >= 2:
        first = min(dated, key=lambda r: r["handover_time"])
        out["earliest_handover"] = {"project": first["name"],
                                    "handover": _project_for_model(first)["handover"]}
    if ranked and ranked.get("recommendation") is not None:
        out["recommendation"] = ranked["recommendation"]
        out["recommendation_note"] = (
            "Answer with a markdown side-by-side table of ONLY these compared projects, "
            "then end with **Acot Recommend:** and Why (price, size, sales/transactions, rentals). "
            "Keep the tone positive and clean; do not invent a different winner."
        )
    return ToolResult(out, [_project_card(r) for r in found])


def market_stats(raw) -> ToolResult:
    a = _obj(raw)
    f = _project_filters(a)
    group_key = str(a.get("group_by") or "none")
    if group_key != "none" and group_key not in PROJECT_GROUPS:
        raise ToolInputError(f"group_by must be one of: none, {', '.join(PROJECT_GROUPS)}.")
    parts, params = ["coalesce(name, '') <> ''"], []
    if f.get("communities"):
        names = [n for c in f["communities"] for n in (c, ALIASES.get(c.lower())) if n]
        clause, extra = contains_place_sql(["master_project", "area", "name"], names)
        parts.append(clause)
        params.extend(extra)
    if f.get("developers"):
        parts.append("developer ilike any(%s::text[])")
        params.append(_like(f["developers"]))
    for token in _name_tokens(f.get("name") or ""):
        parts.append("name ilike %s")
        params.append(_like([token])[0])
    if f.get("handover_year") is not None:
        parts.append("extract(year from end_date) = %s")
        params.append(f["handover_year"])
    where = " and ".join(parts)
    group_sql = None
    if group_key == "community":
        group_sql = "coalesce(nullif(btrim(master_project), ''), area)"
    elif group_key == "developer":
        group_sql = "btrim(developer)"
    sql = (
        f"select {group_sql + ' as grp,' if group_sql else ''} count(*)::int as projects, "
        "count(*) filter (where end_date is not null)::int as with_handover "
        f"from propai_projects where {where}"
        + (f" group by {group_sql} order by count(*) desc limit 15" if group_sql else "")
    )
    rows = fetch(sql, tuple(params))
    years = fetch(
        "select extract(year from end_date)::int as year, count(*)::int as projects "
        f"from propai_projects where {where} and end_date is not null group by 1 order by 1",
        tuple(params),
    )
    out = []
    for r in rows:
        row = {group_key: r["grp"]} if group_sql else {}
        row.update({"projects": r["projects"], "reliability": _reliability(r["projects"]),
                    "min_starting_price_aed": None, "median_starting_price_aed": None,
                    "max_starting_price_aed": None,
                    "bedrooms_offered": None, "projects_with_handover_date": r["with_handover"]})
        out.append(row)
    result = {
        "grouped_by": group_key, "rows": out, "handover_years": years,
        "note": "Project counts are from recorded project data. new_data, when present, is recorded sales "
                "from the last 24 months and is the price to quote. There is no yield, ROI or price growth.",
    }
    if f.get("communities"):
        try:
            from app.retrieval.propai_catalog import sale_summary
            recorded = sale_summary(f["communities"])
        except Exception as exc:
            print(f"Prop-AI sale summary unavailable: {exc}")
            recorded = None
        if recorded:
            result["new_data"] = recorded
    return ToolResult(result)


def search_brochures(raw, pgvector=None) -> ToolResult:
    """Semantic search over Prop-AI project text (propai_raw / propai_project_status)."""
    from app.retrieval.propai_vector_search import search_propai

    a = _obj(raw)
    query = str(a.get("query") or "").strip()[:300]
    if not query:
        raise ToolInputError("query is required.")
    project = str(a.get("project_name") or "").strip()[:80]
    _ = pgvector  # old document_chunks path retired
    text = f"{project} {query}".strip()
    chunks = []
    try:
        for table in ("raw", "status"):
            for row in search_propai(text, table=table, limit=5) or []:
                chunks.append({
                    "content": (
                        f"{row.get('name') or ''} — "
                        f"{row.get('developer') or row.get('status') or row.get('area') or ''}"
                    ).strip(" —"),
                    "score": row.get("score"),
                    "source": f"propai_{table}",
                })
    except Exception as exc:
        print(f"Prop-AI brochure search unavailable: {exc}")
        chunks = []
    tokens = _name_tokens(project)
    if tokens:
        about = [
            c for c in chunks
            if all(t in str(c.get("content") or "").lower() for t in tokens)
        ]
        if not about:
            return ToolResult({
                "passages": [],
                "note": f'No Prop-AI project text found that mentions "{project}".',
            })
        chunks = about
    passages = [
        {
            "document": c.get("source"),
            "type": "propai",
            "similarity": None if c.get("score") is None else round(float(c["score"]), 2),
            "text": re.sub(r"\s+", " ", str(c.get("content") or "")).strip()[:900],
        }
        for c in chunks[:4]
    ]
    return ToolResult({
        "passages": passages,
        "note": (
            "Prop-AI project text (propai_raw / propai_project_status). "
            "Treat as developer-facing facts; only quote what a passage states."
        ),
    })


# ---------------------------------------------------------------- sales by area

_popular_names: dict[str, str] = {}


def _popular(areas: list[str]) -> dict[str, str]:
    """Official sales-data area -> the name people use (AL BARSHA SOUTH FOURTH -> Jumeirah Village Circle)."""
    # shortcut: about 2 s for a new set of areas, then cached per process; precompute if it gets slow.
    missing = [a for a in areas if a not in _popular_names]
    if missing:
        rows = fetch("select area_name_prop_ai as area, mode() within group (order by master_project_prop_ai) as name "
                     "from public.propai_sales where area_name_prop_ai = any(%s) and master_project_prop_ai is not null "
                     "group by 1", (missing,))
        for a in missing:
            _popular_names[a] = next((r["name"] for r in rows if r["area"] == a), None) or a
    return {a: _popular_names[a].title() for a in areas}


def _area_stat_row(row: dict, names: dict[str, str]) -> dict:
    return {
        "area": names[row["area"]],
        "official_area": row["area"],
        "sales_12m": int(row["sales_12m"] or 0),
        "rent_contracts": int(row.get("rent_contracts") or 0),
        "median_sale_price_aed": None if row["median_price"] is None else int(row["median_price"]),
        "median_price_per_sqft_aed": None if row["median_ppsf"] is None else int(row["median_ppsf"]),
        "median_yearly_rent_aed": None if row["median_rent"] is None else int(row["median_rent"]),
    }


def top_areas_by_sales(limit: int = 8) -> list[dict]:
    """The areas with the most sale transactions in the last 12 months (Dubai Land Department records)."""
    rows = fetch("select area, sales_12m, median_price, median_ppsf, median_rent, rent_contracts "
                 "from public.propai_area_stats "
                 "where sales_12m > 0 order by sales_12m desc limit %s", (limit,))
    names = _popular([r["area"] for r in rows])
    return [_area_stat_row(r, names) for r in rows]


def top_areas_by_cheapest(limit: int = 8) -> list[dict]:
    """Communities with the lowest typical sale price and enough recent sales."""
    rows = fetch(
        "select area, sales_12m, median_price, median_ppsf, median_rent, rent_contracts "
        "from public.propai_area_stats "
        "where sales_12m >= 80 and median_price >= 200000 "
        "order by median_price asc limit %s",
        (max(int(limit) * 4, 20),),
    )
    names = _popular([r["area"] for r in rows])
    items = []
    seen: set[str] = set()
    for row in rows:
        item = _area_stat_row(row, names)
        label = str(item.get("area") or "").lower()
        if "industrial" in label or "labour" in label:
            continue
        if label in seen:
            continue
        seen.add(label)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def top_areas_by_rent(limit: int = 8) -> list[dict]:
    """Areas with the most recorded rental contracts, plus typical sale and rent."""
    rows = fetch(
        "select area, sales_12m, median_price, median_ppsf, median_rent, rent_contracts "
        "from public.propai_area_stats "
        "where rent_contracts > 0 and median_rent > 0 "
        "order by rent_contracts desc limit %s",
        (limit,),
    )
    names = _popular([r["area"] for r in rows])
    return [_area_stat_row(r, names) for r in rows]


def area_sales(raw) -> ToolResult:
    a = _obj(raw)
    limit = int(_num(a, "limit", 3, 15) or 8)
    return ToolResult({
        "areas": top_areas_by_sales(limit),
        "note": "Dubai Land Department sale transactions in the last 12 months, all property types, ranked by count. "
                "'area' is the name people use; 'official_area' is the land-registry name. The app shows this list.",
    })


# ---------------------------------------------------------------- user profile

PROFILE_GOALS = ("buy_to_live", "invest", "rent_out", "holiday_home", "other")


def clean_profile(raw: Any) -> dict:
    """Validated user preferences. Used for what the user tells us and for what the app sends."""
    a = _obj(raw)
    out: dict = {}
    for key in ("budget_min_aed", "budget_max_aed"):
        v = _num(a, key, 0, 10_000_000_000)
        if v is not None:
            out[key] = int(v)
    if {"budget_min_aed", "budget_max_aed"} <= out.keys() and out["budget_min_aed"] > out["budget_max_aed"]:
        raise ToolInputError("budget_min_aed cannot exceed budget_max_aed.")
    beds = _beds(a, "bedrooms", 0, 12)
    if beds is not None:
        out["bedrooms"] = int(beds)
    family = _num(a, "family_size", 1, 30)
    if family is not None:
        out["family_size"] = int(family)
    goal = a.get("goal")
    if goal not in (None, ""):
        if goal not in PROFILE_GOALS:
            raise ToolInputError(f"goal must be one of: {', '.join(PROFILE_GOALS)}.")
        out["goal"] = goal
    areas = _strings(a, "preferred_areas")
    if areas:
        out["preferred_areas"] = areas
    notes = re.sub(r"\s+", " ", str(a.get("notes") or "")).strip()[:MAX_PROFILE_NOTE]
    if notes:
        out["notes"] = notes
    return out


def _broker_card(row: dict) -> dict:
    return {"listing_id": str(row.get("brn") or row.get("id") or ""), "kind": "broker", "row": row}


def search_brokers(raw) -> ToolResult:
    a = _obj(raw)
    name = str(a.get("name") or "").strip() or None
    brn = str(a.get("brn") or "").strip() or None
    place = str(a.get("place") or "").strip() or None
    band = str(a.get("band") or "").strip().lower() or None
    if band not in (None, "", "top", "mid"):
        band = None
    limit = int(round(_num(a, "limit", 1, 10) or 8))
    from app.retrieval.propai_catalog import list_brokers, resolve_dubai_place
    if place:
        resolved = resolve_dubai_place(place)
        if resolved is None:
            asked = place
            return ToolResult(
                {
                    "matching_brokers": 0,
                    "showing": 0,
                    "out_of_coverage": True,
                    "place": asked,
                    "note": (
                        "Out of coverage. Tell the user you can only show licensed agents in Dubai. "
                        "Do not list any agents or invent names."
                    ),
                    "brokers": [],
                },
                [],
            )
        place = resolved
    rows = list_brokers(name=name, brn=brn, limit=limit, place=place, band=band)
    out = {
        "matching_brokers": len(rows),
        "showing": len(rows),
        "note": (
            "Licensed DLD agents ranked by card rank. When talking to the user say agent or agents, never broker. "
            "Quote only these fields. Email and phone are not in this data and must not be invented."
        ),
        "brokers": rows,
    }
    if not rows:
        out["note"] = (
            "No licensed agent matched. Tell the user, and offer the top-ranked agents "
            "by calling search_brokers with no name."
        )
    return ToolResult(out, [_broker_card(row) for row in rows])


def update_profile(raw, profile=None) -> ToolResult:
    a = _obj(raw)
    if profile is None:
        return ToolResult({"saved": False, "note": "Preferences cannot be stored in this session."})
    if a.get("clear") is True:
        profile.clear()
        return ToolResult({"saved": True, "profile": {}})
    profile.update(clean_profile({k: v for k, v in a.items() if k != "clear"}))
    return ToolResult({"saved": True, "profile": dict(profile)})


TOOL_FUNCTIONS = {
    "find_areas": find_areas,
    "search_listings": search_listings,
    "area_stats": area_stats,
    "get_listing": get_listing,
    "dataset_info": dataset_info,
    "search_projects": search_projects,
    "get_project": get_project,
    "compare_projects": compare_projects,
    "market_stats": market_stats,
    "search_brochures": search_brochures,
    "search_brokers": search_brokers,
    "area_sales": area_sales,
    "update_profile": update_profile,
}


def run_tool(name: str, raw_args: Any, pgvector=None, profile=None) -> ToolResult:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ToolInputError(f"Unknown tool: {name}")
    if name == "search_brochures":
        return function(raw_args, pgvector=pgvector)
    if name == "update_profile":
        return function(raw_args, profile=profile)
    return function(raw_args)
