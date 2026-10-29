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
           "community, sub_community, stored_photos")


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
        "beds_min": _num(a, "beds_min", 0, 10),
        "beds_max": _num(a, "beds_max", 0, 10),
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
        parts.append("(community ILIKE ANY(%s::text[]) OR sub_community ILIKE ANY(%s::text[]) "
                     "OR location ILIKE ANY(%s::text[]))")
        params += [_like(names)] * 3
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
        "size_sqft": size if usable else None,
        "furnishing": row.get("furnishing"),
        "community": row.get("community"),
        "sub_community": row.get("sub_community"),
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

def find_areas(raw) -> ToolResult:
    a = _obj(raw)
    text = str(a.get("text") or "").strip()[:80]
    if not text:
        raise ToolInputError("text is required.")
    patterns = _like([n for n in (text, ALIASES.get(text.lower())) if n])
    areas = fetch("select community, count(*)::int as listings from listings where community ILIKE ANY(%s::text[]) "
                  "group by community order by listings desc limit 10", (patterns,))
    subs = fetch("select sub_community, community, count(*)::int as listings from listings "
                 "where sub_community ILIKE ANY(%s::text[]) group by sub_community, community "
                 "order by listings desc limit 10", (patterns,))
    buildings = fetch("select regexp_replace(location, '^.*>\\s*', '') as building, community, count(*)::int as listings "
                      "from listings where location ILIKE ANY(%s::text[]) group by 1, community "
                      "order by listings desc limit 10", (patterns,))
    total = fetch("select count(*)::int as total from listings where community ILIKE ANY(%s::text[]) "
                  "or sub_community ILIKE ANY(%s::text[]) or location ILIKE ANY(%s::text[])", (patterns,) * 3)[0]["total"]
    if not (areas or subs or buildings):
        top = fetch("select community, count(*)::int as listings from listings group by community "
                    "order by listings desc limit 15")
        return ToolResult({"matches": [], "note": f'No area or building matching "{text}" in the data.',
                           "most_listed_areas": top})
    return ToolResult({
        "total_listings_matching": total,  # exact, across ALL matches; the lists show only the top 10
        "communities": areas, "sub_areas": subs, "buildings": buildings,
    })


def search_listings(raw) -> ToolResult:
    a = _obj(raw)
    f = parse_filters(a)
    sort = str(a.get("sort") or "cheapest")
    if sort not in SORTS:
        raise ToolInputError(f"sort must be one of: {', '.join(SORTS)}.")
    limit = int(round(_num(a, "limit", 1, 10) or 5))
    where, params = _where(f)
    rows = fetch(f"select {COLUMNS} from listings where {where} order by {SORTS[sort]} limit %s", (*params, limit))
    total = fetch(f"select count(*)::int as total from listings where {where}", tuple(params))[0]["total"]
    cards = [_card(r) for r in rows]
    output = {"matching_listings": total, "showing": len(cards), "sorted_by": sort,
              "listings": [_for_model(c) for c in cards]}
    if total == 0 and (f.get("property_types") or f.get("beds_min") is not None or f.get("beds_max") is not None):
        # nothing matched: tell the model what the same place does have, so it can offer a real alternative
        loose_where, loose_params = _where({**f, "property_types": None, "beds_min": None, "beds_max": None})
        nearby = fetch(f"select property_type, beds, count(*)::int as listings from listings where {loose_where} "
                       "group by property_type, beds order by listings desc limit 8", tuple(loose_params))
        if nearby:
            output["note"] = ("No listing matches the property type or bedrooms asked for. "
                              "In the same place and budget the data has these instead:")
            output["available_instead"] = nearby
    return ToolResult(output, cards)


def area_stats(raw) -> ToolResult:
    a = _obj(raw)
    f = parse_filters(a)
    group_key = str(a.get("group_by") or "none")
    if group_key != "none" and group_key not in GROUPS:
        raise ToolInputError(f"group_by must be one of: none, {', '.join(GROUPS)}.")
    group = None if group_key == "none" else GROUPS[group_key]
    where, params = _where(f)
    sql = (
        f"select {group + ' as grp,' if group else ''} count(*)::int as listings, "
        "round(avg(price_aed))::int as avg_annual, "
        "round(percentile_cont(0.5) within group (order by price_aed))::int as median_annual, "
        "round(percentile_cont(0.25) within group (order by price_aed))::int as p25_annual, "
        "round(percentile_cont(0.75) within group (order by price_aed))::int as p75_annual, "
        "min(price_aed)::int as min_annual, max(price_aed)::int as max_annual, "
        f"round((avg(price_aed / size_sqft) filter (where size_sqft between {MIN_SQFT} and {MAX_SQFT}))::numeric, 1) "
        f"as psf from listings where {where}"
        + (f" group by {group} order by {'grp' if group_key == 'beds' else 'count(*) desc'} limit 15" if group else "")
    )
    rows = fetch(sql, tuple(params))
    out = []
    for r in rows:
        row = {group_key: r["grp"]} if group else {}
        n = r["listings"]
        row.update({
            "listings": n, "reliability": _reliability(n),
            "avg_annual_rent_aed": r["avg_annual"], "avg_monthly_rent_aed": _monthly(r["avg_annual"]),
            "median_annual_rent_aed": r["median_annual"], "median_monthly_rent_aed": _monthly(r["median_annual"]),
            "typical_range_annual_aed": [r["p25_annual"], r["p75_annual"]],
            "typical_range_monthly_aed": [_monthly(r["p25_annual"]), _monthly(r["p75_annual"])],
            "min_annual_rent_aed": r["min_annual"], "max_annual_rent_aed": r["max_annual"],
            "avg_rent_per_sqft_per_year_aed": None if r["psf"] is None else float(r["psf"]),
        })
        out.append(row)
    return ToolResult({"grouped_by": group_key,
                       "note": "Statistics describe the listings in our dataset, which is a sample of the market.",
                       "rows": out})


def get_listing(raw) -> ToolResult:
    listing_id = str(_obj(raw).get("listing_id") or "").strip()
    if not re.fullmatch(r"\d{1,12}", listing_id):
        raise ToolInputError("listing_id must be the numeric listing id.")
    rows = fetch(f"select {COLUMNS}, listed_at, agency_name, amenities from listings where listing_id = %s", (listing_id,))
    if not rows:
        return ToolResult({"found": False, "note": f"No listing with id {listing_id} in the data."})
    row = rows[0]
    peers = fetch(
        "select count(*)::int as n, round(percentile_cont(0.5) within group (order by price_aed))::int as median_annual, "
        "round(100.0 * count(*) filter (where price_aed < %s) / nullif(count(*), 0))::int as pct_cheaper "
        "from listings where community = %s and beds is not distinct from %s",
        (row["price_aed"], row["community"], row["beds"]))[0]
    card = _card(row)
    return ToolResult({
        "found": True,
        "listing": {**_for_model(card), "listed_at": row.get("listed_at"), "agency": row.get("agency_name"),
                    "amenities": row.get("amenities") or None},
        "compared_with_same_area_and_bedrooms": {
            "listings": peers["n"], "reliability": _reliability(peers["n"]),
            "median_annual_rent_aed": peers["median_annual"], "median_monthly_rent_aed": _monthly(peers["median_annual"]),
            "percent_of_them_cheaper_than_this_one": peers["pct_cheaper"],
        },
    }, [card])


def dataset_info(raw=None) -> ToolResult:
    t = fetch("select count(*)::int as listings, count(distinct community)::int as areas, "
              "max(scraped_at) as last_scraped, "
              "round(percentile_cont(0.5) within group (order by price_aed))::int as median_annual from listings")[0]
    types = fetch("select property_type, count(*)::int as listings from listings group by 1 order by 2 desc")
    beds = fetch("select beds, count(*)::int as listings from listings group by 1 order by 1")
    return ToolResult({
        "total_listings": t["listings"], "areas": t["areas"], "snapshot_taken": str(t["last_scraped"])[:10],
        "median_annual_rent_aed": t["median_annual"], "median_monthly_rent_aed": _monthly(t["median_annual"]),
        "property_types": types,
        "bedrooms": [{"bedrooms": "studio" if b["beds"] == 0 else b["beds"], "listings": b["listings"]} for b in beds],
        "what_this_is": "Advertised (asking) yearly rents from Bayut for Dubai residential listings, one snapshot, "
                        "a sample of the market.",
        "not_included": "No sale prices, no property values or yields, no official RERA index, no Ejari contract rents.",
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
}
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
        "bedrooms": _num(a, "bedrooms", 0, 12),
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
        parts.append("(community ILIKE ANY(%s::text[]) OR sub_community ILIKE ANY(%s::text[]))")
        params += [_like(names)] * 2
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
        "property_types": row.get("property_types") or None,
        "status": row.get("project_status") or row.get("status"),
        "has_brochure": bool(row.get("brochure_url")),
    }


def _project_card(row: dict) -> dict:
    return {"listing_id": str(row["id"]), "kind": "project", "row": row}


def _available_places() -> dict:
    """Shown when a search finds nothing, so the model can offer real alternatives instead of a dead end."""
    return {
        "communities_with_projects": fetch("select community, count(*)::int as projects from projects "
                                            "where community is not null group by community order by projects desc"),
        "developers": fetch("select btrim(developer_name) as developer, count(*)::int as projects from projects "
                            "where developer_name is not null group by 1 order by projects desc"),
    }


_known_names: set[str] | None = None


def mentions_known_name(text: str) -> bool:
    """True when the text names a community, project, developer or common short form that is in our data."""
    global _known_names
    if _known_names is None:
        try:
            rows = fetch("select lower(btrim(community)) as n from projects union "
                         "select lower(btrim(community)) from listings union "
                         "select lower(btrim(developer_name)) from projects union "
                         "select lower(btrim(name)) from projects")
        except Exception:
            return False  # no database: do not force anything
        _known_names = {r["n"] for r in rows if r.get("n") and len(r["n"]) >= 4}
        _known_names |= {alias for alias in ALIASES if len(alias) >= 3}
    lowered = (text or "").lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered) for name in _known_names)


def search_projects(raw) -> ToolResult:
    a = _obj(raw)
    f = _project_filters(a)
    sort = str(a.get("sort") or "cheapest")
    if sort not in PROJECT_SORTS:
        raise ToolInputError(f"sort must be one of: {', '.join(PROJECT_SORTS)}.")
    limit = int(round(_num(a, "limit", 1, 10) or 8))
    where, params = _project_where(f)
    rows = fetch(f"select {PROJECT_COLUMNS} from projects where {where} order by {PROJECT_SORTS[sort]} limit %s",
                 (*params, limit))
    total = fetch(f"select count(*)::int as total from projects where {where}", tuple(params))[0]["total"]
    out = {
        "matching_projects": total, "showing": len(rows), "sorted_by": sort,
        "price_note": "Prices are STARTING prices for the project, not prices of a particular unit.",
        "projects": [_project_for_model(r) for r in rows],
    }
    cards = [_project_card(r) for r in rows]
    if total == 0:
        out["note"] = ("No project matches these filters. Tell the user, and offer to loosen a filter or pick "
                       "from what exists. Unless the user asked to buy, or for sale or off-plan, ALSO call "
                       "search_listings for the same place, bedrooms and size before you answer, and show the "
                       "rentals too. Only call a result 'over budget' if the user actually gave a budget.")
        closest = _closest_projects(f)
        if closest:
            out["closest_matches"] = [_project_for_model(r) | _gap(r, f) for r in closest]
            out["closest_matches_note"] = ("These are the nearest real projects after relaxing: "
                                           f"{closest[0]['_relaxed']}. Present them as the closest options, "
                                           "say plainly that they do not fully match, and quote the gap given."
                                           + ("" if f.get("max_price") else
                                              " The user gave no budget, so do not say anything is over budget "
                                              "or compute a gap."))
            cards += [_project_card({k: v for k, v in r.items() if k != "_relaxed"}) for r in closest]
        out.update(_available_places())
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
        where, params = _project_where(loosened)
        # with a budget cap, the nearest projects are the cheapest ones; with a floor, the cheapest above it
        order = "NULLIF(price, 0) ASC NULLS LAST, id" if f.get("max_price") or f.get("min_price") else \
            PROJECT_SORTS["cheapest"]
        rows = fetch(f"select {PROJECT_COLUMNS} from projects where {where} and price > 0 "
                     f"order by {order} limit %s", (*params, limit))
        if rows:
            return [{**r, "_relaxed": label} for r in rows]
    return []


def _gap(row: dict, f: dict) -> dict:
    """How far a near-miss is from the budget, computed here so the model never does the arithmetic."""
    price, cap = row.get("price"), f.get("max_price")
    if price and cap and price > cap:
        return {"above_budget_by_aed": int(price - cap)}
    return {}


def _find_projects(ref: str, limit: int = 6) -> list[dict]:
    ref = str(ref or "").strip()[:80]
    if not ref:
        return []
    if re.fullmatch(r"\d{1,12}", ref):
        return fetch(f"select {PROJECT_COLUMNS} from projects where id = %s", (int(ref),))
    exact = fetch(f"select {PROJECT_COLUMNS} from projects where lower(btrim(name)) = lower(%s) limit 2", (ref,))
    if len(exact) == 1:
        return exact
    where, params = _project_where({"name": ref})
    return fetch(f"select {PROJECT_COLUMNS} from projects where {where} order by length(name), id limit %s",
                 (*params, limit))


def get_project(raw) -> ToolResult:
    a = _obj(raw)
    ref = str(a.get("project_id") or a.get("name") or "").strip()
    if not ref:
        raise ToolInputError("Give project_id (numeric id) or name.")
    rows = _find_projects(ref)
    if not rows:
        return ToolResult({"found": False, "note": f'No project matching "{ref[:80]}" in the data.',
                           **_available_places()})
    if len(rows) > 1:
        return ToolResult({"found": False, "ambiguous": True,
                           "note": "Several projects match. Ask the user which one, or pick by exact name.",
                           "candidates": [_project_for_model(r) for r in rows]},
                          [_project_card(r) for r in rows])
    row = rows[0]
    has_price = bool(row.get("price") and row["price"] > 0)
    peers = fetch(
        "select count(*)::int as n, round(percentile_cont(0.5) within group (order by price))::bigint as median, "
        "round(100.0 * count(*) filter (where price < %s) / nullif(count(*), 0))::int as pct_cheaper "
        "from projects where community = %s and price > 0", (row["price"] if has_price else 0, row.get("community")))[0]
    description = re.sub(r"\s+", " ", str(row.get("description") or "")).strip()[:1200] or None
    return ToolResult({
        "found": True,
        "project": _project_for_model(row),
        "developer_marketing_description": description,
        "marketing_note": "The description is developer marketing text: say 'the developer describes it as', "
                          "never state its claims as verified fact.",
        "compared_with_other_projects_in_same_community": {
            "projects": peers["n"], "reliability": _reliability(peers["n"]),
            "median_starting_price_aed": None if peers["median"] is None else int(peers["median"]),
            "percent_of_them_cheaper_than_this_one": peers["pct_cheaper"] if has_price else None,
        },
    }, [_project_card(row)])


def compare_projects(raw) -> ToolResult:
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
    out: dict = {"compared": [_project_for_model(r) for r in found]}
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
            "note": "Starting prices only; not unit prices.",
        }
    dated = [r for r in found if r.get("handover_time") is not None]
    if len(dated) >= 2:
        first = min(dated, key=lambda r: r["handover_time"])
        out["earliest_handover"] = {"project": first["name"],
                                    "handover": _project_for_model(first)["handover"]}
    out["not_in_the_data"] = "Rental yield, ROI, resale value and unit-level prices are not in this data."
    return ToolResult(out, [_project_card(r) for r in found])


def market_stats(raw) -> ToolResult:
    a = _obj(raw)
    f = _project_filters(a)
    group_key = str(a.get("group_by") or "none")
    if group_key != "none" and group_key not in PROJECT_GROUPS:
        raise ToolInputError(f"group_by must be one of: none, {', '.join(PROJECT_GROUPS)}.")
    group = None if group_key == "none" else PROJECT_GROUPS[group_key]
    where, params = _project_where(f)
    sql = (
        f"select {group + ' as grp,' if group else ''} count(*)::int as projects, "
        "(min(price) filter (where price > 0))::bigint as min_price, "
        "round(percentile_cont(0.5) within group (order by price) filter (where price > 0))::bigint as median_price, "
        "(max(price) filter (where price > 0))::bigint as max_price, "
        "min(bedroom_min) as bed_min, max(bedroom_max) as bed_max, "
        "count(*) filter (where handover_time is not null)::int as with_handover "
        f"from projects where {where}"
        + (f" group by {group} order by count(*) desc limit 15" if group else "")
    )
    rows = fetch(sql, tuple(params))
    years = fetch("select extract(year from handover_time at time zone 'UTC')::int as year, count(*)::int as projects "
                  f"from projects where {where} and handover_time is not null group by 1 order by 1", tuple(params))
    out = []
    for r in rows:
        row = {group_key: r["grp"]} if group else {}
        row.update({"projects": r["projects"], "reliability": _reliability(r["projects"]),
                    "min_starting_price_aed": r["min_price"], "median_starting_price_aed": r["median_price"],
                    "max_starting_price_aed": r["max_price"],
                    "bedrooms_offered": [r["bed_min"], r["bed_max"]], "projects_with_handover_date": r["with_handover"]})
        out.append(row)
    return ToolResult({
        "grouped_by": group_key, "rows": out, "handover_years": years,
        "note": "Statistics describe the projects in our database (starting prices), which is a sample of the "
                "market. There is no rental yield, ROI, transaction or price-growth data.",
    })


def search_brochures(raw, pgvector=None) -> ToolResult:
    a = _obj(raw)
    query = str(a.get("query") or "").strip()[:300]
    if not query:
        raise ToolInputError("query is required.")
    project = str(a.get("project_name") or "").strip()[:80]
    if pgvector is None:
        return ToolResult({"passages": [], "note": "Brochure search is not available right now."})
    chunks = pgvector.search_documents(f"{project} {query}".strip(), match_threshold=0.25, match_count=10) or []
    tokens = _name_tokens(project)
    if tokens:
        about = [c for c in chunks if all(t in f"{c.get('document_title') or ''} {c.get('content') or ''}".lower()
                                          for t in tokens)]
        if not about:
            return ToolResult({"passages": [], "note": f'No brochure text found that mentions "{project}".'})
        chunks = about
    passages = [{"document": c.get("document_title"), "type": c.get("document_type"),
                 "similarity": None if c.get("similarity") is None else round(float(c["similarity"]), 2),
                 "text": re.sub(r"\s+", " ", str(c.get("content") or "")).strip()[:900]} for c in chunks[:4]]
    return ToolResult({
        "passages": passages,
        "note": "Brochure and document text. Brochure claims are developer marketing: say 'the brochure says', "
                "and only use what a passage actually states.",
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
    for key, lo, hi in (("bedrooms", 0, 12), ("family_size", 1, 30)):
        v = _num(a, key, lo, hi)
        if v is not None:
            out[key] = int(v)
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
