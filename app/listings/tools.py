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
    return ToolResult({"matching_listings": total, "showing": len(cards), "sorted_by": sort,
                       "listings": [_for_model(c) for c in cards]}, cards)


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


TOOL_FUNCTIONS = {
    "find_areas": find_areas,
    "search_listings": search_listings,
    "area_stats": area_stats,
    "get_listing": get_listing,
    "dataset_info": dataset_info,
}


def run_tool(name: str, raw_args: Any) -> ToolResult:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ToolInputError(f"Unknown tool: {name}")
    return function(raw_args)
