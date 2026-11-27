"""
Prop-AI is the sole runtime data source for ACOT.

Main tables / views (all under public):
  propai_projects, propai_places, propai_sales, propai_rents,
  propai_brokers, propai_project_images, propai_project_plans,
  propai_project_status, propai_raw, propai_area_stats, propai_project_prices

Old Bayut tables (listings, communities, sub_communities, document_chunks)
are not queried here.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import time
from datetime import datetime

from app.database.postgres import CLEAN_SALE, fetch
from app.listings.tools import ALIASES
from app.retrieval.place_lock import (
    asked_keys,
    contains_place_sql,
    drop_numbered_siblings,
    is_numbered_sibling,
    sales_place_sql,
)
from app.retrieval.propai_vector_search import search_places

MIN_SCORE = 0.55
# Browse / recommend sale facts: present market only (not multi-year-old history).
RECENT_SALE_MONTHS = 12
DEFAULT_PROJECT_SORT = "latest_handover"
_PROJECT_SORTS = {
    "cheapest",
    "priciest",
    "earliest_handover",
    "latest_handover",
    "most_recent_sale",
    "largest",
    "more",
}
_FILLER = re.compile(
    r"\b(?:show|me|find|list|properties|property|projects|project|apartments|apartment|"
    r"villas|villa|available|please|what|which|are|is|there|in|at|the|a|an|for|of|with|"
    r"under|over|bedroom|bedrooms|beds|bed|br|rent|rental|sale|buy|buying)\b",
    re.I,
)
_BEDROOMS = re.compile(r"\b(\d+)\s*[- ]?\s*(?:bed(?:room)?s?|br)\b", re.I)


def _focus(question: str) -> str:
    text = " ".join((question or "").split())
    lowered = text.lower()
    for key in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            return ALIASES[key]
    kept = _FILLER.sub(" ", text)
    kept = re.sub(r"[^a-z0-9\s]", " ", kept, flags=re.I)
    kept = " ".join(kept.split())
    if not kept:
        return kept
    label = known_dubai_label(kept)
    if label:
        return label
    return kept


def _photo_base() -> str:
    explicit = os.getenv("PROPAI_PHOTOS_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    photos = os.getenv(
        "PHOTOS_BASE_URL",
        "https://storage.googleapis.com/kinetic-axle-442714-m6-dubai-photos/photos",
    ).strip().rstrip("/")
    if photos.endswith("/photos"):
        return photos[: -len("/photos")]
    return photos


def _public_photo(object_name) -> str | None:
    name = str(object_name or "").strip().lstrip("/")
    if not name or ".." in name.split("/"):
        return None
    return f"{_photo_base()}/{name}"


def _photos(value) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.startswith("http"):
        items = [value]
    else:
        items = []
    kept = []
    for item in items:
        url = str(item)
        if url.startswith("http") and "X-Amz-" not in url:
            kept.append(url)
        if len(kept) >= 8:
            break
    return kept


def _attach_brochures(projects: list[dict]) -> None:
    """Do not copy brochure URLs from the old projects table."""
    return


def _attach_photos(projects: list[dict]) -> None:
    """Use the copied JPEGs. The links stored on the project row expire after one hour."""
    keys = [str(project["id"]) for project in projects if project.get("id")]
    if not keys:
        return
    try:
        rows = fetch(
            """
            select project_key, object_name
            from propai_project_images
            where project_key = any(%s)
            order by project_key, position
            """,
            (keys,),
        )
    except Exception:
        return
    grouped: dict[str, list[str]] = {}
    for row in rows:
        url = _public_photo(row.get("object_name"))
        if not url:
            continue
        photos = grouped.setdefault(str(row.get("project_key")), [])
        if len(photos) < 8:
            photos.append(url)
    for project in projects:
        stored = grouped.get(str(project.get("id")))
        if stored:
            project["photos"] = stored
    _fallback_photos(projects)


def _fallback_photos(projects: list[dict]) -> None:
    """A project with no photo of its own shows a same-named project's photo, else its area's cover photo
    (marked area_photo so the app can say so). Never an empty card when something real exists."""
    missing = [p for p in projects if not p.get("photos")]
    if not missing:
        return
    names = list({str(p["name"]).upper() for p in missing if p.get("name")})
    areas = list({str(p.get("sub_community") or "").upper() for p in missing if p.get("sub_community")})
    try:
        siblings = {r["n"]: _public_photo(r["object_name"]) for r in fetch(
            "select distinct on (upper(p.name)) upper(p.name) as n, i.object_name from propai_projects p "
            "join propai_project_images i on i.project_key = p.key where upper(p.name) = any(%s) "
            "order by upper(p.name), i.position", (names,))} if names else {}
        covers = {r["a"]: _public_photo(r["cover_object"]) for r in fetch(
            "select upper(area) as a, cover_object from propai_area_stats where upper(area) = any(%s) and cover_object is not null",
            (areas,))} if areas else {}
    except Exception:
        return
    for project in missing:
        same = siblings.get(str(project.get("name") or "").upper())
        if same:
            project["photos"] = [same]
            continue
        cover = covers.get(str(project.get("sub_community") or "").upper())
        if cover:
            project["photos"] = [cover]
            project["area_photo"] = True


def _add_rent_yield(projects: list[dict], bedrooms=None) -> None:
    """Gross rental yield per project: the median yearly rent of its recent contracts over its median recent sale price.

    Needs at least 5 recent contracts and a sale price; otherwise the project simply has no yield (never a guess).
    With a bedroom filter both sides use that bedroom count, so it compares like with like."""
    keys = [str(p["id"]) for p in projects if p.get("id") and p.get("price")]
    if not keys:
        return
    parts = [
        "location_key_project = any(%s)", "deleted_at is null", "annual_amount > 0", "coalesce(outlier_flag, '') <> '1'",
        "contract_start_date between current_date - interval '24 months' and current_date",
    ]
    params: list = [keys]
    try:
        if bedrooms is not None and bedrooms != "":
            parts.append("bedrooms = %s")
            params.append(int(float(bedrooms)))
        rows = fetch(
            "select location_key_project as k, count(*)::int as n, "
            "percentile_cont(0.5) within group (order by annual_amount)::float8 as rent "
            f"from propai_rents where {' and '.join(parts)} group by 1",
            tuple(params),
        )
    except Exception:
        return
    by_key = {r["k"]: r for r in rows}
    for project in projects:
        stat = by_key.get(str(project.get("id")))
        if not stat or stat["n"] < 5 or not project.get("price"):
            continue
        gross = stat["rent"] / float(project["price"]) * 100
        if 1 <= gross <= 20:  # outside this the rent or the price is a data slip, not a yield
            project["gross_yield_pct"] = round(gross, 1)
            project["median_rent_aed"] = int(round(stat["rent"]))
            project["rent_contracts"] = stat["n"]


def top_yield_projects(max_price=None, min_price=None, place=None, bedrooms=None, limit: int = 8) -> tuple[list[dict], int]:
    """Residential projects ranked by gross rental yield: median yearly rent of contracts from the last 24 months over the
    median sale price of the last 12 months. Needs 10+ sales and 20+ contracts, and the yield must be between 2% and 12%
    (outside that the rents and prices describe different kinds of unit, not a real return). With a bedroom count, rents and sales are both for that count."""
    cache_key = (max_price, min_price, str(place or "").lower(), str(bedrooms), int(limit))
    cached = _YIELD_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < 1800:
        return [dict(p) for p in cached[1]], cached[2]
    bed_sql, bed_params = "", []
    if bedrooms is not None and bedrooms != "":
        bed_sql, bed_params = " and bedrooms = %s", [int(float(bedrooms))]
    where = ["not p.is_plot", "coalesce(p.development_status, '') <> 'CANCELLED'", "lower(coalesce(p.usage, '')) = 'residential'",
             "r.rent / s.price * 100 between 2 and 12"]
    wparams: list = []
    if min_price is not None:
        where.append("s.price >= %s")
        wparams.append(float(min_price))
    if max_price is not None:
        where.append("s.price <= %s")
        wparams.append(float(max_price))
    if place:
        clause, extra = contains_place_sql(["p.master_project", "p.area", "p.name"], _expanded([place]))
        where.append(clause)
        wparams.extend(extra)
    rows = fetch(
        f"""
        with s as (
          select location_key_project as k, count(*)::int as n, percentile_cont(0.5) within group (order by sale_price)::float8 as price,
                 max(instance_date) as last_sale
          from propai_sales
          where location_key_project is not null and deleted_at is null and sale_price between 100000 and 2000000000
            and coalesce(outlier_flag, '') <> 'True' and (property_type = 'Land' or price_per_sqft is null or price_per_sqft >= 150)
            and instance_date between current_date - interval '12 months' and current_date{bed_sql}
          group by 1 having count(*) >= 10),
        r as (
          select location_key_project as k, count(*)::int as n, percentile_cont(0.5) within group (order by annual_amount)::float8 as rent
          from propai_rents
          where location_key_project is not null and deleted_at is null and annual_amount > 0 and coalesce(outlier_flag, '') <> '1'
            and contract_start_date between current_date - interval '24 months' and current_date{bed_sql}
          group by 1 having count(*) >= 20)
        select p.key, p.name, p.developer, p.area, p.master_project, p.offplan_status, p.end_date,
               s.price, s.n as sales, s.last_sale, r.rent, r.n as contracts, r.rent / s.price * 100 as gross,
               count(*) over () as total
        from s join r on r.k = s.k join propai_projects p on p.location_key = s.k
        where {' and '.join(where)}
        order by gross desc, r.n desc
        limit %s
        """,
        tuple([*bed_params, *bed_params, *wparams, int(limit) * 3]),
    )
    projects, seen = [], set()
    for row in rows:
        name = str(row["name"] or "").upper()
        if not name or name in seen:
            continue
        seen.add(name)
        projects.append({
            "id": row["key"], "name": row["name"], "developer_name": row.get("developer"),
            "community": row.get("master_project") or row.get("area"), "sub_community": row.get("area"), "city": "Dubai",
            "price": int(round(row["price"])), "price_kind": "recent_sale",
            "bedroom_min": int(float(bedrooms)) if bedrooms not in (None, "") else None,
            "bedroom_max": int(float(bedrooms)) if bedrooms not in (None, "") else None,
            "status": row.get("offplan_status"), "project_status": row.get("offplan_status"), "handover_time": _handover(row.get("end_date")),
            "photos": [], "source": "propai", "size_min": None, "size_max": None, "property_types": None, "brochure_url": None,
            "sales_count": row["sales"], "last_sale": row.get("last_sale"), "last_sold_on": str(row.get("last_sale") or "")[:10] or None,
            "gross_yield_pct": round(row["gross"], 1), "median_rent_aed": int(round(row["rent"])), "rent_contracts": row["contracts"],
        })
        if len(projects) >= limit:
            break
    total = int(rows[0]["total"]) if rows else 0
    _attach_photos(projects)
    for project in projects:
        project["key"] = project["id"]
        project["id"] = _agent_id(str(project["id"]))
    _YIELD_CACHE[cache_key] = (time.time(), [dict(p) for p in projects], total)
    return projects, total


_YIELD_CACHE: dict = {}  # the ranking scans a year of sales and two years of rents (about 4 s), so answers are kept 30 minutes


def _handover(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    if parsed.year < 1970 or parsed.year > 2100:
        return None
    return parsed


def _projects_for(place: dict, limit: int = 8) -> list[dict]:
    # A master place must not use its parent area, or Marina rows appear under JBR.
    level = str(place.get("level") or "").lower()
    if level == "master":
        names = [place.get("label"), place.get("master")]
    elif level == "area":
        names = [place.get("label"), place.get("area")]
    else:
        names = [place.get("label"), place.get("master"), place.get("area")]
    names = [name for name in names if name]
    if not names:
        return []
    keys = list(dict.fromkeys(n.upper() for n in names))
    norms = list(dict.fromkeys(re.sub(r"[^A-Z0-9]+", "", key) for key in keys if re.sub(r"[^A-Z0-9]+", "", key)))
    if level == "area":
        clause = (
            "upper(payload->>'area_name_prop-ai') = any(%s) "
            "or upper(payload->>'area_name_label') = any(%s)"
        )
        clause_args = (keys, keys)
    else:
        # JBR is stored as both "JUMEIRAH BEACH RESIDENCE - JBR" and "Jumeirah Beach Residence (JBR)".
        clause = (
            "upper(payload->>'master_project_en') = any(%s) "
            "or upper(payload->>'master_project_prop-ai') = any(%s) "
            "or upper(payload->>'development_name_prop-ai') = any(%s) "
            "or regexp_replace(upper(coalesce(payload->>'master_project_en', '')), '[^A-Z0-9]+', '', 'g') = any(%s) "
            "or regexp_replace(upper(coalesce(payload->>'master_project_prop-ai', '')), '[^A-Z0-9]+', '', 'g') = any(%s)"
        )
        clause_args = (keys, keys, keys, norms, norms)
    rows = fetch(
        f"""
        select row_key, name, developer_name, master, area_label, status, handover, description, image_urls
        from (
            select distinct on (upper(payload->>'project_name_en'))
                   row_key,
                   payload->>'project_name_en' as name,
                   payload->>'developer_name' as developer_name,
                   payload->>'master_project_en' as master,
                   payload->>'area_name_label' as area_label,
                   payload->>'offplan_status' as status,
                   payload->>'project_end_date' as handover,
                   left(payload->>'project_description_en', 400) as description,
                   payload->'image_urls' as image_urls
            from propai_raw
            where dataset = 'projects'
              and coalesce(payload->>'project_name_en', '') <> ''
              and ({clause})
            order by upper(payload->>'project_name_en'),
                     (payload->>'developer_name') is not null desc,
                     payload->>'project_name_en'
        ) unique_projects
        order by handover desc nulls last, name
        limit %s
        """,
        (*clause_args, limit),
    )
    projects = []
    for row in rows:
        projects.append({
            "id": row.get("row_key"),
            "name": row.get("name"),
            "developer_name": row.get("developer_name"),
            "community": place.get("label") or row.get("area_label"),
            "sub_community": row.get("master"),
            "city": "Dubai",
            "price": None,
            "price_kind": None,
            "bedroom_min": None,
            "bedroom_max": None,
            "status": row.get("status"),
            "project_status": row.get("status"),
            "handover_time": _handover(row.get("handover")),
            "description": row.get("description"),
            "photos": _photos(row.get("image_urls")),
            "source": "propai",
        })
    _add_sale_facts(projects)
    _attach_photos(projects)
    _attach_brochures(projects)
    return projects


_SALE_FACTS: dict[str, dict] = {}
# Bi-daily DB loads: do not keep sale medians/last_sale forever in-process.
_SALE_FACTS_LOADED_AT: float | None = None
_SALE_FACTS_TTL_SEC = 6 * 3600


def _maybe_expire_sale_facts() -> None:
    global _SALE_FACTS_LOADED_AT
    now = time.time()
    if (
        _SALE_FACTS_LOADED_AT is not None
        and (now - _SALE_FACTS_LOADED_AT) >= _SALE_FACTS_TTL_SEC
    ):
        _SALE_FACTS.clear()
        _SALE_FACTS_LOADED_AT = None


def _add_sale_facts(projects: list[dict], bedrooms=None) -> None:
    global _SALE_FACTS_LOADED_AT
    _maybe_expire_sale_facts()
    names = [str(project["name"]).upper() for project in projects if project.get("name")]
    if not names:
        return
    bed_key = None
    try:
        bed_key = int(float(bedrooms)) if bedrooms is not None and bedrooms != "" else None
    except (OverflowError, TypeError, ValueError):
        bed_key = None
    cache_prefix = (
        f"M{int(RECENT_SALE_MONTHS)}|"
        + (f"B{bed_key}|" if bed_key is not None else "")
    )
    missing = [
        name for name in dict.fromkeys(names)
        if f"{cache_prefix}{name}" not in _SALE_FACTS
    ]
    if missing:
        parts = [
            CLEAN_SALE,
            f"instance_date >= current_date - interval '{int(RECENT_SALE_MONTHS)} months'",
            "upper(project_name_prop_ai) = any(%s)",
        ]
        params: list = [missing]
        if bed_key is not None:
            parts.append("bedrooms = %s")
            params.append(bed_key)
        try:
            stats = fetch(
                f"""
                select upper(project_name_prop_ai) as name,
                       count(*)::int as sales,
                       percentile_cont(0.5) within group (order by sale_price) as price,
                       min(bedrooms) filter (where bedrooms between 0 and 10) as bedroom_min,
                       max(bedrooms) filter (where bedrooms between 0 and 10) as bedroom_max,
                       max(instance_date) as last_sale
                from propai_sales
                where {' and '.join(parts)}
                group by 1
                """,
                tuple(params),
            )
        except Exception:
            stats = []
        for row in stats:
            _SALE_FACTS[f"{cache_prefix}{row['name']}"] = row
        for name in missing:
            _SALE_FACTS.setdefault(f"{cache_prefix}{name}", {})
        if _SALE_FACTS_LOADED_AT is None:
            _SALE_FACTS_LOADED_AT = time.time()
    for project in projects:
        stat = _SALE_FACTS.get(f"{cache_prefix}{str(project['name']).upper()}")
        if not stat:
            # No invented recent_sale label when SQL returned no sale facts.
            if project.get("price") is None:
                project["price_kind"] = None
            continue
        if stat.get("price"):
            project["price"] = int(round(float(stat["price"])))
        project["bedroom_min"] = stat.get("bedroom_min")
        project["bedroom_max"] = stat.get("bedroom_max")
        if stat.get("sales") is not None:
            project["sales_count"] = int(stat["sales"] or 0)
        if stat.get("last_sale"):
            project["last_sale"] = stat["last_sale"]
            # Frontend cards read last_sold_on; keep both keys in sync with the text.
            project["last_sold_on"] = (
                stat["last_sale"].isoformat()[:10]
                if hasattr(stat["last_sale"], "isoformat")
                else str(stat["last_sale"])[:10]
            )
        # price_kind tracks grounded sale price only — never claim sale without price.
        if project.get("price") is not None:
            project["price_kind"] = "recent_sale"
        else:
            project["price_kind"] = None


def lookup(question: str) -> dict | None:
    """Projects from the new data for this question, or None so the old tables answer."""
    phrase = _focus(question)
    if len(phrase) < 3:
        return None
    hits = search_places(phrase, limit=3)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return None
    place = hits[0]
    level = str(place.get("level") or "").lower()
    # Use the same project table as the agent. Keep master searches off the parent area
    # so JBR does not pull every Marsa Dubai / Marina row.
    communities = [phrase, place.get("label")]
    if level == "master":
        communities.append(place.get("master"))
    elif level == "area":
        communities.append(place.get("area"))
    communities = [name for name in communities if name]
    projects, _ = agent_projects({"communities": communities}, limit=8)
    if not projects:
        return None
    bedrooms = _BEDROOMS.search(question or "")
    if bedrooms:
        wanted = int(bedrooms.group(1))
        narrowed = [
            project for project in projects
            if project.get("bedroom_min") is not None
            and project["bedroom_min"] <= wanted <= (project.get("bedroom_max") or wanted)
        ]
        if narrowed:
            projects = narrowed
    return {
        "place": place.get("label"),
        "projects": projects,
        "question_type": "search",
    }


def _agent_id(key: str) -> str:
    number = int(hashlib.sha1(key.encode()).hexdigest()[:15], 16) % 1_000_000_000_000
    return f"P{number:012d}"


_AGENT_IDS: dict[str, str] | None = None


def _id_index() -> dict[str, str]:
    global _AGENT_IDS
    if _AGENT_IDS is None:
        rows = fetch("select key from propai_projects where coalesce(key, '') <> ''")
        _AGENT_IDS = {_agent_id(row["key"]): row["key"] for row in rows}
    return _AGENT_IDS


def _expanded(values) -> list[str]:
    names = []
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        names.append(text)
        alias = ALIASES.get(text.lower())
        if alias and alias not in names:
            names.append(alias)
    return names


def _likes(values: list[str]) -> list[str]:
    return ["%" + re.sub(r"([%_\\])", r"\\\1", value) + "%" for value in values]


def _official_areas(values) -> list[str]:
    names = _expanded(values)
    if not names:
        return []
    clause, extra = contains_place_sql(["label", "area", "master"], names)
    try:
        rows = fetch(
            f"""
            select distinct upper(area) as area, label, master
            from propai_places
            where level in ('area', 'master')
              and area is not null
              and {clause}
            """,
            tuple(extra),
        )
    except Exception:
        return asked_keys(names)
    found = []
    for row in rows or []:
        area = row.get("area")
        if not area:
            continue
        if any(
            is_numbered_sibling(name, str(row.get("label") or ""))
            or is_numbered_sibling(name, str(row.get("master") or ""))
            for name in names
        ):
            continue
        if area not in found:
            found.append(area)
    return found or asked_keys(names)


_PLACE_LABELS: list[str] | None = None


def place_named(text: str) -> str | None:
    """The community or area a question names ("in Dubai Marina", "JVC"), longest match first; None if none."""
    global _PLACE_LABELS
    if _PLACE_LABELS is None:
        try:
            rows = fetch("select distinct lower(btrim(label)) as n from propai_places where level in ('area', 'master')")
        except Exception:
            return None  # no database: do not guess
        labels = {row["n"] for row in rows if row.get("n") and len(row["n"]) >= 4}
        _PLACE_LABELS = sorted(labels | set(ALIASES), key=len, reverse=True)
    lowered = (text or "").lower()
    for label in _PLACE_LABELS:
        if re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", lowered):
            # "Arabian Ranches 3" is the numbered master (stored as ... III), not the original.
            numbered = re.search(
                rf"(?<![a-z0-9]){re.escape(label)}[\s-]*([23])\b"
                r"(?!\s*-?\s*(?:bed|br|bhk|million|m\b|k\b|,\d))",
                lowered,
            )
            if numbered:
                sibling = f"{label} {'ii' if numbered.group(1) == '2' else 'iii'}"
                if sibling in _PLACE_LABELS:
                    return sibling.title().replace(" Iii", " III").replace(" Ii", " II")
            return ALIASES.get(label) or label.title()
    return None


def _stamp_value(value) -> float:
    parsed = value if hasattr(value, "timestamp") else _handover(value)
    if parsed is None:
        return -1.0
    try:
        return float(parsed.timestamp())
    except (OSError, OverflowError, ValueError, TypeError):
        return -1.0


def _project_sort_key(project: dict, sort: str):
    price = project.get("price") or 0
    handover = project.get("handover_time")
    last_sale = project.get("last_sale") or project.get("last_sold_on")
    if sort == "priciest":
        return (price <= 0, -(price or 0), -_stamp_value(last_sale))
    if sort == "earliest_handover":
        return (handover is None, handover or datetime(2099, 12, 31))
    if sort == "latest_handover":
        return (handover is None, -_stamp_value(handover))
    if sort == "most_recent_sale":
        # Present market activity first — never rank on decade-old handover alone.
        return (
            last_sale is None,
            -_stamp_value(last_sale),
            price <= 0,
            price or 10**18,
        )
    if sort == "largest":
        size = project.get("size_max") or project.get("size_min") or 0
        return (size <= 0, -(size or 0))
    # cheapest (default price sort): budget fit, then freshest sale.
    return (price <= 0, price or 10**18, -_stamp_value(last_sale))


def _recent_best_key(project: dict) -> tuple:
    """Newest handover first, then latest recorded sale, then activity."""
    sales = 0.0
    try:
        sales = float(project.get("sales_count") or 0)
    except (TypeError, ValueError):
        sales = 0.0
    has_price = 1.0 if (project.get("price") or 0) > 0 else 0.0
    return (
        _stamp_value(project.get("handover_time") or project.get("handover")),
        _stamp_value(project.get("last_sale") or project.get("last_sold_on")),
        sales,
        has_price,
    )


def _more_mix(projects: list[dict], limit: int, rng=None) -> list[dict]:
    """Half the strongest remaining projects, half a random sample of the rest."""
    if len(projects) <= limit:
        return list(projects)
    rng = rng or random
    ranked = sorted(projects, key=_recent_best_key, reverse=True)
    best_n = max(1, (limit + 1) // 2)
    best = ranked[:best_n]
    rest = ranked[best_n:]
    rng.shuffle(rest)
    randoms = rest[: max(0, limit - len(best))]
    mixed = []
    extra = list(randoms)
    for item in best:
        mixed.append(item)
        if extra:
            mixed.append(extra.pop(0))
    mixed.extend(extra)
    return mixed[:limit]


def _bedroom_fit_key(project: dict, wanted: int) -> tuple:
    """Prefer exact / narrow bedroom ranges that include the requested count."""
    lo = project.get("bedroom_min")
    hi = project.get("bedroom_max")
    if lo is None:
        return (9, 99.0, 99.0)
    try:
        lo_f = float(lo)
        hi_f = float(hi) if hi is not None else lo_f
    except (TypeError, ValueError):
        return (9, 99.0, 99.0)
    exact = 0 if lo_f == hi_f == float(wanted) else 1
    span = hi_f - lo_f
    center = abs((lo_f + hi_f) / 2.0 - float(wanted))
    return (exact, span, center)


def _order_projects(
    projects: list[dict],
    sort: str,
    limit: int,
    skip_names=None,
    rng=None,
    prefer_bedrooms=None,
) -> list[dict]:
    skip = {
        str(name).strip().lower()
        for name in (skip_names or [])
        if str(name or "").strip()
    }
    remaining = [
        project for project in projects
        if str(project.get("name") or "").strip().lower() not in skip
    ]
    if prefer_bedrooms is not None:
        wanted = int(prefer_bedrooms)
        if sort == "more":
            remaining.sort(key=lambda project: _bedroom_fit_key(project, wanted))
            return _more_mix(remaining, limit, rng=rng)
        remaining.sort(
            key=lambda project: (
                _bedroom_fit_key(project, wanted),
                _project_sort_key(project, sort),
            )
        )
        return remaining[:limit]
    if sort == "more":
        return _more_mix(remaining, limit, rng=rng)
    remaining.sort(key=lambda project: _project_sort_key(project, sort))
    return remaining[:limit]


_APARTMENT_TYPES = {
    "apartment", "apartments", "flat", "flats", "unit", "units",
}


def _propai_rejects_filters(filters: dict) -> bool:
    """True when a filter has no column on Prop-AI project rows.

    Apartment is the default residential stock here, so that type is not a
    reject. Villa / townhouse / size still fall through to the older table.
    """
    if filters.get("min_size") or filters.get("max_size"):
        return True
    types = [
        str(item).strip().lower()
        for item in (filters.get("property_types") or [])
        if str(item).strip()
    ]
    if not types:
        return False
    return any(
        item not in _APARTMENT_TYPES and item.rstrip("s") not in {"apartment", "flat", "unit"}
        for item in types
    )


def browse_projects_near_budget(
    filters: dict,
    limit: int = 8,
    sort: str = DEFAULT_PROJECT_SORT,
    skip_names=None,
) -> tuple[list[dict], int, dict | None]:
    """Run agent_projects; if a max budget yields nothing, return cheapest above it.

    Third value is None on a normal hit, or a small meta dict on budget miss:
    {budget_miss, asked_max_price, lowest_price}.
    """
    rows, total = agent_projects(
        filters, limit=limit, sort=sort, skip_names=skip_names
    )
    asked_max = filters.get("max_price")
    if rows or asked_max is None:
        return rows, total, None
    loose = {
        key: value
        for key, value in (filters or {}).items()
        if key not in {"max_price", "min_price"}
    }
    near, near_total = agent_projects(
        loose, limit=limit, sort="cheapest", skip_names=skip_names
    )
    lowest = None
    if near and near[0].get("price") is not None:
        try:
            lowest = int(near[0]["price"])
        except (TypeError, ValueError):
            lowest = None
    return near, near_total, {
        "budget_miss": True,
        "asked_max_price": int(asked_max) if asked_max is not None else None,
        "lowest_price": lowest,
    }


def agent_projects(
    filters: dict,
    limit: int = 8,
    sort: str = DEFAULT_PROJECT_SORT,
    skip_names=None,
) -> tuple[list[dict], int]:
    """New-data projects in the old project-row shape. Empty when a filter the new rows cannot answer is set."""
    if _propai_rejects_filters(filters):
        return [], 0
    parts, params = ["coalesce(name, '') <> ''"], []
    if filters.get("key"):
        parts.append("key = %s")
        params.append(filters["key"])
    communities = _expanded(filters.get("communities"))
    if communities:
        clause, extra = contains_place_sql(["master_project", "area", "name"], communities)
        parts.append(clause)
        params.extend(extra)
    developers = [str(name).strip() for name in (filters.get("developers") or []) if str(name).strip()]
    if developers:
        parts.append("developer ilike any(%s::text[])")
        params.append(_likes(developers))
    name = str(filters.get("name") or "").strip()
    tokens = [token for token in re.findall(r"[a-z0-9]+", name.lower()) if token not in {"by", "at", "the", "of", "in", "and", "a"}]
    for token in tokens:
        parts.append("name ilike %s")
        params.append(_likes([token])[0])
    if filters.get("handover_year") is not None:
        parts.append("extract(year from end_date) = %s")
        params.append(int(filters["handover_year"]))
    if filters.get("ready_only"):
        # Ready = already completed / handed over — not only rows whose status
        # text happens to contain "ready" (most JVC towers use other labels).
        # Escape % for psycopg (%% → literal %).
        parts.append(
            "("
            "("
            "offplan_status ilike '%%ready%%' "
            "and offplan_status not ilike '%%off-plan%%'"
            ") or ("
            "status_group = 'Completed' or development_status = 'FINISHED'"
            " or development_status ilike '%%complet%%'"
            " or development_status ilike '%%exist%%'"
            ") or ("
            "end_date is not null and end_date::date <= current_date"
            " and coalesce(offplan_status, '') not ilike '%%off-plan%%'"
            ")"
            ")"
        )
    where = " and ".join(parts)
    if sort not in _PROJECT_SORTS:
        sort = DEFAULT_PROJECT_SORT
    # Ready / budget browses must surface present sales activity, not old end_dates.
    if (
        sort == DEFAULT_PROJECT_SORT
        and (filters.get("ready_only") or filters.get("max_price") is not None)
    ):
        sort = "most_recent_sale"
    if name:
        ranked = "lower(name) = lower(%s) desc, length(name), name"
    elif sort in {"latest_handover", "more"}:
        ranked = "end_date desc nulls last, name"
    elif sort == "earliest_handover":
        ranked = "end_date asc nulls last, name"
    else:
        # most_recent_sale / cheapest / etc. re-ranked after sale facts attach.
        ranked = "name"
    # Cap candidates before sale stats; ranking still uses the same pool size as before.
    fetch_limit = 200
    query_params = [*params, name] if name else list(params)
    # A price filter (min / max) is decided in SQL from each project's recent median sale price, so it looks at every
    # project. Without this the 200 candidates were the first 200 by name and everything after "Ag..." was never seen.
    band_join, band_params, band_select = "", [], ""
    lo, hi = filters.get("min_price"), filters.get("max_price")
    if lo is not None or hi is not None:
        median = "percentile_cont(0.5) within group (order by sale_price)"
        band_where = [
            CLEAN_SALE, "coalesce(project_name_prop_ai, '') <> ''",
            f"instance_date >= current_date - interval '{int(RECENT_SALE_MONTHS)} months'",
        ]
        having = []
        if filters.get("bedrooms") is not None:
            band_where.append("bedrooms = %s")
            band_params.append(int(float(filters["bedrooms"])))
        if lo is not None:
            having.append(f"{median} >= %s")
            band_params.append(lo)
        if hi is not None:
            having.append(f"{median} <= %s")
            band_params.append(hi)
        band_join = (
            f"join (select upper(project_name_prop_ai) as band_name, {median} as band_price from propai_sales "
            f"where {' and '.join(band_where)} group by 1 having {' and '.join(having)}) band on band.band_name = upper(name)"
        )
        band_select = ", band_price"
        if sort == "priciest":
            ranked = "band_price desc nulls last, name"
        elif sort == "cheapest":
            ranked = "band_price asc nulls last, name"
    rows = fetch(
        f"""
        select key, name, developer, area, master_project, offplan_status, end_date, description
        from (
            select distinct on (upper(name))
                   key, name, developer, area, master_project, offplan_status, end_date,
                   left(description, 400) as description{band_select}
            from propai_projects
            {band_join}
            where {where}
            order by upper(name),
                     (exists (select 1 from propai_project_images i where i.project_key = propai_projects.key)) desc,
                     (developer is not null) desc, name
        ) unique_projects
        order by {ranked}
        limit %s
        """,
        tuple([*band_params, *query_params, fetch_limit]),
    )
    if not rows:
        return [], 0
    # Skip a second count scan when the page is already the full set.
    if len(rows) < fetch_limit:
        total = len(rows)
    elif band_join:
        count = fetch(
            f"select count(distinct upper(name))::int as n from propai_projects {band_join} where {where}",
            tuple([*band_params, *params]),
        )
        total = count[0]["n"] if count else len(rows)
    else:
        count = fetch(
            f"select count(distinct upper(name))::int as n from propai_projects where {where}",
            tuple(params),
        )
        total = count[0]["n"] if count else len(rows)
    projects = []
    for row in rows:
        projects.append({
            "id": row["key"],
            "name": row.get("name"),
            "developer_name": row.get("developer"),
            "community": row.get("master_project") or row.get("area"),
            "sub_community": row.get("area"),
            "city": "Dubai",
            "price": None,
            "price_kind": None,
            "bedroom_min": None,
            "bedroom_max": None,
            "status": row.get("offplan_status"),
            "project_status": row.get("offplan_status"),
            "handover_time": _handover(row.get("end_date")),
            "description": row.get("description"),
            "photos": [],
            "source": "propai",
            "size_min": None,
            "size_max": None,
            "property_types": None,
            "brochure_url": None,
        })
    bedrooms = filters.get("bedrooms")
    _add_sale_facts(projects, bedrooms=bedrooms)
    _add_rent_yield(projects, bedrooms=bedrooms)
    if communities:
        projects = drop_numbered_siblings(projects, communities)
        # Soft reject only: drop rows that clearly name a *different* marketing
        # place (e.g. Dubai South on a JVC ask). Do not require the marketing
        # name inside DLD area codes — that emptied real JVC/Marina lists.
        allowed = set(asked_keys(_expanded(communities)))
        allowed.update(asked_keys(_official_areas(communities)))
        rivals = {
            asked_keys([proper])[0]
            for proper in _static_dubai_names().values()
            if proper and asked_keys([proper])
        }
        rivals -= allowed
        # Ignore short tokens (JLT, DIFC) — too many false conflicts.
        rivals = {key for key in rivals if len(key) >= 10}
        if rivals:
            kept = []
            for project in projects:
                blob_n = " ".join(
                    str(project.get(field) or "")
                    for field in ("community", "sub_community", "name")
                ).upper()
                blob_n = " ".join(blob_n.split())
                if any(key in blob_n for key in allowed if len(key) >= 5):
                    kept.append(project)
                    continue
                if any(rival in blob_n for rival in rivals):
                    continue
                kept.append(project)
            projects = kept
        total = len(projects)
    if bedrooms is not None:
        wanted = int(bedrooms)
        projects = [
            project for project in projects
            if project.get("bedroom_min") is not None
            and float(project["bedroom_min"]) <= wanted <= float(project.get("bedroom_max") or wanted)
        ]
        total = len(projects)
    if filters.get("min_price") is not None or filters.get("max_price") is not None:
        lo = filters.get("min_price")
        hi = filters.get("max_price")
        projects = [
            project for project in projects
            if project.get("price")
            and (lo is None or project["price"] >= lo)
            and (hi is None or project["price"] <= hi)
            # Budget answers only from recent recorded sales — never stale catalog.
            and (project.get("last_sold_on") or project.get("last_sale"))
        ]
        total = len(projects) if len(rows) < fetch_limit or not band_join else max(total, len(projects))
    elif filters.get("ready_only"):
        # Ready list without budget: still prefer projects with present-window sales.
        with_sales = [
            project for project in projects
            if project.get("last_sold_on") or project.get("last_sale")
        ]
        if with_sales:
            projects = with_sales
            total = len(projects)
    for project in projects:
        for key in ("bedroom_min", "bedroom_max"):
            if project.get(key) is not None:
                project[key] = int(float(project[key]))

    projects = _order_projects(
        projects,
        sort,
        limit,
        skip_names=skip_names,
        prefer_bedrooms=filters.get("bedrooms"),
    )
    _attach_photos(projects)
    _attach_brochures(projects)
    for project in projects:
        # The app's project page is keyed by the catalog key; the agent id is a one-way hash of it.
        project["key"] = project["id"]
        project["id"] = _agent_id(str(project["id"]))
    return projects, total


def recent_sales(name, limit: int = 8, per_project: int = 5, offset: int = 0) -> dict:
    """Recent recorded sales for the named projects.

    Newest sale first overall (bi-daily loads stay on top). Each project still
    contributes its own newest page so one active project cannot crowd others out.
    """
    raw = name if isinstance(name, (list, tuple)) else [name]
    projects = []
    for item in raw:
        text = " ".join(str(item or "").split())
        if len(text) < 3:
            continue
        for key in (text,):
            upper = key.upper()
            if upper not in projects:
                projects.append(upper)
        # Catalog titles like "The Woods at Sobha Sanctuary" are stored as
        # short sale names ("THE WOODS") under the same master.
        row = _exact_project_row(text)
        if row:
            short = _short_project_sale_alias(row.get("name"))
            if short and short.upper() not in projects:
                projects.append(short.upper())
        if len(projects) >= max(limit * 2, 8):
            break
    if not projects:
        return {"count": 0, "sales": []}
    per = max(1, min(int(per_project or 5), 8))
    start = max(0, int(offset or 0))
    rows = fetch(
        """
        select project, sold_on, price, bedrooms, size_sqft, property_type
        from (
            select project_name_prop_ai as project,
                   instance_date as sold_on,
                   sale_price::bigint as price,
                   bedrooms,
                   round(procedure_area_sqft)::int as size_sqft,
                   property_type_en as property_type,
                   row_number() over (
                       partition by upper(project_name_prop_ai)
                       order by instance_date desc nulls last
                   ) as rn
            from propai_sales
            where deleted_at is null
              and sale_price > 0
              and upper(project_name_prop_ai) = any(%s)
        ) ranked
        where rn > %s and rn <= %s
        order by sold_on desc nulls last
        """,
        (projects, start, start + per),
    )
    sales = []
    for row in rows or []:
        sold = row.get("sold_on")
        sales.append({
            "project": row.get("project"),
            "date": sold.isoformat() if hasattr(sold, "isoformat") else (str(sold)[:10] if sold else None),
            "price": None if row.get("price") is None else int(row["price"]),
            "bedrooms": None if row.get("bedrooms") is None else int(float(row["bedrooms"])),
            "size_sqft": None if row.get("size_sqft") is None else int(row["size_sqft"]),
            "property_type": row.get("property_type"),
        })
    # Defense in depth if the driver drops the outer order.
    sales.sort(key=lambda sale: str(sale.get("date") or ""), reverse=True)
    return {"count": len(sales), "sales": sales}


def _exact_project_row(place: str | None) -> dict | None:
    """Catalog project row when `place` matches one Prop-AI project name."""
    text = str(place or "").strip()
    if len(text) < 2:
        return None
    try:
        rows = fetch(
            """
            select name, master_project
            from propai_projects
            where coalesce(name, '') <> ''
              and upper(btrim(name)) = upper(btrim(%s))
            limit 1
            """,
            (text,),
        )
    except Exception:
        return None
    if rows and rows[0].get("name"):
        return {
            "name": str(rows[0]["name"]).strip(),
            "master_project": (
                str(rows[0].get("master_project") or "").strip() or None
            ),
        }
    return None


def _exact_project_label(place: str | None) -> str | None:
    """Canonical project name when `place` is exactly one Prop-AI project (e.g. AC3)."""
    row = _exact_project_row(place)
    return row["name"] if row else None


def _short_project_sale_alias(name: str | None) -> str | None:
    """'The Woods at Sobha Sanctuary' → 'The Woods' (sale-table short title)."""
    text = str(name or "").strip()
    lowered = text.lower()
    for sep in (" at ", " by ", " - ", " – "):
        idx = lowered.find(sep)
        if idx >= 3:
            short = text[:idx].strip()
            if len(short) >= 3:
                return short
    return None


def _named_project_sales_clause(project_row: dict) -> tuple[str, list]:
    """Match catalog project sales, including short sale-table aliases + master."""
    name = str(project_row.get("name") or "").strip()
    master = str(project_row.get("master_project") or "").strip()
    short = _short_project_sale_alias(name)
    if short and master and short.lower() != name.lower():
        return (
            "("
            "upper(btrim(project_name_prop_ai)) = upper(btrim(%s))"
            " or ("
            "upper(btrim(project_name_prop_ai)) = upper(btrim(%s))"
            " and upper(btrim(coalesce(master_project_prop_ai, ''))) = upper(btrim(%s))"
            ")"
            ")",
            [name, short, master],
        )
    return ("upper(btrim(project_name_prop_ai)) = upper(btrim(%s))", [name])


def latest_sales(
    place=None,
    limit: int = 8,
    offset: int = 0,
    bedrooms=None,
    year: int | None = None,
    month: int | None = None,
) -> dict:
    """Newest recorded sales, optionally limited to one Dubai area or project name."""
    cap = max(1, min(int(limit or 8), 15))
    start = max(0, int(offset or 0))
    parts = [
        CLEAN_SALE,
        "instance_date is not null",
        "coalesce(project_name_prop_ai, '') <> ''",
    ]
    params: list = []
    text = str(place or "").strip()
    project_row = _exact_project_row(text) if text else None
    if project_row and project_row.get("master_project") and (
        " ".join(project_row["name"].upper().split()) == " ".join(project_row["master_project"].upper().split())
    ):
        # A record named like its own community ("Dubai Marina" inside the Dubai Marina master project) is the
        # community's namesake, not one building: the sales wanted are the whole community's.
        project_row = None
    if project_row:
        # Named project (AC3 / The Woods at Sobha Sanctuary): keep building/plot
        # sales and resolve short sale-table titles under the same master.
        clause, extra = _named_project_sales_clause(project_row)
        parts.append(clause)
        params.extend(extra)
    else:
        parts.append(
            "(procedure_area_sqft is null or procedure_area_sqft between 50 and 100000)"
        )
        if text:
            clause, extra = contains_place_sql(
                ["area_name_prop_ai", "master_project_prop_ai", "project_name_prop_ai"],
                _expanded([text]),
            )
            parts.append(clause)
            params.extend(extra)
    if bedrooms is not None and bedrooms != "":
        try:
            parts.append("bedrooms = %s")
            params.append(int(float(bedrooms)))
        except (OverflowError, TypeError, ValueError):
            pass
    if year is not None and month is not None:
        parts.append("extract(year from instance_date) = %s")
        parts.append("extract(month from instance_date) = %s")
        params.extend([int(year), int(month)])
    elif year is not None:
        parts.append("extract(year from instance_date) = %s")
        params.append(int(year))
    rows = fetch(
        f"""
        select project_name_prop_ai as project,
               instance_date as sold_on,
               sale_price::bigint as price,
               bedrooms,
               procedure_area_sqft as size_sqft,
               property_type_en as property_type
        from propai_sales
        where {' and '.join(parts)}
        order by instance_date desc nulls last
        limit %s offset %s
        """,
        tuple([*params, cap, start]),
    )
    sales = []
    for row in rows or []:
        sold = row.get("sold_on")
        size = row.get("size_sqft")
        try:
            size_sqft = int(round(float(size))) if size is not None else None
        except (OverflowError, TypeError, ValueError):
            size_sqft = None
        beds = row.get("bedrooms")
        try:
            bedrooms = int(float(beds)) if beds is not None else None
        except (OverflowError, TypeError, ValueError):
            bedrooms = None
        sales.append({
            "project": row.get("project"),
            "date": sold.isoformat() if hasattr(sold, "isoformat") else (str(sold)[:10] if sold else None),
            "price": None if row.get("price") is None else int(row["price"]),
            "bedrooms": bedrooms,
            "size_sqft": size_sqft,
            "property_type": row.get("property_type"),
        })
    return {"count": len(sales), "sales": sales}


_MONTH_NUMBERS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_YEAR = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"[a-z]*\s+(20\d{2})\b",
    re.I,
)
_LAST_MONTHS = re.compile(r"\b(?:last|past)\s+(\d{1,2})\s+months?\b", re.I)
_PROJECT_HINT_FILLER = re.compile(
    r"\b(?:what|is|the|a|an|cheapest|lowest|highest|most\s+expensive|sale|sales|"
    r"sold|bedroom|bedrooms|bed|beds|studio|studios|apartment|apartments|"
    r"in|at|for|of|last|past|months?|years?|year|during|this|with|and|"
    r"dubai|date|price|prices)\b",
    re.I,
)


def parse_sale_month_year(question: str | None) -> tuple[int | None, int | None]:
    match = _MONTH_YEAR.search(question or "")
    if not match:
        return None, None
    month = _MONTH_NUMBERS.get(match.group(1).lower())
    try:
        year = int(match.group(2))
    except (TypeError, ValueError):
        return month, None
    return month, year


def parse_lookback_months(question: str | None, default: int | None = None) -> int | None:
    match = _LAST_MONTHS.search(question or "")
    if match:
        return max(1, min(int(match.group(1)), 60))
    if re.search(r"\blast\s+(?:one\s+)?year\b|\bpast\s+year\b|\b12\s+months\b", question or "", re.I):
        return 12
    return default


def project_named_in_question(question: str | None) -> str | None:
    """Project name left after stripping communities and sale filler (e.g. Hanover Square)."""
    text = question or ""
    # Prefer a project name that literally appears in the ask (longest match).
    rows = fetch(
        """
        select name as project
        from propai_projects
        where coalesce(name, '') <> ''
          and length(name) >= 5
          and position(upper(name) in upper(%s)) > 0
        order by length(name) desc
        limit 1
        """,
        (text,),
    )
    if rows:
        return rows[0].get("project")

    text = _MONTH_YEAR.sub(" ", text)
    text = _LAST_MONTHS.sub(" ", text)
    text = re.sub(r"\b20\d{2}\b", " ", text)
    text = re.sub(r"\blast\s+(?:one\s+)?year\b|\bpast\s+year\b", " ", text, flags=re.I)
    text = re.sub(
        r"\b(?:studio|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*-?\s*"
        r"(?:bed(?:room)?s?|br)\b",
        " ",
        text,
        flags=re.I,
    )
    for key in sorted(ALIASES, key=len, reverse=True):
        text = re.sub(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", " ", text, flags=re.I)
    for label in sorted({str(v) for v in ALIASES.values() if v}, key=len, reverse=True):
        text = re.sub(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", " ", text, flags=re.I)
    text = _PROJECT_HINT_FILLER.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s@&'.-]", " ", text, flags=re.I)
    hint = " ".join(text.split()).strip(" .?-")
    hint = re.sub(r"^\d+\s*-?\s*", "", hint).strip(" .?-")
    if len(hint) < 3:
        return None
    names = _sale_project_names()
    wanted = hint.upper()
    exact = next((name for name in names if name.upper() == wanted), None)
    if exact:
        return exact
    # Shortest name containing the hint (was LIKE '%hint%' over 1.4M rows: ~4 s per question).
    return min((name for name in names if wanted in name.upper()), key=len, default=None)


_SALE_PROJECT_NAMES: tuple[float, list[str]] | None = None


def _sale_project_names() -> list[str]:
    """Distinct project names in recorded sales (~4k), loaded once and kept for 6 hours."""
    # ponytail: per-process copy refreshed every 6 h; new names after a data refresh show up within that window.
    global _SALE_PROJECT_NAMES
    if _SALE_PROJECT_NAMES is None or time.time() - _SALE_PROJECT_NAMES[0] > 6 * 3600:
        rows = fetch(
            "select distinct project_name_prop_ai as project from propai_sales "
            "where deleted_at is null and coalesce(project_name_prop_ai, '') <> ''"
        )
        _SALE_PROJECT_NAMES = (time.time(), [row["project"] for row in rows if row.get("project")])
    return _SALE_PROJECT_NAMES[1]


def count_sales(
    place=None,
    bedrooms=None,
    year: int | None = None,
    month: int | None = None,
) -> dict:
    """How many recorded sales match place / beds / calendar month."""
    parts = [
        CLEAN_SALE,
        "instance_date is not null",
        "coalesce(project_name_prop_ai, '') <> ''",
    ]
    params: list = []
    text = str(place or "").strip()
    if text:
        clause, extra = contains_place_sql(
            ["area_name_prop_ai", "master_project_prop_ai", "project_name_prop_ai"],
            _expanded([text]),
        )
        parts.append(clause)
        params.extend(extra)
    if bedrooms is not None and bedrooms != "":
        parts.append("bedrooms = %s")
        params.append(int(float(bedrooms)))
    if year is not None and month is not None:
        parts.append("extract(year from instance_date) = %s")
        parts.append("extract(month from instance_date) = %s")
        params.extend([int(year), int(month)])
    elif year is not None:
        parts.append("extract(year from instance_date) = %s")
        params.append(int(year))
    rows = fetch(
        f"""
        select count(*)::int as n,
               min(instance_date)::date as first_sale,
               max(instance_date)::date as last_sale
        from propai_sales
        where {' and '.join(parts)}
        """,
        tuple(params),
    )
    row = rows[0] if rows else {}
    return {
        "count": int(row.get("n") or 0),
        "first_sale": row.get("first_sale"),
        "last_sale": row.get("last_sale"),
    }


def sale_window_stats(
    project=None,
    place=None,
    bedrooms=None,
    months: int | None = 12,
) -> dict:
    """Evidence behind a cheapest/highest answer: how many sales, min/median/max price.

    Uses the same outlier filters as cheapest_sale so the numbers agree.
    """
    parts = [
        CLEAN_SALE,
        "(procedure_area_sqft is null or procedure_area_sqft between 200 and 100000)",
        "instance_date is not null",
        "coalesce(project_name_prop_ai, '') <> ''",
    ]
    params: list = []
    project_name = str(project or "").strip()
    if project_name:
        parts.append("upper(project_name_prop_ai) = upper(%s)")
        params.append(project_name)
    else:
        text = str(place or "").strip()
        if text:
            clause, extra = contains_place_sql(
                ["area_name_prop_ai", "master_project_prop_ai", "project_name_prop_ai"],
                _expanded([text]),
            )
            parts.append(clause)
            params.extend(extra)
    if bedrooms is not None and bedrooms != "":
        parts.append("bedrooms = %s")
        params.append(int(float(bedrooms)))
    if months is not None:
        parts.append("instance_date >= current_date - (%s * interval '1 month')")
        params.append(int(months))
    rows = fetch(
        f"""
        select count(*)::int as n,
               min(sale_price)::bigint as low,
               max(sale_price)::bigint as high,
               percentile_cont(0.5) within group (order by sale_price)::bigint as median
        from propai_sales
        where {' and '.join(parts)}
        """,
        tuple(params),
    )
    row = rows[0] if rows else {}
    return {
        "count": int(row.get("n") or 0),
        "low": None if row.get("low") is None else int(row["low"]),
        "high": None if row.get("high") is None else int(row["high"]),
        "median": None if row.get("median") is None else int(row["median"]),
    }


def cheapest_sale(
    project=None,
    place=None,
    bedrooms=None,
    months: int | None = 12,
    highest: bool = False,
) -> dict | None:
    """Lowest (or, with highest=True, highest) non-outlier sale for a project or place."""
    parts = [
        CLEAN_SALE,
        # Tiny areas are usually bad rows, not livable 1-beds.
        "(procedure_area_sqft is null or procedure_area_sqft between 200 and 100000)",
        "instance_date is not null",
        "coalesce(project_name_prop_ai, '') <> ''",
    ]
    params: list = []
    project_name = str(project or "").strip()
    if project_name:
        project_row = _exact_project_row(project_name) or {"name": project_name}
        clause, extra = _named_project_sales_clause(project_row)
        parts.append(clause)
        params.extend(extra)
    else:
        text = str(place or "").strip()
        if text:
            project_row = _exact_project_row(text)
            if project_row:
                clause, extra = _named_project_sales_clause(project_row)
                parts.append(clause)
                params.extend(extra)
            else:
                clause, extra = contains_place_sql(
                    ["area_name_prop_ai", "master_project_prop_ai", "project_name_prop_ai"],
                    _expanded([text]),
                )
                parts.append(clause)
                params.extend(extra)
    if bedrooms is not None and bedrooms != "":
        parts.append("bedrooms = %s")
        params.append(int(float(bedrooms)))
    if months is not None:
        parts.append("instance_date >= current_date - (%s * interval '1 month')")
        params.append(int(months))
    rows = fetch(
        f"""
        select project_name_prop_ai as project,
               instance_date as sold_on,
               sale_price::bigint as price,
               bedrooms,
               procedure_area_sqft as size_sqft,
               property_type_en as property_type
        from propai_sales
        where {' and '.join(parts)}
        order by sale_price {'desc' if highest else 'asc'}, instance_date desc nulls last
        limit 1
        """,
        tuple(params),
    )
    if not rows:
        return None
    row = rows[0]
    sold = row.get("sold_on")
    size = row.get("size_sqft")
    try:
        size_sqft = int(round(float(size))) if size is not None else None
    except (OverflowError, TypeError, ValueError):
        size_sqft = None
    beds = row.get("bedrooms")
    try:
        bed_n = int(float(beds)) if beds is not None else None
    except (OverflowError, TypeError, ValueError):
        bed_n = None
    return {
        "project": row.get("project"),
        "date": sold.isoformat() if hasattr(sold, "isoformat") else (str(sold)[:10] if sold else None),
        "price": None if row.get("price") is None else int(row["price"]),
        "bedrooms": bed_n,
        "size_sqft": size_sqft,
        "property_type": row.get("property_type"),
    }


def top_developers_by_value(limit: int = 5, months: int = 12) -> list[dict]:
    """Developers ranked by total recorded sale value."""
    cap = max(1, min(int(limit or 5), 20))
    lookback = max(1, min(int(months or 12), 60))
    rows = fetch(
        f"""
        select developer_group as developer,
               count(*)::int as sales,
               sum(sale_price)::bigint as value
        from propai_sales
        where {CLEAN_SALE}
          and instance_date >= current_date - (%s * interval '1 month')
          and coalesce(developer_group, '') <> ''
        group by 1
        order by value desc nulls last
        limit %s
        """,
        (lookback, cap),
    )
    out = []
    for row in rows or []:
        out.append({
            "developer": row.get("developer"),
            "sales": int(row.get("sales") or 0),
            "value": int(row.get("value") or 0),
        })
    return out


_OFF_PLAN_STATUSES = ("ACTIVE", "PENDING", "NOT_STARTED", "CONDITIONAL_ACTIVATING")


def count_projects(place: str | None = None, developer: str | None = None, status: str | None = None) -> int:
    """Distinct residential project names, counted the way the site's project list counts them
    (no land plots, no cancelled projects). Optional place, developer and status (off_plan / ready)."""
    parts = [
        "coalesce(name, '') <> ''", "not is_plot",
        "coalesce(development_status, '') <> 'CANCELLED'", "lower(coalesce(usage, '')) = 'residential'",
    ]
    params: list = []
    if place and str(place).strip().lower() != "dubai":
        clause, extra = contains_place_sql(["master_project", "area", "name"], _expanded([str(place)]))
        parts.append(clause)
        params.extend(extra)
    if developer:
        parts.append("(developer ilike %s or developer_group ilike %s)")
        params.extend([f"%{developer}%"] * 2)
    if status == "off_plan":
        parts.append("development_status = any(%s)")
        params.append(list(_OFF_PLAN_STATUSES))
    elif status == "ready":
        parts.append("development_status = 'FINISHED'")
    rows = fetch(
        f"select count(distinct upper(name))::int as n from propai_projects where {' and '.join(parts)}",
        tuple(params),
    )
    return int(rows[0]["n"]) if rows else 0


def handover_projects(
    when: str = "recent", place: str | None = None, limit: int = 10, year: int | None = None
) -> list[dict]:
    """Residential projects ranked by handover.

    recent: DLD status FINISHED, newest completion date first (a past planned date alone is not a handover:
    many such projects are still 0% built). upcoming: under construction, nearest planned handover first.
    """
    parts = [
        "coalesce(name, '') <> ''", "not is_plot", "lower(coalesce(usage, '')) = 'residential'",
    ]
    params: list = []
    if when == "upcoming":
        parts += ["development_status = 'ACTIVE'", "end_date::date > current_date"]
        order, date_col = "end_date::date asc", "end_date"
    else:
        parts += ["development_status = 'FINISHED'", "completion_date <= current_date"]
        order, date_col = "completion_date desc", "completion_date"
    if year:
        # A named year: every project due (or completed) that year, in date order.
        parts.append(f"extract(year from {date_col}) = %s")
        params.append(int(year))
    if place and str(place).strip().lower() != "dubai":
        clause, extra = contains_place_sql(["master_project", "area", "name"], _expanded([str(place)]))
        parts.append(clause)
        params.extend(extra)
    rows = fetch(
        f"""
        select * from (
          select distinct on (upper(name)) key, name, area, master_project, developer, developer_group,
                 percent_completed, {date_col}::date as handover_on
          from propai_projects
          where {' and '.join(parts)}
          order by upper(name), {date_col} desc
        ) one_per_name
        order by handover_on {'asc' if when == 'upcoming' else 'desc'}, name
        limit %s
        """,
        (*params, max(1, min(int(limit), 20))),
    )
    return [{
        "id": row["key"],
        "name": row.get("name"),
        "developer_name": str(row.get("developer_group") or row.get("developer") or "").title() or None,
        "community": row.get("master_project") or row.get("area"),
        "sub_community": row.get("area"),
        "city": "Dubai",
        "handover_time": _handover(row.get("handover_on")),
        "handover_on": row.get("handover_on"),
        "percent_completed": float(row["percent_completed"]) if row.get("percent_completed") is not None else None,
        "source": "propai",
    } for row in rows]


def _place_label(value) -> str:
    text = str(value or "").strip()
    return text.title() if text.isupper() else text


def area_ranking(metric: str, beds: int | None = None, limit: int = 8) -> list[dict]:
    """Dubai areas ranked over the last 12 months, apartments only so areas compare like for like.

    cheapest_rent / priciest_rent: median yearly rent (registered contracts).
    cheapest_buy / priciest_buy: median sale price (recorded sales).
    best_yield: median yearly rent / median sale price (gross, rough: different homes on each side).
    Areas need at least 200 rents / 100 sales so one odd building cannot top the list.
    """
    place = "coalesce(nullif(btrim(master_project_prop_ai), ''), area_name_prop_ai)"
    bed_sql, bed_params = ("and bedrooms = %s", [int(beds)]) if beds is not None else ("", [])
    rents = f"""
        select {place} as place, count(*) as rents,
               percentile_cont(0.5) within group (order by annual_amount) as rent
        from propai_rents
        where deleted_at is null and coalesce(outlier_flag, '') <> 'True' and annual_amount > 0
          and property_type = 'Apartment' and contract_start_date between current_date - 365 and current_date
          {bed_sql}
        group by 1 having count(*) >= 200"""
    # ponytail: name filter; labour-camp "apartments" in industrial zones are not homes people compare.
    rents = f"select * from ({rents}) all_rents where place !~* 'industrial'"
    # Yield compares rents with ready homes: off-plan launch prices are not what a rented flat costs.
    ready_sql = "and development_status = 'Existing'" if metric == "best_yield" else ""
    sales = f"""
        select {place} as place, count(*) as sales,
               percentile_cont(0.5) within group (order by sale_price) as price
        from propai_sales
        where {CLEAN_SALE} and property_sub_type_en = 'Flat' and instance_date >= current_date - 365
          {ready_sql} {bed_sql}
        group by 1 having count(*) >= 100"""
    sales = f"select * from ({sales}) all_sales where place !~* 'industrial'"
    cap = max(1, min(int(limit), 15))
    if metric in ("cheapest_rent", "priciest_rent"):
        sql = f"select place, rents, rent::bigint as rent from ({rents}) r order by rent {'desc' if metric == 'priciest_rent' else 'asc'} limit %s"
        params = (*bed_params, cap)
    elif metric in ("cheapest_buy", "priciest_buy"):
        sql = f"select place, sales, price::bigint as price from ({sales}) s order by price {'desc' if metric == 'priciest_buy' else 'asc'} limit %s"
        params = (*bed_params, cap)
    else:
        sql = f"""select r.place, r.rents, s.sales, r.rent::bigint as rent, s.price::bigint as price,
                         round((100 * r.rent / s.price)::numeric, 1)::float as yield_pct
                  from ({rents}) r join ({sales}) s using (place)
                  order by yield_pct desc limit %s"""
        params = (*bed_params, *bed_params, cap)
    return [{**row, "place": _place_label(row["place"])} for row in fetch(sql, params)]


def top_projects_by_sales(
    place: str | None = None, year: int | None = None, months: int = 12, limit: int = 10
) -> list[dict]:
    """Projects ranked by number of recorded sales: a calendar year when given, else the last N months."""
    parts, params = [CLEAN_SALE, "coalesce(project_name_prop_ai, '') <> ''"], []
    if year:
        parts.append("extract(year from instance_date) = %s")
        params.append(int(year))
    else:
        parts.append("instance_date >= current_date - (%s * interval '1 month')")
        params.append(max(1, min(int(months or 12), 60)))
    if place and str(place).strip().lower() != "dubai":
        clause, extra = contains_place_sql(
            ["area_name_prop_ai", "master_project_prop_ai", "project_name_prop_ai"], _expanded([str(place)])
        )
        parts.append(clause)
        params.extend(extra)
    rows = fetch(
        f"""
        select project_name_prop_ai as name,
               max(coalesce(nullif(btrim(master_project_prop_ai), ''), area_name_prop_ai)) as community,
               max(developer_group) as developer,
               count(*)::int as sales,
               percentile_cont(0.5) within group (order by sale_price)::bigint as median_price
        from propai_sales
        where {' and '.join(parts)}
        group by 1
        order by sales desc, name
        limit %s
        """,
        (*params, max(1, min(int(limit), 15))),
    )
    return [{**row, "name": _place_label(row["name"]), "community": _place_label(row["community"]),
             "developer": _place_label(row["developer"])} for row in rows]


def agent_by_id(agent_id: str) -> dict | None:
    key = _id_index().get(agent_id)
    if not key:
        return None
    rows, _ = agent_projects({"key": key}, limit=1)
    return rows[0] if rows else None


def rent_summary(areas, beds_min=None, beds_max=None) -> dict | None:
    """Median registered rent for a place. None when the question did not name a place."""
    asked = _expanded(areas)
    if not asked:
        return None
    lock, lock_params = sales_place_sql(asked, _official_areas(asked))
    clauses = [
        "deleted_at is null",
        "annual_amount > 0",
        "contract_start_date >= date '2025-09-01'",
        lock,
    ]
    params: list = list(lock_params)
    if beds_min is not None:
        clauses.append("bedrooms >= %s")
        params.append(beds_min)
    if beds_max is not None:
        clauses.append("bedrooms <= %s")
        params.append(beds_max)
    try:
        row = fetch(
            f"""
            select count(*)::int as n,
                   percentile_cont(0.5) within group (order by annual_amount)::bigint as median
            from propai_rents
            where {' and '.join(clauses)}
            """,
            tuple(params),
        )[0]
    except Exception:
        return None
    if not row.get("n"):
        return None
    median = row.get("median")
    return {
        "contracts": row["n"],
        "median_annual_rent_aed": None if median is None else int(median),
        "note": "Registered Prop-AI rents since September 2025 (propai_rents).",
    }


def _rent_row_public(row: dict) -> dict:
    """Card-shaped rent contract from propai_rents."""
    annual = row.get("annual_amount")
    try:
        annual_i = int(round(float(annual))) if annual is not None else None
    except (TypeError, ValueError):
        annual_i = None
    beds = row.get("bedrooms")
    try:
        beds_i = int(float(beds)) if beds is not None else None
    except (TypeError, ValueError):
        beds_i = None
    size = row.get("actual_area_sqft")
    try:
        size_i = int(round(float(size))) if size is not None else None
    except (TypeError, ValueError):
        size_i = None
    start = row.get("contract_start_date")
    title_bits = [
        str(row.get("building_name_prop_ai") or row.get("project_name_prop_ai") or "").strip(),
        f"{beds_i} bed" if beds_i is not None else None,
        str(row.get("property_type") or "").strip() or None,
    ]
    title = " · ".join(bit for bit in title_bits if bit) or "Registered rent"
    contract_id = str(row.get("pai_contract_id") or "").strip()
    return {
        "listing_id": contract_id or None,
        "title": title,
        "property_type": row.get("property_type"),
        "beds": beds_i,
        "baths": None,
        "size_sqft": size_i,
        "furnishing": None,
        "community": row.get("area_name_prop_ai") or row.get("master_project_prop_ai"),
        "sub_community": row.get("master_project_prop_ai"),
        "agency_name": None,
        "price_aed": annual_i,
        "annual_rent_aed": annual_i,
        "monthly_rent_aed": None if annual_i is None else int(round(annual_i / 12)),
        "rent_frequency": "yearly",
        "contract_start": start.isoformat() if hasattr(start, "isoformat") else (str(start)[:10] if start else None),
        "project": row.get("project_name_prop_ai"),
        "building": row.get("building_name_prop_ai"),
        "source": "propai_rents",
        "entity_type": "listing",
    }


def list_rents(
    areas=None,
    beds_min=None,
    beds_max=None,
    *,
    limit: int = 8,
    sort: str = "newest",
) -> dict:
    """Browse registered rents from propai_rents (not Bayut listings).

    Default newest contracts first (bi-daily loads). Pass sort='cheapest' /
    'priciest' when the user asks for price extremes.
    """
    asked = _expanded(areas)
    if not asked:
        return {"matching_listings": 0, "listings": [], "note": "Name a Dubai area to search Prop-AI rents."}
    lock, lock_params = sales_place_sql(asked, _official_areas(asked))
    clauses = [
        "deleted_at is null",
        "annual_amount > 0",
        "contract_start_date >= date '2025-09-01'",
        lock,
    ]
    params: list = list(lock_params)
    if beds_min is not None:
        clauses.append("bedrooms >= %s")
        params.append(beds_min)
    if beds_max is not None:
        clauses.append("bedrooms <= %s")
        params.append(beds_max)
    sort_key = str(sort or "newest").lower()
    order = {
        "cheapest": "annual_amount asc nulls last, contract_start_date desc nulls last",
        "priciest": "annual_amount desc nulls last, contract_start_date desc nulls last",
        "newest": "contract_start_date desc nulls last, annual_amount asc nulls last",
    }.get(sort_key, "contract_start_date desc nulls last, annual_amount asc nulls last")
    cap = max(1, min(int(limit or 8), 20))
    where = " and ".join(clauses)
    try:
        total = fetch(
            f"select count(*)::int as n from propai_rents where {where}",
            tuple(params),
        )[0]["n"]
        rows = fetch(
            f"""
            select pai_contract_id, property_type, bedrooms, actual_area_sqft, annual_amount,
                   contract_start_date, area_name_prop_ai, master_project_prop_ai,
                   project_name_prop_ai, building_name_prop_ai
            from propai_rents
            where {where}
            order by {order}
            limit %s
            """,
            tuple([*params, cap]),
        )
    except Exception:
        return {"matching_listings": 0, "listings": [], "note": "Prop-AI rent search failed."}
    listings = [_rent_row_public(row) for row in (rows or [])]
    return {
        "matching_listings": int(total or 0),
        "showing": len(listings),
        "sorted_by": sort,
        "listings": listings,
        "note": "Registered Prop-AI rents (propai_rents), not advertised Bayut listings.",
    }


def rent_stats(areas=None, beds_min=None, beds_max=None) -> dict | None:
    """Aggregate Prop-AI rents for an area."""
    asked = _expanded(areas)
    if not asked:
        return None
    lock, lock_params = sales_place_sql(asked, _official_areas(asked))
    clauses = [
        "deleted_at is null",
        # Under AED 10k a year is a typo or a parking/storage contract (a "2-bed" at AED 125 was the lowest).
        "coalesce(outlier_flag, '') <> 'True'",
        "annual_amount >= 10000",
        "contract_start_date >= date '2025-09-01'",
        lock,
    ]
    params: list = list(lock_params)
    if beds_min is not None:
        clauses.append("bedrooms >= %s")
        params.append(beds_min)
    if beds_max is not None:
        clauses.append("bedrooms <= %s")
        params.append(beds_max)
    try:
        row = fetch(
            f"""
            select count(*)::int as n,
                   round(avg(annual_amount))::bigint as avg_annual,
                   percentile_cont(0.5) within group (order by annual_amount)::bigint as median_annual,
                   percentile_cont(0.25) within group (order by annual_amount)::bigint as p25,
                   percentile_cont(0.75) within group (order by annual_amount)::bigint as p75,
                   min(annual_amount)::bigint as min_annual,
                   max(annual_amount)::bigint as max_annual
            from propai_rents
            where {' and '.join(clauses)}
            """,
            tuple(params),
        )[0]
    except Exception:
        return None
    if not row.get("n"):
        return None
    return {
        "listings": int(row["n"]),
        "avg_annual_rent_aed": None if row.get("avg_annual") is None else int(row["avg_annual"]),
        "median_annual_rent_aed": None if row.get("median_annual") is None else int(row["median_annual"]),
        "typical_range_annual_aed": [
            None if row.get("p25") is None else int(row["p25"]),
            None if row.get("p75") is None else int(row["p75"]),
        ],
        "min_annual_rent_aed": None if row.get("min_annual") is None else int(row["min_annual"]),
        "max_annual_rent_aed": None if row.get("max_annual") is None else int(row["max_annual"]),
        "note": "Registered Prop-AI rents (propai_rents).",
    }


def project_payment_plans(project_key: str | None = None, name: str | None = None) -> dict | None:
    """Payment plans from propai_project_plans for a project key or name."""
    key = str(project_key or "").strip()
    needle = str(name or "").strip()
    row = None
    try:
        if key:
            found = fetch(
                "select location_key, payment_plans, property_types "
                "from propai_project_plans where location_key = %s limit 1",
                (key,),
            )
            row = found[0] if found else None
        if row is None and needle:
            found = fetch(
                """
                select p.location_key, p.payment_plans, p.property_types, pr.name
                from propai_project_plans p
                join propai_projects pr on pr.key = p.location_key
                where pr.name ilike %s
                order by jsonb_array_length(coalesce(p.payment_plans, '[]'::jsonb)) desc
                limit 1
                """,
                (f"%{needle}%",),
            )
            row = found[0] if found else None
    except Exception:
        return None
    if not row:
        return None
    plans = row.get("payment_plans") or []
    if isinstance(plans, str):
        try:
            import json
            plans = json.loads(plans)
        except Exception:
            plans = []
    if not isinstance(plans, list) or not plans:
        return {
            "location_key": row.get("location_key"),
            "name": row.get("name"),
            "payment_plans": [],
            "property_types": row.get("property_types"),
            "note": "No payment plan recorded for this project in Prop-AI.",
        }
    return {
        "location_key": row.get("location_key"),
        "name": row.get("name"),
        "payment_plans": plans,
        "property_types": row.get("property_types"),
        "note": "From propai_project_plans.",
    }


_BROKER_COLUMNS = (
    "brn, name_en, agency_name, card_rank, card_rank_id, orn, license_number, "
    "awards_count, card_expiry_date, photo_url, office_rank"
)
_BROKER_PUBLIC = (
    "id", "brn", "name", "name_en", "agency_name", "card_rank", "orn",
    "license_number", "awards_count", "card_expiry_date", "photo_url", "office_rank",
)


def _public_broker(row: dict) -> dict:
    """Customer-facing broker fields. Email and phone stay out."""
    expiry = row.get("card_expiry_date")
    expiry_text = str(expiry)[:10] if expiry else None
    name = str(row.get("name_en") or "").strip() or None
    brn = str(row.get("brn") or "").strip()
    awards = row.get("awards_count")
    return {
        "id": brn,
        "brn": brn,
        "name": name,
        "name_en": name,
        "agency_name": str(row.get("agency_name") or "").strip() or None,
        "card_rank": str(row.get("card_rank") or "").strip() or None,
        "orn": str(row.get("orn") or "").strip() or None,
        "license_number": str(row.get("license_number") or "").strip() or None,
        "awards_count": None if awards is None else int(awards),
        "card_expiry_date": expiry_text,
        "photo_url": str(row.get("photo_url") or "").strip() or None,
        "office_rank": str(row.get("office_rank") or "").strip() or None,
    }


_BROKER_POOL = 80


def _broker_rows(name=None, brn=None, brns=None, ranks=None, limit: int = 8) -> list[dict]:
    parts = ["coalesce(name_en, '') <> ''", "(card_expiry_date is null or card_expiry_date >= current_date)"]
    params: list = []
    wanted = str(brn or "").strip()
    if wanted:
        parts.append("brn = %s")
        params.append(wanted)
    elif brns:
        ids = [str(item).strip() for item in brns if str(item).strip()]
        if not ids:
            return []
        parts.append("brn = any(%s)")
        params.append(ids)
    else:
        text = str(name or "").strip()
        if text:
            if re.fullmatch(r"\d{3,12}", text):
                parts.append("brn = %s")
                params.append(text)
            else:
                parts.append("(name_en ilike %s or agency_name ilike %s)")
                like = f"%{text}%"
                params.extend([like, like])
        if ranks == "top":
            parts.append("lower(coalesce(card_rank, '')) like %s")
            params.append("%five%star%")
        elif ranks == "mid":
            parts.append(
                "(lower(coalesce(card_rank, '')) like %s "
                "or lower(coalesce(card_rank, '')) like %s)"
            )
            params.extend(["%four%star%", "%three%star%"])
    rows = fetch(
        f"""
        select {_BROKER_COLUMNS}
        from propai_brokers
        where {' and '.join(parts)}
        order by card_rank_id desc nulls last, awards_count desc nulls last, brn
        limit %s
        """,
        tuple([*params, limit]),
    )
    return rows or []


def _place_seed(place: str) -> int:
    return int(hashlib.sha256((place or "").strip().lower().encode()).hexdigest(), 16)


_WHOLE_MARKET = {
    "dubai",
    "uae",
    "u.a.e",
    "u.a.e.",
    "united arab emirates",
    "emirate of dubai",
    "dubai uae",
    "dubai, uae",
}
# Cities and emirates ACOT does not cover. Checked before any DLD lookup so a
# foreign name can never be used as a seed for Dubai licensed agents.
_OUTSIDE_DUBAI = {
    "chennai", "mumbai", "delhi", "new delhi", "bangalore", "bengaluru",
    "hyderabad", "kolkata", "calcutta", "pune", "ahmedabad", "kochi", "goa",
    "anantapur", "vijayawada", "visakhapatnam", "tirupati", "warangal",
    "coimbatore", "madurai", "mysore", "mysuru", "nagpur", "jaipur", "lucknow",
    "london", "manchester", "paris", "berlin", "rome", "madrid",
    "new york", "nyc", "los angeles", "toronto", "singapore", "hong kong",
    "tokyo", "sydney", "melbourne", "cairo", "istanbul", "moscow",
    "doha", "riyadh", "jeddah", "kuwait", "bahrain", "muscat",
    "abu dhabi", "abudhabi", "sharjah", "ajman", "fujairah",
    "ras al khaimah", "rak", "umm al quwain", "al ain",
}
def _normalize_place(place) -> str:
    text = re.sub(r"\s+", " ", str(place or "").strip(" .?!,;:'\""))
    text = re.sub(r"^(?:the|in|at|for)\s+", "", text, flags=re.I)
    # Chip / spoken aliases: "Jumeirah Village Circle (JVC)" → community name.
    # Leaving "(JVC)" made resolve_dubai_place fuzzy-match unrelated projects.
    text = re.sub(r"\s*\([^)]{1,40}\)", "", text)
    # Drop budget tails so "Dubai starting above AED 5 million" → "Dubai".
    text = re.sub(
        r"\b(?:starting(?:\s+(?:above|from|at))?|above|over|under|below|up\s+to|"
        r"around|about|budget|max(?:imum)?|min(?:imum)?|price|prices|"
        r"aed|dh|dhs)\b.*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s+(?:area|community|city|emirate|details?|agents?|brokers?)$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r",?\s*dubai$", "", text, flags=re.I).strip()
    return " ".join(text.split())


def _static_dubai_names() -> dict[str, str]:
    names = {key.lower(): ALIASES[key] for key in ALIASES}
    for proper in ALIASES.values():
        names[proper.lower()] = proper
    extras = (
        "Dubai Marina", "Business Bay", "Downtown Dubai", "Palm Jumeirah",
        "Jumeirah Village Circle", "Jumeirah Village Triangle",
        "Jumeirah Lake Towers", "Jumeirah Beach Residence", "Dubai Hills Estate",
        "Dubai Creek Harbour", "Dubai Silicon Oasis", "Dubai Sports City",
        "Dubai Investments Park", "Al Barsha", "Al Furjan", "Arabian Ranches",
        "Motor City", "Damac Hills", "Meydan", "City Walk", "DIFC",
        "Dubai Harbour", "Dubai South", "MBR City", "Mohammed Bin Rashid City",
        "International City", "Discovery Gardens", "Dubai Production City",
        "Dubai Studio City", "Jumeirah Islands", "Jumeirah Park", "The Springs",
        "The Meadows", "Emirates Hills", "The Lakes", "The Views", "Al Jaddaf",
        "Deira", "Bur Dubai", "Karama", "Mirdif", "Town Square", "Tilal Al Ghaf",
        "Emaar Beachfront", "Bluewaters", "Bluewaters Island", "Dubai Islands",
        "Port De La Mer", "La Mer", "Arjan",
    )
    for proper in extras:
        names.setdefault(proper.lower(), proper)
    # The data stores numbered master plans in Roman numerals (ARABIAN RANCHES III);
    # "Arabian Ranches 3" must stay on that master, not the original.
    for number, roman in (("2", "II"), ("3", "III")):
        proper = f"Arabian Ranches {roman}"
        names[proper.lower()] = proper
        names[f"arabian ranches {number}"] = proper
        names[f"arabian ranches-{number}"] = proper
    return names


def is_outside_dubai(place) -> bool:
    """True for a named city or emirate that is not Dubai."""
    return outside_dubai_city(place) is not None


def outside_dubai_city(text) -> str | None:
    """The foreign city/emirate named in this text, if any."""
    blob = str(text or "").lower()
    if not blob:
        return None
    if known_dubai_label(text):
        return None
    for city in sorted(_OUTSIDE_DUBAI, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", blob):
            return " ".join(part.capitalize() for part in city.split())
    return None


def _compact_place(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _outside_compacts() -> set[str]:
    return {_compact_place(city) for city in _OUTSIDE_DUBAI}


def _fuzzy_dubai_label(compact: str, known: dict[str, str]) -> str | None:
    """One known Dubai name within 1–2 letter edits. Ties stay unmatched."""
    from app.retrieval.hybrid.question_text import _edit_distance

    scored: dict[str, int] = {}
    for name, proper in known.items():
        name_compact = _compact_place(name)
        if abs(len(compact) - len(name_compact)) > 2:
            continue
        dist = _edit_distance(compact, name_compact)
        if dist > 2:
            continue
        previous = scored.get(proper)
        if previous is None or dist < previous:
            scored[proper] = dist
    if not scored:
        return None
    best = min(scored.values())
    winners = [proper for proper, dist in scored.items() if dist == best]
    if len(winners) != 1:
        return None
    return winners[0]


def known_dubai_label(place) -> str | None:
    """Whole-market or a known Dubai community from the static list. Does not query the database."""
    raw = _normalize_place(place)
    if not raw:
        whole = re.sub(r"\s+", " ", str(place or "").strip(" .?!,;:'\"")).lower()
        if whole in _WHOLE_MARKET or re.sub(r",?\s*dubai$", "", whole).strip() == "":
            return "Dubai"
        return None
    key = raw.lower()
    if key in _WHOLE_MARKET:
        return "Dubai"
    known = _static_dubai_names()
    if key in known:
        return known[key]
    compact = _compact_place(key)
    if len(compact) < 4:
        return None
    for name, proper in known.items():
        if _compact_place(name) == compact:
            return proper
    if key in _OUTSIDE_DUBAI or compact in _outside_compacts():
        return None
    if len(compact) < 5:
        return None
    return _fuzzy_dubai_label(compact, known)


def resolve_dubai_place(place) -> str | None:
    """Canonical Dubai area/project name, 'Dubai' for the whole market, or None if out of coverage."""
    if is_outside_dubai(place):
        return None
    known = known_dubai_label(place)
    if known:
        return known
    raw = _normalize_place(place)
    key = raw.lower()
    if len(key) < 3:
        return None
    like = "%" + re.sub(r"([%_\\])", r"\\\1", key) + "%"
    try:
        rows = fetch(
            """
            select label, area, master, 1 as weight
            from propai_places
            where level in ('area', 'master')
              and (
                lower(btrim(coalesce(label, ''))) = %s
                or lower(btrim(coalesce(area, ''))) = %s
                or lower(btrim(coalesce(master, ''))) = %s
                or (length(%s) >= 5 and (
                    lower(label) like %s
                    or lower(area) like %s
                    or lower(coalesce(master, '')) like %s
                ))
              )
            union all
            select name as label, area, master_project as master, 2 as weight
            from propai_projects
            where coalesce(name, '') <> ''
              and (
                lower(btrim(name)) = %s
                or lower(btrim(coalesce(area, ''))) = %s
                or lower(btrim(coalesce(master_project, ''))) = %s
                or (length(%s) >= 5 and (
                    lower(name) like %s
                    or lower(area) like %s
                    or lower(coalesce(master_project, '')) like %s
                ))
              )
            order by weight
            limit 5
            """,
            (key, key, key, key, like, like, like, key, key, key, key, like, like, like),
        )
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    return (
        str(row.get("label") or row.get("master") or row.get("area") or "").strip()
        or None
    )


def seeded_broker_pick(rows: list[dict], place: str | None, limit: int, shuffle: bool) -> list[dict]:
    """Stable per place: same community keeps the same slice, another community shifts."""
    public = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "name_en" in row or "card_rank_id" in row:
            public.append(_public_broker(row))
        else:
            public.append(dict(row))
    if not public:
        return []
    cap = max(1, min(int(limit or 5), 10))
    seed = _place_seed(place or "")
    if shuffle:
        picked = list(public)
        random.Random(seed).shuffle(picked)
        return picked[:cap]
    if len(public) <= cap:
        return public
    start = seed % (len(public) - cap + 1)
    return public[start:start + cap]


def list_brokers(name=None, brn=None, limit: int = 8, place=None, band=None) -> list[dict]:
    """Named lookup, or a place-seeded slice of Five Star (top) / 3-4 star (mid) brokers."""
    cap = max(1, min(int(limit or 8), 10))
    if (place and is_outside_dubai(place)) or (name and not brn and is_outside_dubai(name)):
        return []
    if name or brn:
        rows = _broker_rows(name=name, brn=brn, limit=cap)
        if rows or not str(name or "").strip() or brn:
            return [_public_broker(row) for row in rows]
        try:
            from app.retrieval.propai_vector_search import search_propai
            hits = search_propai(str(name).strip(), table="brokers", limit=cap)
        except Exception:
            return []
        ids = [str(hit.get("brn") or "").strip() for hit in hits if hit.get("brn")]
        if not ids:
            return []
        found = [_public_broker(row) for row in _broker_rows(brns=ids, limit=cap)]
        by_id = {row["brn"]: row for row in found}
        return [by_id[item] for item in ids if item in by_id][:cap]

    pool = max(_BROKER_POOL, cap)
    if band == "mid":
        rows = _broker_rows(ranks="mid", limit=pool)
        if not rows:
            all_rows = _broker_rows(limit=pool)
            rows = all_rows[len(all_rows) // 2:] or all_rows
        return seeded_broker_pick(rows, place, cap, shuffle=True)
    if band == "top" or place:
        rows = _broker_rows(ranks="top", limit=pool)
        if not rows:
            rows = _broker_rows(limit=pool)
        return seeded_broker_pick(rows, place, cap, shuffle=False)
    return [_public_broker(row) for row in _broker_rows(limit=cap)]


def sale_summary(communities) -> dict | None:
    """Project count and median recent sale for a place in the new data."""
    asked = _expanded(communities)
    if not asked:
        return None
    official = _official_areas(asked)
    project_lock, project_params = contains_place_sql(["master_project", "area", "name"], asked)
    sales_lock, sales_params = sales_place_sql(asked, official)
    try:
        projects = fetch(
            f"""
            select count(*)::int as n
            from propai_projects
            where coalesce(name, '') <> ''
              and {project_lock}
            """,
            tuple(project_params),
        )[0]["n"]
        sales = fetch(
            f"""
            select count(*)::int as n,
                   percentile_cont(0.5) within group (order by sale_price)::bigint as median
            from propai_sales
            where {CLEAN_SALE}
              and instance_date >= current_date - interval '{int(RECENT_SALE_MONTHS)} months'
              and {sales_lock}
            """,
            tuple(sales_params),
        )[0]
    except Exception:
        return None
    if not projects and not sales.get("n"):
        return None
    median = sales.get("median")
    return {
        "projects": projects,
        "sales": sales.get("n") or 0,
        "median_recent_sale_aed": None if median is None else int(median),
        "note": (
            f"Recorded sales from the last {int(RECENT_SALE_MONTHS)} months in the new data. "
            "Prefer this price over an older starting price."
        ),
    }
