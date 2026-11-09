"""
Read the new Prop-AI tables in the same shape as the old project rows.

A question is matched to a place with the alias list, then the place embedding.
Projects, status, and a recent sale price are loaded with SQL and returned as the
fields the current cards already show. If nothing matches, the caller keeps the old data.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime

from app.database.postgres import fetch
from app.listings.tools import ALIASES
from app.retrieval.propai_vector_search import search_places

MIN_SCORE = 0.55
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
    return " ".join(kept.split())


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


def _handover(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


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
        order by name
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
            "price_kind": "recent_sale",
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
    return projects


_SALE_FACTS: dict[str, dict] = {}


def _add_sale_facts(projects: list[dict]) -> None:
    names = [str(project["name"]).upper() for project in projects if project.get("name")]
    if not names:
        return
    missing = [name for name in dict.fromkeys(names) if name not in _SALE_FACTS]
    if missing:
        try:
            stats = fetch(
                """
                select upper(project_name_prop_ai) as name,
                       percentile_cont(0.5) within group (order by sale_price) as price,
                       min(bedrooms) filter (where bedrooms between 0 and 10) as bedroom_min,
                       max(bedrooms) filter (where bedrooms between 0 and 10) as bedroom_max
                from propai_sales
                where deleted_at is null and sale_price > 0
                  and instance_date >= current_date - interval '24 months'
                  and upper(project_name_prop_ai) = any(%s)
                group by 1
                """,
                (missing,),
            )
        except Exception:
            stats = []
        for row in stats:
            _SALE_FACTS[str(row["name"])] = row
        for name in missing:
            _SALE_FACTS.setdefault(name, {})
    for project in projects:
        stat = _SALE_FACTS.get(str(project["name"]).upper())
        if not stat:
            continue
        if stat.get("price"):
            project["price"] = int(round(float(stat["price"])))
        project["bedroom_min"] = stat.get("bedroom_min")
        project["bedroom_max"] = stat.get("bedroom_max")


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
    likes = _likes(names)
    try:
        rows = fetch(
            """
            select distinct upper(area) as area
            from propai_places
            where level in ('area', 'master')
              and area is not null
              and (label ilike any(%s::text[]) or area ilike any(%s::text[]) or master ilike any(%s::text[]))
            """,
            (likes, likes, likes),
        )
    except Exception:
        return [name.upper() for name in names]
    found = [row["area"] for row in rows if row.get("area")]
    return found or [name.upper() for name in names]


def agent_projects(filters: dict, limit: int = 8, sort: str = "cheapest") -> tuple[list[dict], int]:
    """New-data projects in the old project-row shape. Empty when a filter the new rows cannot answer is set."""
    if filters.get("min_size") or filters.get("max_size") or filters.get("property_types"):
        return [], 0
    parts, params = ["coalesce(name, '') <> ''"], []
    if filters.get("key"):
        parts.append("key = %s")
        params.append(filters["key"])
    communities = _expanded(filters.get("communities"))
    if communities:
        likes = _likes(communities)
        parts.append("(master_project ilike any(%s::text[]) or area ilike any(%s::text[]) or name ilike any(%s::text[]))")
        params += [likes, likes, likes]
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
    where = " and ".join(parts)
    ranked = (
        "lower(name) = lower(%s) desc, length(name), name"
        if name else
        "name"
    )
    # Cap candidates before sale stats; ranking still uses the same pool size as before.
    fetch_limit = 200
    query_params = [*params, name] if name else list(params)
    rows = fetch(
        f"""
        select key, name, developer, area, master_project, offplan_status, end_date, description
        from (
            select distinct on (upper(name))
                   key, name, developer, area, master_project, offplan_status, end_date,
                   left(description, 400) as description
            from propai_projects
            where {where}
            order by upper(name), (developer is not null) desc, name
        ) unique_projects
        order by {ranked}
        limit %s
        """,
        tuple([*query_params, fetch_limit]),
    )
    if not rows:
        return [], 0
    # Skip a second count scan when the page is already the full set.
    if len(rows) < fetch_limit:
        total = len(rows)
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
            "price_kind": "recent_sale",
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
    _add_sale_facts(projects)
    bedrooms = filters.get("bedrooms")
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
        ]
        total = len(projects)
    for project in projects:
        for key in ("bedroom_min", "bedroom_max"):
            if project.get(key) is not None:
                project[key] = int(float(project[key]))

    def rank(project):
        price = project.get("price") or 0
        handover = project.get("handover_time") or datetime.max
        if sort == "priciest":
            return (price <= 0, -(price or 0))
        if sort == "earliest_handover":
            return (project.get("handover_time") is None, handover)
        if sort == "latest_handover":
            missing = project.get("handover_time") is None
            return (missing, datetime.min if missing else handover)
        return (price <= 0, price or 10**18)

    if sort == "latest_handover":
        projects.sort(
            key=lambda project: (project.get("handover_time") is not None, project.get("handover_time") or datetime.min),
            reverse=True,
        )
    else:
        projects.sort(key=rank)
    projects = projects[:limit]
    _attach_photos(projects)
    for project in projects:
        project["id"] = _agent_id(str(project["id"]))
    return projects, total


def recent_sales(name, limit: int = 8, per_project: int = 5, offset: int = 0) -> dict:
    """Recent recorded sales for the named projects. One query, newest first per project."""
    raw = name if isinstance(name, (list, tuple)) else [name]
    projects = []
    for item in raw:
        text = " ".join(str(item or "").split())
        if len(text) >= 3 and text.upper() not in projects:
            projects.append(text.upper())
        if len(projects) >= limit:
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
        """,
        (projects, start, start + per),
    )
    by_name = {key: [] for key in projects}
    for row in rows:
        sold = row.get("sold_on")
        key = str(row.get("project") or "").upper()
        sale = {
            "project": row.get("project"),
            "date": sold.isoformat() if hasattr(sold, "isoformat") else (str(sold)[:10] if sold else None),
            "price": None if row.get("price") is None else int(row["price"]),
            "bedrooms": None if row.get("bedrooms") is None else int(float(row["bedrooms"])),
            "size_sqft": None if row.get("size_sqft") is None else int(row["size_sqft"]),
            "property_type": row.get("property_type"),
        }
        if key in by_name:
            by_name[key].append(sale)
    sales = []
    for key in projects:
        sales.extend(by_name.get(key) or [])
    return {"count": len(sales), "sales": sales}


def agent_by_id(agent_id: str) -> dict | None:
    key = _id_index().get(agent_id)
    if not key:
        return None
    rows, _ = agent_projects({"key": key}, limit=1)
    return rows[0] if rows else None


def rent_summary(areas, beds_min=None, beds_max=None) -> dict | None:
    """Median registered rent for a place. None when the question did not name a place."""
    official = _official_areas(areas)
    if not official:
        return None
    clauses = [
        "deleted_at is null",
        "annual_amount > 0",
        "contract_start_date >= date '2025-09-01'",
        "(upper(area_name_prop_ai) = any(%s) or upper(master_project_prop_ai) = any(%s))",
    ]
    params: list = [official, official]
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
        "note": "Registered rents from the new data since September 2025. Listing cards remain advertised asking rents.",
    }


def sale_summary(communities) -> dict | None:
    """Project count and median recent sale for a place in the new data."""
    official = _official_areas(communities)
    names = list(dict.fromkeys(official + [name.upper() for name in _expanded(communities)]))
    if not names:
        return None
    try:
        projects = fetch(
            """
            select count(*)::int as n
            from propai_projects
            where coalesce(name, '') <> ''
              and (upper(master_project) = any(%s) or upper(area) = any(%s))
            """,
            (names, names),
        )[0]["n"]
        sales = fetch(
            """
            select count(*)::int as n,
                   percentile_cont(0.5) within group (order by sale_price)::bigint as median
            from propai_sales
            where deleted_at is null and sale_price > 0
              and instance_date >= current_date - interval '24 months'
              and (upper(area_name_prop_ai) = any(%s) or upper(master_project_prop_ai) = any(%s))
            """,
            (names, names),
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
        "note": "Recorded sales from the last 24 months in the new data. Prefer this price over an older starting price.",
    }
