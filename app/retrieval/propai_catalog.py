"""
Read the new Prop-AI tables in the same shape as the old project rows.

A question is matched to a place with the alias list, then the place embedding.
Projects, status, and a recent sale price are loaded with SQL and returned as the
fields the current cards already show. If nothing matches, the caller keeps the old data.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from datetime import datetime

from app.database.postgres import fetch
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
    """Catalog rows have no brochures; the listings projects table does. Match by name ("Sobha One")."""
    names = [str(project.get("name") or "").strip().lower() for project in projects if project.get("name")]
    if not names:
        return
    try:
        rows = fetch(
            """
            select distinct on (lower(btrim(name))) lower(btrim(name)) as name, brochure_url
            from projects
            where coalesce(brochure_url, '') <> '' and lower(btrim(name)) = any(%s)
            order by lower(btrim(name)), id
            """,
            (names,),
        )
    except Exception:
        return
    found = {row["name"]: row["brochure_url"] for row in rows}
    for project in projects:
        url = found.get(str(project.get("name") or "").strip().lower())
        if url and not project.get("brochure_url"):
            project["brochure_url"] = url


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
    _attach_brochures(projects)
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
            return ALIASES.get(label) or label.title()
    return None


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
    if communities:
        projects = drop_numbered_siblings(projects, communities)
        total = len(projects)
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
        handover = project.get("handover_time")
        if sort == "priciest":
            return (price <= 0, -(price or 0))
        if sort == "earliest_handover":
            return (handover is None, handover or datetime(2099, 12, 31))
        if sort == "latest_handover":
            return (handover is None, handover or datetime(1970, 1, 1))
        return (price <= 0, price or 10**18)

    if sort == "latest_handover":
        projects.sort(
            key=lambda project: (
                project.get("handover_time") is not None,
                project.get("handover_time") or datetime(1970, 1, 1),
            ),
            reverse=True,
        )
    else:
        projects.sort(key=rank)
    projects = projects[:limit]
    _attach_photos(projects)
    _attach_brochures(projects)
    for project in projects:
        # The app's project page is keyed by the catalog key; the agent id is a one-way hash of it.
        project["key"] = project["id"]
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


def latest_sales(place=None, limit: int = 8, offset: int = 0) -> dict:
    """Newest recorded sales, optionally limited to one Dubai area or project name."""
    cap = max(1, min(int(limit or 8), 15))
    start = max(0, int(offset or 0))
    parts = [
        "deleted_at is null",
        "sale_price between 10000 and 2000000000",
        "instance_date is not null",
        "coalesce(project_name_prop_ai, '') <> ''",
        "(procedure_area_sqft is null or procedure_area_sqft between 50 and 100000)",
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
        "note": "Registered rents from the new data since September 2025. Listing cards remain advertised asking rents.",
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
            where deleted_at is null and sale_price > 0
              and instance_date >= current_date - interval '24 months'
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
        "note": "Recorded sales from the last 24 months in the new data. Prefer this price over an older starting price.",
    }
