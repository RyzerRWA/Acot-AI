"""
Shared project recommendation: score Prop-AI / project rows and format a clear pick.

Used by named compare, which-of-these shortlists, and open invest/buy answers.
Reasons stay positive and cite price, size, sales/transactions, and rentals.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_READY = re.compile(r"\b(?:ready|completed|on[\s-]?sale|available)\b", re.I)
_INACTIVE = re.compile(r"\b(?:cancelled|canceled|hold|inactive|sold[\s-]?out)\b", re.I)
_BEDS = re.compile(r"\b(\d+)\s*[- ]?\s*(?:bed(?:room)?s?|br)\b", re.I)

_WHY_LABELS = {
    "lower price": "Attractive price among these options",
    "budget fit": "Close to the budget you gave",
    "more space": "Larger recorded size",
    "more bedrooms": "Wider bedroom range",
    "bedroom fit": "Strong bedroom fit for your request",
    "earlier handover": "Earlier handover timing",
    "latest recorded sale": "Most recent recorded sale among these options",
    "readiness": "Strong readiness / availability status",
    "more recent sales activity": "Strong recent sales and transaction activity",
    "active rental market": "Active rental market for this project",
}

# Recommend quotes the last year of DLD sales, not a 20-year-old completion date.
_RECENT_SALES_INTERVAL = "12 months"
_BUDGET_FLOOR = 0.6
RECOMMEND_POOL = 40
RECOMMEND_SHOW = 6

# Phase 4: goal → factor weights. Higher = more influence on the winner.
# Keys match factor names produced in rank_projects().
_GOAL_WEIGHTS: dict[str, dict[str, float]] = {
    "budget": {
        "budget fit": 4.0,
        "lower price": 3.0,
        "bedroom fit": 1.5,
        "more bedrooms": 1.0,
        "more space": 1.0,
        "readiness": 1.0,
        "earlier handover": 0.8,
        "latest recorded sale": 0.8,
        "more recent sales activity": 0.6,
        "active rental market": 0.4,
    },
    "afford": {
        "budget fit": 4.0,
        "lower price": 3.5,
        "bedroom fit": 1.5,
        "more bedrooms": 1.0,
        "more space": 1.0,
        "readiness": 1.2,
        "earlier handover": 0.8,
        "latest recorded sale": 0.8,
        "more recent sales activity": 0.5,
        "active rental market": 0.4,
    },
    "family": {
        "bedroom fit": 4.0,
        "more bedrooms": 3.0,
        "more space": 3.0,
        "readiness": 2.5,
        "earlier handover": 1.5,
        "budget fit": 2.0,
        "lower price": 1.0,
        "latest recorded sale": 1.0,
        "more recent sales activity": 0.8,
        "active rental market": 0.4,
    },
    "invest": {
        # Market evidence first — handover alone must never win an invest pick.
        "more recent sales activity": 4.5,
        "active rental market": 4.0,
        "latest recorded sale": 2.5,
        "lower price": 2.0,
        "budget fit": 1.5,
        "more space": 1.0,
        "readiness": 0.6,
        "earlier handover": 0.2,
        "bedroom fit": 0.5,
        "more bedrooms": 0.4,
    },
    "ready": {
        "readiness": 4.0,
        "earlier handover": 3.5,
        "latest recorded sale": 2.5,
        "bedroom fit": 1.5,
        "more bedrooms": 1.0,
        "more space": 1.0,
        "budget fit": 1.5,
        "lower price": 1.0,
        "more recent sales activity": 1.0,
        "active rental market": 0.6,
    },
    "compare": {
        "budget fit": 2.0,
        "lower price": 1.5,
        "bedroom fit": 1.5,
        "more bedrooms": 1.2,
        "more space": 1.5,
        "readiness": 1.5,
        "earlier handover": 1.2,
        "latest recorded sale": 1.5,
        "more recent sales activity": 1.5,
        "active rental market": 1.5,
    },
    "open_buy": {
        "budget fit": 3.0,
        "lower price": 1.5,
        "bedroom fit": 1.5,
        "more bedrooms": 1.2,
        "more space": 1.2,
        "readiness": 1.5,
        "earlier handover": 1.0,
        "latest recorded sale": 1.2,
        "more recent sales activity": 1.5,
        "active rental market": 1.2,
    },
}


def _weights_for_goal(goal: str | None) -> dict[str, float]:
    key = (goal or "open_buy").strip().lower()
    if key not in _GOAL_WEIGHTS:
        key = "open_buy"
    return _GOAL_WEIGHTS[key]


def _factor_weight(goal: str | None, factor: str) -> float:
    weights = _weights_for_goal(goal)
    if factor in weights:
        return float(weights[factor])
    # Legacy default: budget fit was triple-weighted when no goal profile hit.
    if factor == "budget fit" and not goal:
        return 3.0
    return 1.0


def wanted_bedrooms(question: str | None) -> int | None:
    match = _BEDS.search(question or "")
    return int(match.group(1)) if match else None


def _num(*values) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:  # not NaN
            return number
    return None


def _handover_stamp(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # Unix seconds, not a calendar year. Windows rejects out-of-range stamps.
        if number < 0 or number > 4_102_444_800:  # 1970–2100
            return None
        return number
    year = getattr(value, "year", None)
    if year is not None and (year < 1970 or year > 2100):
        return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except (OSError, OverflowError, ValueError, TypeError):
            pass
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.year < 1970 or parsed.year > 2100:
            return None
        return parsed.timestamp()
    except (OSError, OverflowError, ValueError, TypeError):
        return None


def _is_future_stamp(stamp: float | None) -> bool:
    if stamp is None:
        return False
    try:
        return datetime.fromtimestamp(stamp) > datetime.now()
    except (OSError, OverflowError, ValueError, TypeError):
        return False


def _month_year(stamp: float | None) -> str | None:
    if stamp is None:
        return None
    try:
        return datetime.fromtimestamp(stamp).strftime("%b %Y")
    except (OSError, OverflowError, ValueError, TypeError):
        return None


def _status_rank(status) -> int | None:
    text = str(status or "").strip()
    if not text:
        return None
    if _INACTIVE.search(text):
        return 0
    if _READY.search(text):
        return 2
    return 1


def features(row: dict) -> dict[str, Any]:
    """Normalize one project/listing-like row for scoring."""
    price = _num(row.get("price"), row.get("price_aed"), row.get("starting_price_aed"))
    if price is not None and price <= 0:
        price = None
    bed_min = _num(row.get("bedroom_min"), row.get("beds"))
    bed_max = _num(row.get("bedroom_max"), bed_min)
    if bed_min is not None and bed_max is not None and bed_min == bed_max:
        beds = bed_min
    elif bed_min is not None and bed_max is not None:
        beds = (bed_min + bed_max) / 2
    else:
        beds = bed_min if bed_min is not None else bed_max
    size = _num(row.get("size"), row.get("size_sqft"), row.get("size_min"))
    size_max = _num(row.get("size_max"))
    if size is None and size_max is not None:
        size = size_max
    elif size is not None and size_max is not None:
        size = (size + size_max) / 2
    return {
        "id": row.get("id"),
        "name": row.get("name") or row.get("title") or "Option",
        "price": price,
        "beds": beds,
        "bed_min": bed_min,
        "bed_max": bed_max,
        "size": size,
        "handover": _handover_stamp(row.get("handover_time") or row.get("handover")),
        "last_sale": _handover_stamp(row.get("last_sold_on") or row.get("last_sale")),
        "status_rank": _status_rank(row.get("project_status") or row.get("status")),
        "sales_count": _num(row.get("sales_count"), row.get("recent_sales_count")),
        "rent_count": _num(row.get("rent_count"), row.get("rental_count")),
        "median_rent": _num(row.get("median_rent"), row.get("median_annual_rent_aed")),
        "price_kind": row.get("price_kind"),
        "row": row,
    }


def attach_market_activity(projects: list[dict]) -> None:
    """Fill sales and rental activity from Prop-AI when missing. Best-effort."""
    from concurrent.futures import ThreadPoolExecutor

    names = []
    for project in projects:
        name = str(project.get("name") or "").strip()
        if len(name) >= 3:
            names.append(name.upper())
    names = list(dict.fromkeys(names))
    if not names:
        return
    try:
        from app.database.postgres import fetch
    except Exception:
        return

    need_sales = any(
        project.get("sales_count") is None and project.get("recent_sales_count") is None
        for project in projects
    )
    need_size = any(
        _num(project.get("size"), project.get("size_sqft"), project.get("size_min"), project.get("size_max")) is None
        for project in projects
    )
    need_sold = any(project.get("last_sold_on") is None and project.get("last_sale") is None for project in projects)
    need_rent = any(
        project.get("rent_count") is None and project.get("rental_count") is None
        for project in projects
    )

    def _sales_rows():
        if not (need_sales or need_size or need_sold):
            return []
        try:
            return fetch(
                f"""
                select upper(project_name_prop_ai) as name,
                       count(*)::int as n,
                       max(instance_date) as last_sale,
                       percentile_cont(0.5) within group (
                           order by procedure_area_sqft
                       ) filter (where procedure_area_sqft > 0) as median_size
                from propai_sales
                where deleted_at is null and sale_price > 0
                  and instance_date >= current_date - interval '{_RECENT_SALES_INTERVAL}'
                  and upper(project_name_prop_ai) = any(%s)
                group by 1
                """,
                (names,),
            )
        except Exception:
            return []

    def _rent_rows():
        if not need_rent:
            return []
        try:
            return fetch(
                """
                select upper(project_name_prop_ai) as name,
                       count(*)::int as n,
                       percentile_cont(0.5) within group (order by annual_amount)::bigint as median_rent
                from propai_rents
                where deleted_at is null and annual_amount > 0
                  and contract_start_date >= current_date - interval '18 months'
                  and upper(project_name_prop_ai) = any(%s)
                group by 1
                """,
                (names,),
            )
        except Exception:
            return []

    # Parallel sales + rent aggregates (was sequential ~2× DB RTT).
    if (need_sales or need_size or need_sold) and need_rent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sales_future = pool.submit(_sales_rows)
            rent_future = pool.submit(_rent_rows)
            rows = sales_future.result() or []
            rents = rent_future.result() or []
    else:
        rows = _sales_rows() or []
        rents = _rent_rows() or []

    if rows:
        by_name = {str(row["name"]): row for row in rows if row.get("name")}
        for project in projects:
            key = str(project.get("name") or "").strip().upper()
            row = by_name.get(key)
            if not row:
                continue
            if project.get("sales_count") is None and project.get("recent_sales_count") is None:
                project["sales_count"] = int(row["n"] or 0)
            if project.get("last_sold_on") is None and project.get("last_sale") is None and row.get("last_sale"):
                project["last_sold_on"] = row["last_sale"]
            if need_size and _num(project.get("size"), project.get("size_sqft"), project.get("size_min")) is None:
                if row.get("median_size"):
                    project["size_sqft"] = int(round(float(row["median_size"])))

    if not rents:
        return
    by_rent = {str(row["name"]): row for row in rents if row.get("name")}
    for project in projects:
        key = str(project.get("name") or "").strip().upper()
        row = by_rent.get(key)
        if not row:
            continue
        project["rent_count"] = int(row["n"] or 0)
        if row.get("median_rent") is not None:
            project["median_rent"] = int(row["median_rent"])


def attach_sale_counts(projects: list[dict]) -> None:
    """Back-compat wrapper."""
    attach_market_activity(projects)


def _rank_key(values: list[float | None], lower_is_better: bool) -> list[float | None]:
    known = [value for value in values if value is not None]
    if len(known) < 2 or len(set(known)) < 2:
        return [None] * len(values)
    ordered = sorted(known, reverse=not lower_is_better)
    points = {value: (len(ordered) - 1 - index) / (len(ordered) - 1) for index, value in enumerate(ordered)}
    return [None if value is None else points[value] for value in values]


def _budget_fit_scores(prices: list[float | None], budget: int | None) -> list[float | None]:
    """1.0 at the budget, 0 at 60% of it, weak above it."""
    if not budget or budget <= 0:
        return [None] * len(prices)
    span = 1.0 - _BUDGET_FLOOR
    scores = []
    for price in prices:
        if price is None or price <= 0:
            scores.append(None)
            continue
        ratio = price / budget
        if ratio <= 1:
            scores.append(max(0.0, min(1.0, (ratio - _BUDGET_FLOOR) / span)))
        else:
            scores.append(max(0.0, 1.0 - (ratio - 1.0) * 2))
    if sum(1 for score in scores if score is not None) < 1:
        return [None] * len(prices)
    return scores


def _bed_fit_scores(items: list[dict], wanted: int | None) -> list[float | None]:
    """Wanted bedroom count inside the project's min–max is a full match."""
    if wanted is None:
        return _rank_key([item.get("beds") for item in items], lower_is_better=False)
    scores = []
    for item in items:
        lo = item.get("bed_min")
        hi = item.get("bed_max")
        if lo is None and hi is None:
            scores.append(None)
            continue
        lo = lo if lo is not None else hi
        hi = hi if hi is not None else lo
        if lo <= wanted <= hi:
            scores.append(1.0)
        else:
            gap = min(abs(wanted - lo), abs(wanted - hi))
            scores.append(max(0.0, 1.0 - gap / max(wanted, 1)))
    known = [score for score in scores if score is not None]
    if len(known) < 2 or len(set(known)) < 2:
        return [None] * len(items)
    return scores


def _factor_phrase(names: list[str] | None) -> str:
    labels = [name for name in (names or []) if name]
    if not labels:
        return "the recorded details"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _sale_project_rows(rows: list[dict], place: str, bedrooms: int | None) -> list[dict]:
    """Turn grouped sale rows into project cards, with that bedroom's median price."""
    if not rows:
        return []
    names = [str(row.get("name") or "").strip() for row in rows if row.get("name")]
    catalog: dict[str, dict] = {}
    try:
        from app.database.postgres import fetch
        from app.retrieval.propai_catalog import _agent_id, _handover

        found = fetch(
            """
            select distinct on (upper(name))
                   key, name, developer, area, master_project, offplan_status, end_date
            from propai_projects
            where upper(name) = any(%s)
            order by upper(name), (developer is not null) desc, name
            """,
            ([name.upper() for name in names],),
        )
        catalog = {str(row["name"]).upper(): row for row in found if row.get("name")}
    except Exception:
        _agent_id = None
        _handover = None
    projects = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name or row.get("price") is None:
            continue
        source = catalog.get(name.upper()) or {}
        key = source.get("key")
        beds = None if bedrooms is None else int(bedrooms)
        projects.append({
            "id": _agent_id(str(key)) if key and _agent_id else name,
            "key": key,
            "name": source.get("name") or name.title(),
            "developer_name": source.get("developer"),
            "community": source.get("master_project") or source.get("area") or place,
            "sub_community": source.get("area"),
            "city": "Dubai",
            "price": int(round(float(row["price"]))),
            "price_kind": "recent_sale",
            "bedroom_min": beds,
            "bedroom_max": beds,
            "sales_count": int(row.get("sales") or 0),
            "status": source.get("offplan_status"),
            "project_status": source.get("offplan_status"),
            "handover_time": _handover(source.get("end_date")) if _handover else None,
            "last_sale": row.get("last_sale"),
            "last_sold_on": (
                row["last_sale"].isoformat()[:10]
                if hasattr(row.get("last_sale"), "isoformat")
                else (str(row["last_sale"])[:10] if row.get("last_sale") else None)
            ),
            "source": "propai",
        })
    from app.answers.grounding import ground_project_rows

    return ground_project_rows(projects)


def fetch_recommend_projects(
    place: str,
    bedrooms: int | None = None,
    budget: int | None = None,
    limit: int = RECOMMEND_POOL,
    property_type: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Projects with recorded sales for this place and bedroom, near the budget when given.

    The first list is inside about 60–100% of the budget (about 30–50 candidates).
    The second is the closest recorded sales when that band is empty.
    """
    from app.answers.predict import _sales_place_filter
    from app.database.postgres import fetch

    lock, lock_params = _sales_place_filter(place)
    if lock == "false":
        return [], []
    where = [
        "deleted_at is null",
        "coalesce(outlier_flag, '') <> 'True'",
        "sale_price > 0",
        f"instance_date >= current_date - interval '{_RECENT_SALES_INTERVAL}'",
        "coalesce(project_name_prop_ai, '') <> ''",
        lock,
    ]
    kind = (property_type or "").strip().lower()
    if kind == "villa":
        where.append("property_type_en = 'Villa'")
    elif kind == "townhouse":
        where.append(
            "("
            "property_type_en ilike 'townhouse%%'"
            " or lower(coalesce(property_sub_type_en, '')) like '%%townhouse%%'"
            ")"
        )
    elif kind == "apartment":
        where.append(
            "("
            "property_type_en = 'Unit' and property_sub_type_en = 'Flat'"
            ")"
        )
    params: list[Any] = list(lock_params)
    if bedrooms is not None:
        where.append("bedrooms between %s - 0.2 and %s + 0.2")
        params.extend([int(bedrooms), int(bedrooms)])
    base = f"""
        select project_name_prop_ai as name,
               count(*)::int as sales,
               percentile_cont(0.5) within group (order by sale_price)::float8 as price,
               max(instance_date) as last_sale
        from public.propai_sales
        where {' and '.join(where)}
        group by 1
    """
    target = int(budget or 0)
    try:
        if budget:
            band = fetch(
                base + """
                having percentile_cont(0.5) within group (order by sale_price) between %s and %s
                order by abs(percentile_cont(0.5) within group (order by sale_price) - %s),
                         max(instance_date) desc nulls last,
                         count(*) desc
                limit %s
                """,
                tuple([*params, int(budget * _BUDGET_FLOOR), int(budget), target, int(limit)]),
            )
            matches = _sale_project_rows(band or [], place, bedrooms)
            if matches:
                return matches, []
        closest = fetch(
            base + """
            order by abs(percentile_cont(0.5) within group (order by sale_price) - %s),
                     max(instance_date) desc nulls last,
                     count(*) desc
            limit %s
            """,
            tuple([*params, target, int(limit)]),
        )
    except Exception:
        return [], []
    rows = _sale_project_rows(closest or [], place, bedrooms)
    if budget:
        return [], rows
    return rows, []


def fetch_villa_projects_under_budget(
    budget: int | None = None,
    limit: int = RECOMMEND_SHOW,
    skip_place: str | None = None,
    property_type: str = "villa",
    bedrooms: int | None = None,
) -> list[dict]:
    """Villa / townhouse projects from recorded sales (not catalog-only towers)."""
    from app.database.postgres import fetch

    kind = (property_type or "villa").strip().lower()
    if kind == "townhouse":
        type_sql = (
            "("
            "property_type_en ilike 'townhouse%%'"
            " or lower(coalesce(property_sub_type_en, '')) like '%%townhouse%%'"
            ")"
        )
    else:
        type_sql = "property_type_en = 'Villa'"
    bed_sql = ""
    params: list[Any] = []
    if bedrooms is not None:
        bed_sql = " and bedrooms = %s"
        params.append(int(bedrooms))
    # Require real transaction volume so handover-only catalog rows never win.
    having_parts = ["count(*) >= 3"]
    if budget:
        having_parts.append(
            "percentile_cont(0.5) within group (order by sale_price) "
            "between %s and %s"
        )
        params.extend([int(budget * _BUDGET_FLOOR), int(budget)])
    having = "having " + " and ".join(having_parts)
    params.append(int(limit))
    try:
        rows = fetch(
            f"""
            select project_name_prop_ai as name,
                   count(*)::int as sales,
                   percentile_cont(0.5) within group (order by sale_price)::float8 as price,
                   max(instance_date) as last_sale
            from public.propai_sales
            where deleted_at is null
              and coalesce(outlier_flag, '') <> 'True'
              and sale_price > 0
              and instance_date >= current_date - interval '{_RECENT_SALES_INTERVAL}'
              and coalesce(project_name_prop_ai, '') <> ''
              and {type_sql}
              {bed_sql}
            group by 1
            {having}
            order by max(instance_date) desc nulls last, count(*) desc
            limit %s
            """,
            tuple(params),
        )
    except Exception:
        return []
    projects = _sale_project_rows(rows or [], "Dubai", bedrooms)
    skip = (skip_place or "").strip().lower()
    if skip:
        projects = [
            row for row in projects
            if skip not in str(row.get("community") or "").lower()
            and skip not in str(row.get("name") or "").lower()
        ]
    return projects


def fetch_studio_projects_under_budget(
    budget: int | None = None,
    limit: int = RECOMMEND_SHOW,
) -> list[dict]:
    """Studio apartment projects backed by recorded sales under a budget."""
    from app.database.postgres import fetch

    params: list[Any] = []
    having_parts = ["count(*) >= 3"]
    if budget:
        having_parts.append(
            "percentile_cont(0.5) within group (order by sale_price) "
            "between %s and %s"
        )
        params.extend([int(budget * _BUDGET_FLOOR), int(budget)])
    having = "having " + " and ".join(having_parts)
    params.append(int(limit))
    try:
        rows = fetch(
            f"""
            select project_name_prop_ai as name,
                   count(*)::int as sales,
                   percentile_cont(0.5) within group (order by sale_price)::float8 as price,
                   max(instance_date) as last_sale
            from public.propai_sales
            where deleted_at is null
              and coalesce(outlier_flag, '') <> 'True'
              and sale_price > 0
              and instance_date >= current_date - interval '{_RECENT_SALES_INTERVAL}'
              and coalesce(project_name_prop_ai, '') <> ''
              and bedrooms = 0
              and (
                property_type_en ilike '%%apartment%%'
                or property_type_en ilike '%%flat%%'
                or property_type_en ilike '%%unit%%'
                or property_type_en is null
              )
            group by 1
            {having}
            order by max(instance_date) desc nulls last, count(*) desc
            limit %s
            """,
            tuple(params),
        )
    except Exception:
        return []
    return _sale_project_rows(rows or [], "Dubai", 0)


def _why_line(factor: str) -> str:
    return _WHY_LABELS.get(factor, factor.replace("_", " ").capitalize())


def _sort_factors_for_goal(factors: list[str], goal: str | None) -> list[str]:
    """Put the goal's primary factors first in Why / used_factors."""
    return sorted(
        factors,
        key=lambda name: (-_factor_weight(goal, name), name),
    )


_INVEST_EVIDENCE_FACTORS = frozenset({
    "more recent sales activity",
    "active rental market",
    "latest recorded sale",
    "lower price",
    "budget fit",
})


def _has_transaction_evidence(item: dict[str, Any] | None) -> bool:
    """True only with recorded sales or rents — not handover-only catalog rows."""
    item = item or {}
    try:
        if item.get("sales_count") is not None and int(item["sales_count"]) > 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if item.get("recent_sales_count") is not None and int(item["recent_sales_count"]) > 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if item.get("rent_count") is not None and int(item["rent_count"]) > 0:
            return True
    except (TypeError, ValueError):
        pass
    if item.get("last_sale") or item.get("last_sold_on"):
        return True
    # Sale-backed price from propai_sales (never catalog starting price alone).
    try:
        if (
            str(item.get("price_kind") or "") == "recent_sale"
            and item.get("price") is not None
            and float(item["price"]) > 0
        ):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _has_market_evidence(item: dict[str, Any] | None) -> bool:
    """True only with recorded sales/rent evidence — never catalog price alone."""
    return _has_transaction_evidence(item)


def evidence_projects(
    projects: list[dict] | None,
    *,
    attach_sales: bool = True,
) -> list[dict]:
    """Keep only rows with recorded sales or rents (market-analyst choke-point)."""
    rows = [project for project in (projects or []) if isinstance(project, dict)]
    if not rows:
        return []
    if attach_sales:
        attach_market_activity(rows)
    return [row for row in rows if _has_transaction_evidence(row)]


def rank_projects(
    projects: list[dict],
    *,
    wanted_beds: int | None = None,
    budget_aed: int | None = None,
    attach_sales: bool = True,
    goal: str | None = None,
    require_evidence: bool = True,
) -> dict[str, Any]:
    """Rank projects with goal-weighted factors (Phase 4).

    Goals: family | budget | afford | invest | ready | compare | open_buy

    By default only transaction-backed rows are ranked — handover-only catalog
    towers never become Acot Recommend winners.
    """
    rows = [project for project in projects if isinstance(project, dict)]
    if attach_sales:
        attach_market_activity(rows)
    if require_evidence:
        rows = [row for row in rows if _has_transaction_evidence(row)]
    items = [features(row) for row in rows]
    for item in items:
        item["wanted_beds"] = wanted_beds
        item["goal"] = goal
    if not items:
        return {
            "ranked": [],
            "can_rank": False,
            "goal": goal,
            "used_factors": [],
            "winner": None,
            "alternatives": [],
            "why": [],
            "notes": [
                "Latest recorded sales and rent activity are thin for these options — "
                "share a community or budget and I will match stronger transaction-backed picks."
            ],
            "recommendation": {
                "name": None,
                "id": None,
                "why": ["Need recorded sales or rent activity before recommending."],
                "notes": [
                    "Latest recorded sales and rent activity are thin for these options — "
                    "share a community or budget and I will match stronger transaction-backed picks."
                ],
            },
        }
    if len(items) < 2:
        winner = items[0]
        return {
            "ranked": items,
            "can_rank": False,
            "goal": goal,
            "used_factors": [],
            "winner": winner,
            "alternatives": [],
            "why": [],
            "notes": ["Limited data: share one more project to compare side by side."],
            "recommendation": {
                "name": winner["name"],
                "id": winner.get("id"),
                "why": ["A solid option from the recorded project data."],
                "notes": ["Limited data: share one more project to compare side by side."],
            },
        }

    prices = [item["price"] for item in items]
    budget_pts = _budget_fit_scores(prices, budget_aed)
    price_pts = (
        budget_pts
        if budget_aed and any(point is not None for point in budget_pts)
        else _rank_key(prices, lower_is_better=True)
    )
    price_factor = "budget fit" if budget_aed and price_pts is budget_pts else "lower price"
    size_pts = _rank_key([item["size"] for item in items], lower_is_better=False)
    bed_pts = _bed_fit_scores(items, wanted_beds)
    # Past completion dates are old stock, not a reason to rank a tower first.
    future_handovers = [
        item["handover"] if _is_future_stamp(item.get("handover")) else None
        for item in items
    ]
    hand_pts = _rank_key(future_handovers, lower_is_better=True)
    last_sale_pts = _rank_key([item.get("last_sale") for item in items], lower_is_better=False)
    status_pts = _rank_key([item["status_rank"] for item in items], lower_is_better=False)
    sales_pts = _rank_key([item["sales_count"] for item in items], lower_is_better=False)
    rent_pts = _rank_key([item["rent_count"] for item in items], lower_is_better=False)

    factor_sets = {
        price_factor: price_pts,
        "more space": size_pts,
        "bedroom fit" if wanted_beds is not None else "more bedrooms": bed_pts,
        "earlier handover": hand_pts,
        "latest recorded sale": last_sale_pts,
        "readiness": status_pts,
        "more recent sales activity": sales_pts,
        "active rental market": rent_pts,
    }
    active_factors = _sort_factors_for_goal(
        [
            name for name, points in factor_sets.items()
            if any(point is not None for point in points)
        ],
        goal,
    )
    can_rank = len(active_factors) >= 2
    # Full denominator so thin rows (handover only) cannot inflate their average.
    full_weight = sum(_factor_weight(goal, name) for name in active_factors) or 1.0

    for index, item in enumerate(items):
        weighted = 0.0
        why = []
        evidence = 0
        for name in active_factors:
            points = factor_sets[name]
            point = points[index]
            weight = _factor_weight(goal, name)
            if point is None:
                point = 0.0
            else:
                if name in _INVEST_EVIDENCE_FACTORS and point > 0:
                    evidence += 1
                if point >= 0.75:
                    why.append(name)
            weighted += point * weight
        item["score"] = weighted / full_weight
        item["why"] = _sort_factors_for_goal(why, goal)
        item["factor_count"] = len(why)
        item["market_evidence"] = _has_market_evidence(item) or evidence > 0

    # Never let a no-sales/no-rent row outrank market-backed ones (any goal).
    for item in items:
        if not item.get("market_evidence"):
            item["score"] = float(item.get("score") or 0) * 0.15

    items.sort(
        key=lambda item: (
            -int(bool(item.get("market_evidence"))),
            -(item.get("score") or 0),
            item["name"],
        )
    )
    if not can_rank:
        limited = (
            f"Limited data: ranked on {_factor_phrase(active_factors)}."
            if active_factors else
            "Limited data: too few recorded factors to rank these options."
        )
        return {
            "ranked": items,
            "can_rank": False,
            "goal": goal,
            "used_factors": active_factors,
            "winner": items[0] if items else None,
            "alternatives": items[1:4],
            "why": active_factors,
            "notes": [limited],
            "recommendation": {
                "name": items[0]["name"] if items else None,
                "id": items[0].get("id") if items else None,
                "why": active_factors or ["A practical starting point from the projects in view."],
                "notes": [limited],
                "alternatives": [
                    {"name": item["name"], "id": item.get("id")}
                    for item in items[1:4]
                ],
            },
        }

    best = items[0]
    # Never crown a row with no sales/rent when a backed option exists.
    if not best.get("market_evidence"):
        backed = next((item for item in items if item.get("market_evidence")), None)
        if backed is not None:
            best = backed
            items = [best] + [item for item in items if item is not best]

    tied = [
        item for item in items
        if item["score"] == best["score"]
        and bool(item.get("market_evidence")) == bool(best.get("market_evidence"))
    ]
    if len(tied) > 1:
        names = [item["name"] for item in tied]
        return {
            "ranked": items,
            "can_rank": True,
            "goal": goal,
            "used_factors": active_factors,
            "winner": None,
            "alternatives": tied,
            "why": active_factors,
            "notes": [
                f"{names[0]} and {names[1]} both look strong — choose by the layout and community feel you prefer."
            ],
            "recommendation": {
                "name": None,
                "tied": names,
                "why": [
                    f"Both lead on {_factor_phrase(active_factors)}."
                ],
                "notes": [
                    f"{names[0]} and {names[1]} both look strong — choose by the layout and community feel you prefer."
                ],
            },
        }

    why = list(best.get("why") or active_factors[:3])
    if (goal or "").lower() == "invest":
        market_why = [name for name in why if name in _INVEST_EVIDENCE_FACTORS]
        if market_why:
            why = market_why + [name for name in why if name not in market_why]
        # Do not claim invest strength from handover-only Why lines.
        why = [name for name in why if name != "earlier handover" or market_why]
        if not why and best.get("market_evidence"):
            why = [
                name for name in active_factors
                if name in _INVEST_EVIDENCE_FACTORS
            ][:3] or why
    notes = []
    alts = [
        item for item in items
        if item is not best and (
            (goal or "").lower() not in {"invest", "open_buy"}
            or item.get("market_evidence")
        )
    ][:3]
    if not alts:
        alts = [item for item in items if item is not best][:3]
    if alts:
        notes.append(
            "Also worth reviewing: "
            + ", ".join(item["name"] for item in alts)
            + "."
        )
    if (goal or "").lower() == "invest" and not best.get("market_evidence"):
        notes.append(
            "Limited invest data on these projects — only handover timing was available, not recorded sales or prices."
        )

    return {
        "ranked": items,
        "can_rank": True,
        "goal": goal,
        "used_factors": active_factors,
        "winner": best,
        "alternatives": alts,
        "why": why,
        "notes": notes,
        "recommendation": {
            "name": best["name"],
            "id": best.get("id"),
            "why": why,
            "notes": notes,
            "alternatives": [
                {"name": item["name"], "id": item.get("id")}
                for item in alts
            ],
        },
    }


def _detail_why(
    item: dict | None,
    factors: list[str],
    *,
    alternatives: list[dict] | None = None,
) -> list[str]:
    """Concrete, comparative why lines using the winner vs the next option."""
    item = item or {}
    alt = None
    for candidate in alternatives or []:
        if isinstance(candidate, dict) and candidate.get("name"):
            # Prefer a full ranked row with numbers when available.
            if any(
                candidate.get(key) is not None
                for key in ("price", "size", "beds", "sales_count", "rent_count", "handover")
            ):
                alt = candidate
                break
            alt = alt or candidate
    lines = []
    for factor in factors[:5]:
        if factor == "budget fit" and item.get("price"):
            if alt and alt.get("price") is not None:
                lines.append(
                    f"Closer budget fit at AED {int(item['price']):,} vs "
                    f"{alt['name']} at AED {int(alt['price']):,}"
                )
            else:
                lines.append(f"Close to your budget at AED {int(item['price']):,}")
        elif factor == "lower price" and item.get("price"):
            if alt and alt.get("price") is not None:
                lines.append(
                    f"Lower recorded price at AED {int(item['price']):,} than "
                    f"{alt['name']} at AED {int(alt['price']):,}"
                )
            else:
                lines.append(f"Attractive recent price at AED {int(item['price']):,}")
        elif factor == "more space" and item.get("size"):
            if alt and alt.get("size"):
                lines.append(
                    f"Larger recorded size around {int(item['size']):,} sqft vs "
                    f"{int(alt['size']):,} sqft for {alt['name']}"
                )
            else:
                lines.append(f"Larger recorded size around {int(item['size']):,} sqft")
        elif factor in {"more bedrooms", "bedroom fit"}:
            wanted = item.get("wanted_beds")
            if wanted == 0:
                lines.append("Fits the studio request")
            elif wanted == 1:
                lines.append("Fits the 1-bedroom request")
            elif wanted:
                lines.append(f"Fits the {int(wanted)}-bedroom request")
            elif item.get("beds") is not None:
                if alt and alt.get("beds") is not None:
                    lines.append(
                        f"Bedroom range includes {item['beds']:g} bedrooms vs "
                        f"{alt['beds']:g} for {alt['name']}"
                    )
                else:
                    lines.append(f"Bedroom range includes {item['beds']:g} bedrooms")
            else:
                lines.append(_why_line(factor))
        elif factor == "earlier handover" and item.get("handover") and _is_future_stamp(item.get("handover")):
            stamp = _month_year(item["handover"]) or "an earlier date"
            if alt and alt.get("handover") and _is_future_stamp(alt.get("handover")):
                other = _month_year(alt["handover"]) or "a later date"
                lines.append(
                    f"Earlier handover ({stamp}) than {alt['name']} ({other})"
                )
            else:
                lines.append(f"Earlier handover timing ({stamp})")
        elif factor == "latest recorded sale" and item.get("last_sale"):
            stamp = _month_year(item["last_sale"]) or "this year"
            if alt and alt.get("last_sale"):
                other = _month_year(alt["last_sale"]) or "an earlier sale"
                lines.append(
                    f"Latest recorded sale {stamp} vs {other} for {alt['name']}"
                )
            else:
                lines.append(f"Latest recorded sale {stamp}")
        elif factor == "more recent sales activity" and item.get("sales_count"):
            if alt and alt.get("sales_count") is not None:
                lines.append(
                    f"Stronger recent sales activity ({int(item['sales_count'])} sales) "
                    f"than {alt['name']} ({int(alt['sales_count'])} sales)"
                )
            else:
                lines.append(
                    f"Strong sales and transaction activity "
                    f"({int(item['sales_count'])} recent sales)"
                )
        elif factor == "active rental market" and item.get("rent_count"):
            if alt and alt.get("rent_count") is not None:
                lines.append(
                    f"More active rental market ({int(item['rent_count'])} contracts) "
                    f"than {alt['name']} ({int(alt['rent_count'])} contracts)"
                )
            else:
                lines.append(
                    f"Active rental market ({int(item['rent_count'])} rental contracts)"
                )
        elif factor == "readiness":
            lines.append("Strong readiness / availability status on recorded project data")
        else:
            lines.append(_why_line(factor))
    return lines


def format_footer(result: dict, *, intro: str | None = None) -> str:
    """Positive Acot Recommend block for answers."""
    lines = []
    if intro:
        lines.append(intro)
        lines.append("")
    rec = result.get("recommendation") or {}
    winner = result.get("winner")
    ranked = result.get("ranked") or []
    if rec.get("tied"):
        names = " and ".join(rec["tied"][:2])
        lines.append(f"**Acot Recommend:** {names}")
        lines.append("**Why:**")
        for item in rec.get("why") or [
            "Both lead on a balanced mix of price, size, sales activity and rental activity."
        ]:
            lines.append(f"- {_why_line(item) if item in _WHY_LABELS else item}")
        for note in rec.get("notes") or result.get("notes") or []:
            lines.append(f"- {note}")
        return "\n".join(lines)

    name = rec.get("name") or ((winner or {}).get("name"))
    if not name:
        notes = rec.get("notes") or result.get("notes") or []
        thin = any("thin" in str(note).lower() or "transaction" in str(note).lower() for note in notes)
        if thin:
            lines.append(
                "**Acot Recommend:** Hold for stronger market evidence before picking a winner."
            )
            lines.append("**Why:**")
            for note in notes[:2]:
                lines.append(f"- {note}")
            return "\n".join(lines)
        lines.append("**Acot Recommend:** These projects are ready to compare in more detail.")
        lines.append("**Why:**")
        lines.append("- Each option brings useful price, size or market activity signals.")
        return "\n".join(lines)

    # Prefer full ranked rows for vs-alternative facts (stubs only have name/id).
    alt_rows = [
        row for row in ranked
        if isinstance(row, dict)
        and row.get("name")
        and str(row.get("name")) != str(name)
    ]
    if not alt_rows:
        alt_rows = list(rec.get("alternatives") or [])

    lines.append(f"**Acot Recommend:** {name}")
    lines.append("**Why this is the strongest match:**" if result.get("can_rank") else "**Why:**")
    why = rec.get("why") or result.get("why") or []
    detail = _detail_why(winner, why, alternatives=alt_rows)
    for item in detail:
        lines.append(f"- {item}")
    # Skip weak "Limited data: ranked on …" when concrete fact lines already explain.
    for note in rec.get("notes") or result.get("notes") or []:
        lower = str(note).lower()
        if lower.startswith("limited") and detail:
            continue
        if lower.startswith("limited"):
            lines.append(f"- {note}")
    alts = rec.get("alternatives") or []
    if alts:
        alt_names = ", ".join(
            str(alt.get("name") or "") for alt in alts[:3] if alt.get("name")
        )
        if alt_names:
            lines.append(f"- Also worth reviewing: {alt_names}.")
    return "\n".join(lines)


def format_shortlist(result: dict, *, question: str | None = None, assumptions: str | None = None) -> str:
    """Ranked list + Acot Recommend footer for invest / which-is-best questions."""
    from app.answers.explain_contract import build_understood, slots_from_question

    ranked = result.get("ranked") or []
    goal = (result.get("goal") or "").lower()
    slots = slots_from_question(question)
    lines = [
        build_understood(
            place=slots.get("place"),
            beds=slots.get("beds"),
            budget_aed=slots.get("budget_aed"),
            goal=goal or "invest",
            intent="recommend",
            under_budget=bool(slots.get("under_budget", True)),
        ),
        "",
    ]
    if assumptions:
        lines.append(assumptions)
        lines.append("")
    if goal == "invest":
        lead = "Best matches for investing, based on "
    elif goal == "family":
        lead = "Best matches for a family ask, based on "
    else:
        lead = "Best matches for investing and buying, based on "
    # For invest, emphasize market factors in the header when present.
    header_factors = result.get("used_factors")
    if goal == "invest":
        market = [
            name for name in (header_factors or [])
            if name in _INVEST_EVIDENCE_FACTORS
        ]
        if market:
            header_factors = market
    lines.append(lead + _factor_phrase(header_factors) + ":")
    lines.append("")
    for index, item in enumerate(ranked[:12], start=1):
        bits = [f"{index}. **{item['name']}**"]
        price = item.get("price")
        if price:
            bits.append(f"AED {int(price):,}")
        size = item.get("size")
        if size:
            bits.append(f"{int(size):,} sqft")
        sales = item.get("sales_count")
        if sales:
            bits.append(f"{int(sales)} recent sales")
        rents = item.get("rent_count")
        if rents:
            bits.append(f"{int(rents)} rental contracts")
        if goal == "invest" and not (
            price or sales or rents or item.get("last_sale")
        ):
            bits.append("limited sales/price data")
        lines.append(" — ".join(bits))
    lines.append("")
    lines.append(format_footer(result))
    return "\n".join(lines).rstrip()


def format_explain_on_screen(
    result: dict,
    *,
    goal: str | None = None,
    place: str | None = None,
    beds: int | None = None,
) -> str:
    """Explain why the on-screen shortlist fits — do not re-run a fresh search list."""
    where = place or "Dubai"
    if goal == "family":
        bed_bit = f"{beds}-bedroom " if beds is not None else "family-sized "
        lead = (
            f"For a {bed_bit}family ask in {where}, these on-screen projects were ranked on "
            f"{_factor_phrase(result.get('used_factors'))} from recorded sales — "
            "not schools, parks or lifestyle labels (those fields are not in our data)."
        )
    elif goal == "budget":
        lead = (
            f"These on-screen options in {where} were ranked on "
            f"{_factor_phrase(result.get('used_factors'))}, with budget fit first."
        )
    elif goal == "invest":
        lead = (
            f"These on-screen options in {where} were ranked on "
            f"{_factor_phrase(result.get('used_factors'))} from recorded sales and rental activity."
        )
    else:
        lead = (
            f"These on-screen options in {where} were ranked on "
            f"{_factor_phrase(result.get('used_factors'))} from recorded sales."
        )
    lines = [lead, "", format_footer(result)]
    if not result.get("can_rank"):
        lines.append(
            "Limited data — only a few recorded factors differ between these options."
        )
    lines.append("Share a budget, bedroom tweak or nearby area if you want a tighter list.")
    return "\n".join(lines).rstrip()


def format_compare_table(projects: list[dict], result: dict) -> str:
    """Side-by-side style text for named compare when not using structured snapshots."""
    items = result.get("ranked") or [features(row) for row in projects]
    ordered = []
    for row in projects:
        for item in items:
            if item.get("row") is row and item not in ordered:
                ordered.append(item)
                break
    for item in items:
        if item not in ordered:
            ordered.append(item)
    names = [item["name"] for item in ordered]
    header = "| Factor | " + " | ".join(names) + " |"
    sep = "|" + " --- |" * (len(names) + 1)

    def cell(value):
        return str(value).replace("|", "\\|")

    def price_cell(item):
        if item.get("price") is None:
            return "—"
        kind = str(item.get("price_kind") or "")
        if kind == "recent_sale":
            label = "recent sale"
        elif kind in {"yearly rent", "rent"}:
            label = "yearly rent"
        else:
            label = "price"
        return f"AED {int(item['price']):,} ({label})"

    def beds_cell(item):
        beds = item.get("beds")
        if beds is None:
            return "—"
        if beds == 0:
            return "Studio"
        if beds == 1:
            return "1 bedroom"
        return f"{beds:g} bedrooms"

    def hand_cell(item):
        stamp = item.get("handover")
        if stamp is None:
            return "—"
        return _month_year(stamp) or "—"

    def count_cell(item, key):
        value = item.get(key)
        return "—" if value is None else str(int(value))

    rentals = bool(ordered) and all(
        str(item.get("price_kind") or "") in {"yearly rent", "rent"}
        for item in ordered
    )
    rows = [
        header,
        sep,
        "| Price | " + " | ".join(cell(price_cell(item)) for item in ordered) + " |",
        "| Size | " + " | ".join(
            cell(f"{int(item['size']):,} sqft" if item.get("size") else "—") for item in ordered
        ) + " |",
        "| Bedrooms | " + " | ".join(cell(beds_cell(item)) for item in ordered) + " |",
    ]
    show_activity = (not rentals) or any(
        item.get("sales_count") is not None or item.get("rent_count") is not None
        for item in ordered
    )
    if show_activity and not rentals:
        rows.append(
            "| Recent sales | " + " | ".join(cell(count_cell(item, "sales_count")) for item in ordered) + " |"
        )
        rows.append(
            "| Rental contracts | " + " | ".join(cell(count_cell(item, "rent_count")) for item in ordered) + " |"
        )
    elif any(item.get("sales_count") is not None for item in ordered):
        rows.append(
            "| Recent sales | " + " | ".join(cell(count_cell(item, "sales_count")) for item in ordered) + " |"
        )
    if any(item.get("rent_count") is not None for item in ordered) and rentals:
        rows.append(
            "| Rental contracts | " + " | ".join(cell(count_cell(item, "rent_count")) for item in ordered) + " |"
        )
    if (not rentals) or any(item.get("handover") is not None for item in ordered):
        rows.append(
            "| Handover | " + " | ".join(cell(hand_cell(item)) for item in ordered) + " |"
        )
    scores = [float(item.get("score") or 0) for item in ordered]
    low, high = (min(scores), max(scores)) if scores else (0, 0)
    if not scores or high == low:
        score_cells = ["50%"] * len(ordered)
    else:
        score_cells = [f"{round(100 * (score - low) / (high - low))}%" for score in scores]
    rows.append("| Score | " + " | ".join(cell(value) for value in score_cells) + " |")
    if rentals:
        intro = (
            f"Side-by-side comparison of {', '.join(names)} on "
            + _factor_phrase(result.get("used_factors"))
            + ":"
        )
    else:
        intro = (
            f"Side-by-side comparison of {', '.join(names)} on price, size, sales and rentals:"
        )
    return intro + "\n\n" + "\n".join(rows) + "\n\n" + format_footer(result)


def tag_project(row: dict | None) -> str:
    """Display name only. Cards come from structured data, never from raw ids."""
    name = str((row or {}).get("name") or "this project").strip() or "this project"
    return f"**{name}**"


def _aed_text(value) -> str | None:
    number = _num(value)
    if number is None or number <= 0:
        return None
    return f"AED {int(round(number)):,}"


def _project_points(row: dict | None) -> list[str]:
    from app.answers.grounding import ground_project_row

    row = ground_project_row(row or {})
    points = []
    community = str(row.get("community") or row.get("area") or "").strip()
    if community:
        points.append(community)
    price = _aed_text(row.get("price") or row.get("starting_price_aed") or row.get("price_aed"))
    if price:
        kind = str(row.get("price_kind") or "")
        points.append(
            f"recent sale around {price}" if kind == "recent_sale" else f"from {price}"
        )
    sales = _num(row.get("sales_count"), row.get("recent_sales_count"))
    if sales:
        points.append(f"{int(sales)} sales in the last 12 months")
    sold = _handover_label({"handover_time": row.get("last_sold_on") or row.get("last_sale")})
    if sold:
        points.append(f"latest recorded sale {sold}")
    rents = _num(row.get("rent_count"), row.get("rental_count"))
    median_rent = _aed_text(
        _num(row.get("median_rent"), row.get("median_annual_rent_aed"))
    )
    if rents and median_rent:
        points.append(f"{int(rents)} rent contracts (median ~{median_rent}/yr)")
    elif rents:
        points.append(f"{int(rents)} rent contracts in the last 18 months")
    elif median_rent:
        points.append(f"median rent ~{median_rent}/yr")
    # Handover is secondary context only — never the sole reason to recommend.
    hand = _handover_label(row)
    if (
        hand
        and _is_future_stamp(_handover_stamp(row.get("handover_time") or row.get("handover")))
        and (sales or rents or sold)
    ):
        points.append(f"handover {hand}")
    return points


def _project_bullet(index: int, row: dict) -> str:
    points = _project_points(row)
    detail = f" — {'; '.join(points)}" if points else ""
    return f"{index}. {tag_project(row)}{detail}."


def _handover_label(row: dict) -> str | None:
    value = row.get("handover_time") or row.get("handover")
    stamp = _handover_stamp(value)
    if stamp is None:
        return str(value) if value else None
    return _month_year(stamp)


def within_months(row: dict, months: int) -> bool:
    stamp = _handover_stamp(row.get("handover_time") or row.get("handover"))
    if stamp is None:
        text = str(row.get("status") or row.get("project_status") or "")
        return bool(_READY.search(text))
    horizon = datetime.utcnow().timestamp() + months * 30 * 24 * 3600
    return stamp <= horizon


def format_ranked_picks(
    ranked: dict,
    *,
    intro: str,
    relaxed: str | None = None,
    question: str | None = None,
    place: str | None = None,
    beds: int | None = None,
    budget_aed: int | None = None,
    goal: str | None = None,
) -> str:
    """Search → rank → Acot Recommend + Why (Phase 3 recommend shape).

    Always leads with an Understood line (filters/goal), then grounded picks.
    """
    from app.answers.explain_contract import (
        build_motive,
        build_understood,
        prepend_understood,
        slots_from_question,
    )

    slots = slots_from_question(question)
    understood = build_understood(
        place=place or slots.get("place"),
        beds=beds if beds is not None else slots.get("beds"),
        budget_aed=budget_aed if budget_aed is not None else slots.get("budget_aed"),
        ready_only=bool(slots.get("ready_only")),
        goal=goal,
        intent="recommend",
        under_budget=bool(slots.get("under_budget", True)),
    )

    items = ranked.get("ranked") or []
    winner = ranked.get("winner") or (items[0] if items else None)
    shown = items[:3]
    if not winner:
        return prepend_understood(intro.rstrip(",") + ".", understood)
    motive = build_motive("recommend")
    bridge = (
        "For each pick I show recent sale level, sales activity, rent contracts when recorded, "
        "and community — so you can compare with market facts, not brochure claims alone."
    )
    lines = [motive, "", intro.rstrip(",") + ".", "", bridge, ""]
    for index, item in enumerate(shown, start=1):
        row = item.get("row") or item
        lines.append(_project_bullet(index, {**row, "name": item.get("name") or row.get("name")}))
    lines.append("")
    lines.append(format_footer(ranked))
    if not ranked.get("can_rank"):
        lines.append(
            "Limited differentiating factors — only a few recorded fields differ between these options."
        )
    if relaxed:
        lines.append(relaxed)
    else:
        lines.append(
            "Share a bedroom tweak, a tighter budget, or a nearby community and I’ll refine this shortlist."
        )
    return prepend_understood("\n".join(lines), understood)


_NO_BRIEF = re.compile(
    r"\bno (?:area|budget|place)|without (?:an? )?(?:area|budget)|"
    r"not (?:sure|decided)|haven'?t (?:picked|chosen|decided)|"
    r"no area yet|no budget yet",
    re.I,
)
_BEST_PROPERTY = re.compile(
    r"\bbest (?:property|project)|which one is the best",
    re.I,
)
_WHERE_BUY = re.compile(r"\bwhere should i (?:buy|invest|live)\b", re.I)


def _open_buy_header(question: str | None, *, more: bool = False) -> list[str]:
    """First lines should answer the question, not reuse one canned opener."""
    from app.answers.explain_contract import build_understood, slots_from_question

    text = question or ""
    slots = slots_from_question(text)
    understood = build_understood(
        place=slots.get("place"),
        beds=slots.get("beds"),
        budget_aed=slots.get("budget_aed"),
        ready_only=bool(slots.get("ready_only")),
        goal="open_buy",
        intent="recommend",
        under_budget=bool(slots.get("under_budget", True)),
    )
    if more:
        return [
            understood,
            "",
            "More communities from the same market map, still ranked by recorded sales:",
            "",
        ]
    # Never claim "without an area" when the user already named one.
    try:
        from app.answers.intent_route import extract_place

        locked = extract_place(text)
    except Exception:
        locked = None
    if locked and str(locked).strip().lower() not in {"", "dubai"}:
        return [
            understood,
            "",
            f"Here are strong recorded options in {locked} based on recent sales activity:",
            "",
        ]
    if _NO_BRIEF.search(text) or _WHERE_BUY.search(text):
        return [
            understood,
            "",
            "If you have no area or budget yet, start from the goal — live in the home, or rent it out — rather than picking a building first.",
            "",
            "To map the market, these Dubai areas had the most recorded sales in the last 12 months:",
            "",
        ]
    if _BEST_PROPERTY.search(text):
        return [
            understood,
            "",
            "There isn’t a single best property in Dubai without an area and a budget.",
            "",
            "From recorded sales in the last 12 months, these communities are the most active right now:",
            "",
        ]
    return [
        understood,
        "",
        "Without a locked area or budget, recent sales activity is a useful map of where people are buying.",
        "",
        "From recorded sales in the last 12 months:",
        "",
    ]


def format_open_best(ranked: dict, question: str | None = None) -> str:
    from app.answers.explain_contract import build_understood, prepend_understood

    items = (ranked.get("ranked") or [])[:3]
    if not items:
        return prepend_understood(
            "Share your budget and preferred area and I’ll match buy options from our data.",
            build_understood(goal="open_buy", intent="recommend"),
        )
    lines = _open_buy_header(question)
    lines[-2] = "From our project data, these are practical buy matches right now:"
    for index, item in enumerate(items, start=1):
        row = item.get("row") or item
        lines.append(_project_bullet(index, {**row, "name": item.get("name") or row.get("name")}))
    lines.append("")
    lines.append("What’s your budget, and is this to live in or to rent out?")
    return "\n".join(lines)


_FAMILY_LABELS = {
    "springs": "The Springs",
    "meadows": "The Meadows",
    "dubai hills": "Dubai Hills Estate",
}


def _family_area_label(raw: str) -> str:
    from app.listings.tools import ALIASES

    text = " ".join(str(raw or "").split())
    if not text:
        return ""
    key = text.lower()
    if key in _FAMILY_LABELS:
        return _FAMILY_LABELS[key]
    if key in ALIASES:
        return ALIASES[key]
    for alias, proper in ALIASES.items():
        if key == alias or key.startswith(alias + " "):
            return proper
    return text.title()


def _family_share(part, whole) -> float:
    total = _num(whole) or 0
    if total <= 0:
        return 0.0
    return (_num(part) or 0) / total


def rank_family_areas(items: list[dict], focus: dict | None = None, limit: int = 5) -> list[dict]:
    """Ready villas beat raw 3-bed volume so new apartment launches do not win."""
    focus = focus or {}
    cap = max(3, min(int(limit or 5), 8))
    rows = []
    for item in items:
        existing = int(item.get("existing_sales") or 0)
        villas_ready = int(item.get("existing_villas") or 0)
        villas = int(item.get("villa_sales") or item.get("villa_townhouse_sales") or 0)
        family = int(item.get("existing_family_sales") or item.get("family_sales") or 0)
        homes = int(item.get("sales_12m") or 0)
        row = {
            **item,
            "existing_villas": villas_ready,
            "villa_sales": villas,
            "villa_townhouse_sales": villas,
            "existing_villa_share": _family_share(villas_ready, existing),
            "villa_share": _family_share(villas, homes),
        }
        rows.append(row)

    # Kids / not-towers → rank by villa share (family guides), not raw 3-bed volume.
    low_rise = bool(focus.get("avoid_towers") or focus.get("kids"))
    villa_stock = bool(
        focus.get("want_villas")
        and (focus.get("want_3bed") or focus.get("not_studios"))
        and not low_rise
    )

    if low_rise:
        picked = [
            row for row in rows
            if row["existing_villas"] >= 20 and row["existing_villa_share"] >= 0.35
        ]
        picked.sort(
            key=lambda row: (
                row["existing_villa_share"],
                row["existing_villas"],
                int(row.get("existing_family_sales") or row.get("family_sales") or 0),
            ),
            reverse=True,
        )
        if len(picked) < 3:
            picked = [
                row for row in rows
                if row["existing_villas"] >= 15 and row["existing_villa_share"] >= 0.25
            ]
            picked.sort(
                key=lambda row: (
                    row["existing_villa_share"],
                    row["existing_villas"],
                    int(row.get("existing_family_sales") or row.get("family_sales") or 0),
                ),
                reverse=True,
            )
    elif villa_stock:
        picked = [
            row for row in rows
            if row["villa_sales"] >= 40 and row["villa_share"] >= 0.25
        ]
        picked.sort(
            key=lambda row: (row["villa_sales"], int(row.get("family_sales") or 0)),
            reverse=True,
        )
    else:
        picked = [
            row for row in rows
            if row["existing_villas"] >= 20 and row["existing_villa_share"] >= 0.08
        ]
        picked.sort(
            key=lambda row: (
                row["existing_villas"],
                int(row.get("existing_family_sales") or row.get("family_sales") or 0),
            ),
            reverse=True,
        )
        if len(picked) < 3:
            picked = [
                row for row in rows
                if row["existing_villas"] >= 15 and row["existing_villa_share"] >= 0.08
            ]
            picked.sort(
                key=lambda row: (
                    row["existing_villas"],
                    int(row.get("existing_family_sales") or row.get("family_sales") or 0),
                ),
                reverse=True,
            )
    return picked[:cap]


def fetch_family_areas(limit: int = 5, focus: dict | None = None) -> list[dict]:
    """Communities with ready villa / 3-bed sales, not land or tower volume."""
    from app.database.postgres import fetch

    try:
        rows = fetch(
            """
            with s as (
              select coalesce(nullif(btrim(master_project_prop_ai), ''), area_name_prop_ai) as area,
                     bedrooms,
                     property_type_en,
                     development_status,
                     sale_price
              from public.propai_sales
              where deleted_at is null
                and coalesce(outlier_flag, '') <> 'True'
                and sale_price > 0
                and instance_date >= current_date - interval '12 months'
                and (
                  property_type_en = 'Villa'
                  or (property_type_en = 'Unit' and property_sub_type_en = 'Flat')
                )
                and coalesce(nullif(btrim(master_project_prop_ai), ''), area_name_prop_ai) is not null
            )
            select area,
                   count(*)::int as sales_12m,
                   count(*) filter (where development_status = 'Existing')::int as existing_sales,
                   count(*) filter (where property_type_en = 'Villa')::int as villa_sales,
                   count(*) filter (
                     where property_type_en = 'Villa' and development_status = 'Existing'
                   )::int as existing_villas,
                   count(*) filter (where bedrooms >= 3)::int as family_sales,
                   count(*) filter (
                     where bedrooms >= 3 and development_status = 'Existing'
                   )::int as existing_family_sales,
                   percentile_cont(0.5) within group (order by sale_price)
                     filter (
                       where bedrooms >= 3 and development_status = 'Existing'
                     )::float8 as median_family_sale
            from s
            group by 1
            having count(*) filter (where property_type_en = 'Villa') >= 15
                or count(*) filter (
                     where bedrooms >= 3 and development_status = 'Existing'
                   ) >= 30
            """
        )
    except Exception:
        return []
    items = []
    for row in rows:
        raw = row.get("area")
        if not raw:
            continue
        median = row.get("median_family_sale")
        items.append(
            {
                "area": _family_area_label(raw),
                "official_area": raw,
                "sales_12m": int(row.get("sales_12m") or 0),
                "existing_sales": int(row.get("existing_sales") or 0),
                "villa_sales": int(row.get("villa_sales") or 0),
                "villa_townhouse_sales": int(row.get("villa_sales") or 0),
                "existing_villas": int(row.get("existing_villas") or 0),
                "family_sales": int(row.get("family_sales") or 0),
                "existing_family_sales": int(row.get("existing_family_sales") or 0),
                "median_family_sale": int(round(median)) if median else None,
            }
        )
    return rank_family_areas(items, focus=focus, limit=limit)


def format_family_areas(
    areas: list[dict],
    focus: dict | None = None,
    budget_aed: int | None = None,
    prefer_cheaper: bool = False,
    prefer_premium: bool = False,
) -> str:
    from app.answers.explain_contract import (
        build_family_motive,
        build_understood,
        prepend_understood,
    )

    focus = focus or {}
    understood = build_understood(
        budget_aed=budget_aed,
        goal="family",
        intent="recommend",
        under_budget=True,
    )
    if budget_aed:
        fitting = [
            row for row in (areas or [])
            if row.get("median_family_sale") and int(row["median_family_sale"]) <= int(budget_aed)
        ]
        prices = [
            int(row["median_family_sale"])
            for row in (areas or [])
            if row.get("median_family_sale")
        ]
        if fitting:
            intro = (
                f"You set a budget of AED {int(budget_aed):,} after asking for family areas. "
                "These communities have typical 3+ bedroom sales at or under that level:"
            )
            follow = "Want a villa or a townhouse in one of these?"
            areas = fitting
        else:
            band = ""
            if prices:
                band = (
                    f" Typical 3+ bedroom sales in those family communities sit around "
                    f"AED {min(prices):,} to AED {max(prices):,}."
                )
            intro = (
                f"You set a budget of AED {int(budget_aed):,} after asking for family areas.{band} "
                "That is below recorded villa and 3-bedroom sale prices in our data, "
                "so I am not listing a buy match at this budget."
            )
            follow = "If you meant AED 5 million, or a monthly rent, say which and I’ll rerun this list."
            if not areas:
                return prepend_understood(intro + "\n\n" + follow, understood)
        if focus.get("no_yield"):
            intro = "ACOT does not have rental yield or ROI. " + intro
        lines = [intro, ""]
        for index, area in enumerate(areas[:5], start=1):
            name = str(area.get("area") or "This area").strip()
            bits = []
            ready_villas = area.get("existing_villas")
            if ready_villas:
                bits.append(f"{int(ready_villas):,} ready villa sales")
            family = area.get("existing_family_sales") or area.get("family_sales")
            if family:
                bits.append(f"{int(family):,} ready 3+ bedroom sales")
            price = _aed_text(area.get("median_family_sale"))
            if price:
                bits.append(f"typical 3+ bedroom sale around {price}")
            lines.append(f"{index}. **{name}**")
            if bits:
                lines.append("   " + ". ".join(bits) + ".")
            lines.append("")
        if fitting:
            lines.append("The cards below are example homes from these communities.")
        lines.append(follow)
        return prepend_understood("\n".join(lines), understood)
    if not areas:
        return prepend_understood(
            "Share your budget and whether you want a villa, townhouse or apartment, "
            "and I’ll match family-size homes from our data.",
            understood,
        )
    if focus.get("avoid_towers") or (focus.get("kids") and focus.get("want_villas")):
        intro = (
            "You asked for family communities for kids — low-rise villa stock, "
            "not the busiest apartment towers. "
            "From ready sales in the last 12 months, these areas lead on villa share:"
        )
        follow = "Want me to narrow this to a budget, or to townhouses only?"
    elif focus.get("want_3bed") and focus.get("want_villas"):
        intro = (
            "These Dubai communities currently have the most 3-bedroom and villa sales in our records — "
            "the stock families usually look for:"
        )
        follow = "What’s your budget, and is a villa or a townhouse the priority?"
    elif focus.get("not_studios") or (focus.get("want_villas") and not focus.get("kids")):
        intro = (
            "These communities are selling villas and townhouses, not studios. "
            "Figures are from recorded sales in the last 12 months:"
        )
        follow = "What budget should I use to shortlist a home?"
    elif focus.get("kids"):
        intro = (
            "For a family with children, extra bedrooms and villas are the clearest signal in our sales data. "
            "These communities lead on ready villa share — not tower transaction volume:"
        )
        follow = "What’s your budget, and do you want a villa or a townhouse?"
    else:
        intro = (
            "Families usually need more bedrooms and more low-rise homes. "
            "From ready sales in the last 12 months, these Dubai communities stand out on villa "
            "and 3-bedroom activity — ranked on recorded stock, not lifestyle marketing:"
        )
        follow = "What’s your budget, and do you want a villa, townhouse, or apartment?"
    # Focus-aware motive first so kids/towers vs general family answers diverge immediately.
    intro = (
        build_family_motive(focus)
        + "\n\n"
        + intro
        + "\n\n"
        "For each community I quote ready villa sales, villa share of ready sales, "
        "3+ bedroom sales, and the typical 3+ bedroom sale price when recorded."
    )
    if prefer_cheaper:
        intro = (
            build_family_motive(focus)
            + "\n\n"
            "You asked for cheaper family communities from the same villa / kids ranking. "
            "These are ordered by typical 3+ bedroom sale price, lowest first. "
            "Each line still shows villa sales, villa share, and 3+ bedroom sale level when recorded:"
        )
        follow = "Want a budget cap, or townhouses only?"
    elif prefer_premium:
        intro = (
            build_family_motive(focus)
            + "\n\n"
            "You asked for higher / best family communities from the same villa / kids ranking. "
            "These are ordered by typical 3+ bedroom sale price, highest first. "
            "Each line still shows villa sales, villa share, and 3+ bedroom sale level when recorded:"
        )
        follow = "Want a budget cap, or townhouses only?"
    if focus.get("no_yield"):
        intro = (
            "ACOT does not have rental yield or ROI percentages in this dataset — "
            "I still give the sale facts below.\n\n"
            + intro
        )
    lines = [intro, ""]
    for index, area in enumerate(areas[:5], start=1):
        name = str(area.get("area") or "This area").strip()
        bits = []
        ready_villas = area.get("existing_villas")
        if ready_villas:
            bits.append(f"{int(ready_villas):,} ready villa sales")
        elif area.get("villa_sales") or area.get("villa_townhouse_sales"):
            bits.append(
                f"{int(area.get('villa_sales') or area.get('villa_townhouse_sales')):,} villa sales"
            )
        share = area.get("existing_villa_share")
        if share and share >= 0.08:
            bits.append(f"{int(round(share * 100))}% of ready sales are villas")
        family = area.get("existing_family_sales") or area.get("family_sales")
        if family:
            bits.append(f"{int(family):,} ready 3+ bedroom sales")
        price = _aed_text(area.get("median_family_sale"))
        if price:
            bits.append(f"typical 3+ bedroom sale around {price}")
        lines.append(f"{index}. **{name}**")
        if bits:
            lines.append("   " + ". ".join(bits) + ".")
        lines.append("")
    lines.append("The cards below are example homes from these communities.")
    lines.append(follow)
    return prepend_understood("\n".join(lines), understood)


def format_open_areas(areas: list[dict], question: str | None = None, more: bool = False) -> str:
    if not areas:
        return (
            "If you have no area or budget yet, tell me whether this is to live in or to rent out "
            "and I’ll map buy options from our sales data."
        )
    lines = _open_buy_header(question, more=more)
    for index, area in enumerate(areas[:5], start=1):
        name = str(area.get("area") or "This area").strip()
        bits = []
        if area.get("sales_12m"):
            bits.append(f"{int(area['sales_12m']):,} recorded sales")
        price = _aed_text(area.get("median_sale_price_aed") or area.get("median_price"))
        if price:
            bits.append(f"typical sale around {price}")
        psf = area.get("median_price_per_sqft_aed") or area.get("median_ppsf")
        if psf:
            bits.append(f"about AED {int(psf):,} per sq ft")
        lines.append(f"{index}. **{name}**")
        if bits:
            lines.append("   " + ". ".join(bits) + ".")
        lines.append("")
    lines.append("The cards below are example projects from these areas.")
    lines.append("What’s your budget, and is this to live in or to rent out?")
    return "\n".join(lines)


def format_cheapest(
    areas: list[dict],
    projects: list[dict],
    place: str | None = None,
    more: bool = False,
) -> str:
    """Cheapest communities/projects from recorded sales, not asking rents."""
    from app.answers.explain_contract import build_understood, prepend_understood

    where = f" in {place}" if place else " in Dubai"
    understood = build_understood(place=place, goal="budget", intent="recommend")
    if more:
        lines = [
            f"More of the lowest recorded sale prices{where}:",
            "",
        ]
    else:
        lines = [
            f"Cheapest to live{where} is based on recorded sale prices in the last 12 months, "
            "not asking rents or lifestyle scores.",
            "",
        ]
    if areas:
        lines.append("These communities currently have the lowest typical sale prices:")
        lines.append("")
        for index, area in enumerate(areas[:5], start=1):
            name = str(area.get("area") or "This area").strip()
            bits = []
            price = _aed_text(area.get("median_sale_price_aed") or area.get("median_price"))
            if price:
                bits.append(f"typical sale around {price}")
            if area.get("sales_12m"):
                bits.append(f"{int(area['sales_12m']):,} recorded sales")
            psf = area.get("median_price_per_sqft_aed") or area.get("median_ppsf")
            if psf:
                bits.append(f"about AED {int(psf):,} per sq ft")
            lines.append(f"{index}. **{name}**")
            if bits:
                lines.append("   " + ". ".join(bits) + ".")
            lines.append("")
    shown = projects[:3]
    if shown:
        if areas:
            lines.append("Lowest-priced projects from that same recorded-sales set:")
        else:
            lines.append(f"Lowest-priced projects{where} from recorded sales:")
        lines.append("")
        for index, row in enumerate(shown, start=1):
            lines.append(_project_bullet(index, row))
        lines.append("")
    if not areas and not shown:
        return prepend_understood(
            f"Share a community or a bedroom count and I’ll rank the cheapest recorded sales{where}.",
            understood,
        )
    lines.append("The cards below are example projects from this cheapest list.")
    lines.append("Share a bedroom count or a max budget and I’ll tighten this.")
    return prepend_understood("\n".join(lines), understood)


def format_yield_areas(areas: list[dict], place: str | None = None) -> str:
    """No yield %. Quote recorded rents and sales only."""
    if not areas:
        return (
            "ACOT does not have rental yield or ROI. "
            "Share an area and I’ll quote the recorded rents and sale prices we do have."
        )
    lines = [
        "ACOT does not have rental yield or ROI — that needs matched rent and purchase prices on the same homes.",
        "",
    ]
    if place:
        lines.append(
            f"From recorded rents and sales, here is what we have for {place} and other active rental areas:"
        )
    else:
        lines.append(
            "From recorded rents, these Dubai areas currently have the most rental contracts:"
        )
    lines.append("")
    for index, area in enumerate(areas[:5], start=1):
        name = str(area.get("area") or "This area").strip()
        bits = []
        contracts = area.get("rent_contracts")
        if contracts:
            bits.append(f"{int(contracts):,} recorded rental contracts")
        rent = _aed_text(area.get("median_yearly_rent_aed") or area.get("median_rent"))
        if rent:
            bits.append(f"typical yearly rent around {rent}")
        price = _aed_text(area.get("median_sale_price_aed") or area.get("median_price"))
        if price:
            bits.append(f"typical sale around {price}")
        sales = area.get("sales_12m")
        if sales:
            bits.append(f"{int(sales):,} sales in the last 12 months")
        lines.append(f"{index}. **{name}**")
        if bits:
            lines.append("   " + ". ".join(bits) + ".")
        lines.append("")
    lines.append("The cards below are example projects from these areas.")
    lines.append("What’s your budget, and is this to live in or to rent out?")
    return "\n".join(lines)


def fetch_ready_offplan_stats(place: str | None = None) -> dict[str, dict]:
    """Last-12-month sales split into ready vs off-plan. No yield or forecast."""
    from app.database.postgres import fetch

    clauses = [
        "deleted_at is null",
        "coalesce(outlier_flag, '') <> 'True'",
        "sale_price > 0",
        "instance_date >= current_date - interval '12 months'",
        "coalesce(property_type_en, '') not in ('Land', 'Plot')",
    ]
    params: list = ["%off%plan%", "%exist%", "%ready%", "%complet%"]
    if place:
        clauses.append(
            "(area_name_prop_ai ilike %s or master_project_prop_ai ilike %s)"
        )
        like = f"%{place}%"
        params.extend([like, like])
    try:
        rows = fetch(
            f"""
            select
              case
                when development_status ilike %s then 'offplan'
                when development_status ilike %s
                  or development_status ilike %s
                  or development_status ilike %s
                  then 'ready'
                else 'other'
              end as band,
              count(*)::int as sales_12m,
              percentile_cont(0.5) within group (order by sale_price)::float8 as median_price,
              percentile_cont(0.5) within group (
                order by price_per_sqft
              ) filter (where price_per_sqft > 0)::float8 as median_ppsf
            from public.propai_sales
            where {' and '.join(clauses)}
            group by 1
            """,
            tuple(params),
        )
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for row in rows or []:
        band = str(row.get("band") or "")
        if band not in {"ready", "offplan"}:
            continue
        out[band] = {
            "sales_12m": int(row.get("sales_12m") or 0),
            "median_price": row.get("median_price"),
            "median_ppsf": row.get("median_ppsf"),
        }
    return out


def _status_stat_line(label: str, stats: dict | None) -> list[str]:
    stats = stats or {}
    bits = []
    if stats.get("sales_12m"):
        bits.append(f"{int(stats['sales_12m']):,} recorded sales")
    price = _aed_text(stats.get("median_price"))
    if price:
        bits.append(f"typical sale around {price}")
    psf = _num(stats.get("median_ppsf"))
    if psf:
        bits.append(f"about AED {int(psf):,} per sq ft")
    lines = [f"**{label}**"]
    if bits:
        lines.append("   " + ". ".join(bits) + ".")
    else:
        lines.append("   No recorded sales in this split for the last 12 months.")
    return lines


def format_ready_offplan(
    stats: dict[str, dict] | None,
    ready_projects: list[dict],
    offplan_projects: list[dict],
    place: str | None = None,
) -> str:
    from app.answers.explain_contract import build_understood, prepend_understood

    where = f" in {place}" if place else " in Dubai"
    lines = [
        (
            f"**Acot Recommend:** Ready and off-plan are both active{where}. "
            "I compare them on recorded sales in the last 12 months — sale counts, "
            "typical sale price, and price per sq ft when available — so you can see "
            "what the market cleared for each path. ACOT does not pick a winner: that "
            "depends on whether you need the home now or can wait for handover."
        ),
        "",
        "From recorded sales in the last 12 months:",
        "",
    ]
    lines.extend(_status_stat_line("Ready (existing homes)", (stats or {}).get("ready")))
    lines.append("")
    for index, row in enumerate(ready_projects[:2], start=1):
        lines.append(_project_bullet(index, row))
    if not ready_projects:
        lines.append("Example ready projects are on the cards when the data has them.")
    lines.append("")
    lines.extend(_status_stat_line("Off-plan", (stats or {}).get("offplan")))
    lines.append("")
    start = len(ready_projects[:2]) + 1
    for offset, row in enumerate(offplan_projects[:2]):
        lines.append(_project_bullet(start + offset, row))
    if not offplan_projects:
        lines.append("Example off-plan projects are on the cards when the data has them.")
    lines.append("")
    lines.append(
        "General guidance: ready usually means you can live in or rent it after transfer. "
        "Off-plan usually means a payment plan and waiting until handover. "
        "Typical terms vary by developer — check the sale contract. This is not legal or financial advice."
    )
    lines.append("")
    lines.append("The cards below are example ready and off-plan projects.")
    lines.append("What’s your budget, and do you need to move in this year?")
    return prepend_understood(
        "\n".join(lines),
        build_understood(place=place, goal="ready", intent="recommend"),
    )


def _bed_label(bedrooms: int | None) -> str:
    if bedrooms is None:
        return "home"
    if bedrooms <= 0:
        return "studio"
    if bedrooms == 1:
        return "1-bedroom"
    return f"{int(bedrooms)}-bedroom"


def format_afford_answer(
    place: str,
    budget_aed: int,
    median: dict | None,
    projects: list[dict],
    nearby: list[dict],
    bedrooms: int | None = 1,
) -> str:
    names = [str(row.get("name") or "").strip() for row in projects[:4] if row.get("name")]
    areas = [str(row.get("area") or "").title() for row in nearby[:3] if row.get("area")]
    from app.answers.explain_contract import build_motive, build_understood, prepend_understood

    label = _bed_label(bedrooms)
    lead = prepend_understood(
        build_motive("afford")
        + "\n\n"
        + f"You're asking whether AED {budget_aed:,} is enough for a {label} in {place}.",
        build_understood(
            place=place,
            beds=bedrooms,
            budget_aed=budget_aed,
            goal="afford",
            intent="recommend",
            under_budget=True,
        ),
    )
    lines = [lead]
    if median and median.get("median"):
        centre = f"AED {int(median['median']):,}"
        lines.append(
            f"{label.capitalize()} sales in {place} recently centre around {centre} "
            f"(last 12 months of recorded Dubai sales)."
        )
        lines.append("")
        below = budget_aed < median["median"] * 0.92
        if names:
            # Under-budget recorded sales exist: never say the place is a miss.
            if below:
                lines.append(
                    f"AED {budget_aed:,} sits below that typical {place} level, "
                    f"but recorded {label} sales at or under this budget still exist there."
                )
            else:
                lines.append(f"AED {budget_aed:,} sits in that band.")
            lines.append(
                f"The cards below are recent {label} sales in {place} "
                f"at or under AED {budget_aed:,}."
            )
            if areas and below:
                lines.append(
                    "If you want more choice at this budget, recorded prices also line up in "
                    + " and ".join(areas)
                    + "."
                )
            lines.append("")
            lines.append(
                f"Want me to compare these {place} options, or switch to those nearby areas?"
            )
            return "\n".join(lines)
        if below:
            lines.append(
                f"AED {budget_aed:,} sits below that typical {place} level, "
                f"and I couldn’t find recorded {label} sales there at or under this budget."
            )
            if areas:
                lines.append(
                    f"AED {budget_aed:,} lines up with recorded prices in "
                    + " and ".join(areas)
                    + "."
                )
            lines.append("")
            lines.append(f"Want me to show {label}s in those nearby areas?")
            return "\n".join(lines)
        lines.append(f"AED {budget_aed:,} sits in that band.")
        lines.append("")
        lines.append(f"Want me to list {place} {label}s closest to this budget?")
        return "\n".join(lines)
    if names:
        return (
            lead + "\n\n"
            f"Here are recent {label} sales in {place} at or under AED {budget_aed:,}. "
            "They are on the cards below.\n\n"
            "Want me to compare these options?"
        )
    if areas:
        return (
            lead + "\n\n"
            f"AED {budget_aed:,} lines up with recorded prices in "
            + " and ".join(areas)
            + f". I can show {label}s there now."
        )
    return (
        lead + "\n\n"
        f"Share a bedroom count or a nearby area and I’ll match options around AED {budget_aed:,}."
    )


def format_split_answer(
    studios: list[dict],
    villas: list[dict],
    budget_aed: int | None,
    *,
    villa_beds: int | None = 4,
    question: str | None = None,
) -> str:
    """Explain-first split: Understood → ACOT motive → two shortlists → next step."""
    from app.answers.explain_contract import prepend_understood

    beds = int(villa_beds) if villa_beds is not None else 4
    bed_label = "studio" if beds == 0 else f"{beds}-bedroom"
    budget = f" under AED {budget_aed:,}" if budget_aed else ""
    understood = (
        f"Understood: you asked for both a {bed_label} villa and a studio apartment"
        f"{budget} at the same time."
    )
    from app.answers.explain_contract import build_motive

    # Enrich with sales + rent before bullets (Bayut/PF comps pattern).
    shortlists = [row for row in (list(studios or []) + list(villas or [])) if isinstance(row, dict)]
    if shortlists:
        attach_market_activity(shortlists)

    motive = build_motive(
        "split",
        detail=(
            "**Acot Recommend:** I ranked each path on recorded sales and rent activity "
            "first — studios for a compact buy, and "
            f"{bed_label} villas for family space — so you only see transaction-backed "
            "options, not handover-only towers."
        ),
    )
    lines = [
        motive,
        "",
        f"Studios{budget} (sales + rent evidence):",
    ]
    if studios:
        for index, row in enumerate(studios[:2], start=1):
            lines.append(_project_bullet(index, row))
    else:
        lines.append(
            "- Latest studio sales at that budget are limited — share an area and I will "
            "match the strongest recorded studio options."
        )
    lines.append("")
    lines.append(
        f"{beds}-bedroom villas{budget} (sales + rent evidence):"
    )
    if villas:
        for index, row in enumerate(villas[:2], start=1):
            lines.append(_project_bullet(index, row))
    else:
        lines.append(
            f"- Latest recorded {beds}-bedroom villa sales at that budget are limited — "
            "pick studios for a compact buy now, or share a higher budget / community "
            "and I will match stronger villa sales."
        )
    lines.append("")
    lines.append("Which path should I go deeper on — studios or villas?")
    _ = question
    return prepend_understood("\n".join(lines), understood)
