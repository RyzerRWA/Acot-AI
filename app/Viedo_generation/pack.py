"""
Build the video data pack from searched projects.

Takes history output (or a project list), applies agent filters, loads market
facts from propai_* tables, and returns the top 3–5 projects for a 60s video.
"""

from __future__ import annotations

import os
from typing import Any

from app.database.postgres import fetch

MIN_PHOTOS = 4
DEFAULT_LIMIT = 2
MAX_PHOTOS_PER_PROJECT = 6
# Sale / rent older than this are treated as stale and rejected.
RECENT_MONTHS = 12
# Enrich at most this many filtered candidates (DB lookups are expensive).
# Higher than limit because many fail the complete-data check.
ENRICH_CAP = 30

IMAGES_BASE_URL = (
    os.getenv("PROJECT_IMAGES_BASE_URL")
    or "https://storage.googleapis.com/kinetic-axle-442714-m6-dubai-photos"
).rstrip("/")


def build_pack(
    history_or_projects: dict[str, Any] | list[dict[str, Any]],
    *,
    bedrooms: int | None = None,
    communities: list[str] | None = None,
    project_keys: list[str] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """
    Return a video pack ready for script / TTS / render.

    Args:
      history_or_projects: output of searched_projects(), or a list of project dicts
      bedrooms: 1–4, or None for any
      communities: keep only these community names (case-insensitive); None = all
      project_keys: if set, keep only these catalog keys / names
      limit: legacy input, capped at two projects

    Shape:
      {
        "ok": bool,
        "reason": str | None,          # set when ok is False
        "bedrooms": int | None,
        "communities": list[str],
        "projects": [
          {
            "key", "name", "community", "developer",
            "photos": [url, ...],
            "recent_sale": {price, date, bedrooms, price_per_sqft} | None,
            "rent": {annual, date, bedrooms, rent_per_sqft} | None,
            "handover": str | None,    # "ready" | "Dec 2027" | None
            "status": str | None,
            "sales_12m": int,
            "rent_contracts_12m": int,
            "appeared": int,
            "source": "catalog" | "saved",
          },
          ...
        ],
      }
    """
    candidates = _candidates(history_or_projects)
    candidates = _apply_filters(candidates, bedrooms, communities, project_keys)
    shortlist = _shortlist(candidates, ENRICH_CAP)

    enriched: list[dict[str, Any]] = []
    for candidate in shortlist:
        row = _enrich(candidate, bedrooms)
        if row is None:
            continue
        # Incomplete or stale projects never go into the video.
        if not _is_complete(row):
            continue
        row["explicit_project"] = bool(candidate.get("explicit_project"))
        enriched.append(row)

    # Prefer high sales, then rents, then how often the buyer saw it, then photo count.
    enriched.sort(
        key=lambda p: (
            p["explicit_project"],
            p["sales_12m"],
            p["rent_contracts_12m"],
            p["appeared"],
            len(p["photos"]),
        ),
        reverse=True,
    )
    chosen = enriched[:DEFAULT_LIMIT]

    if len(chosen) < 2:
        return {
            "ok": False,
            "reason": (
                "Not enough complete projects yet. "
                "Each project needs photos, a recent sale, a recent rent, "
                "and a handover or ready status."
            ),
            "bedrooms": bedrooms,
            "communities": communities or [],
            "projects": chosen,
        }

    return {
        "ok": True,
        "reason": None,
        "bedrooms": bedrooms,
        "communities": sorted({p["community"] for p in chosen if p.get("community")}),
        "projects": chosen,
    }


def _candidates(
    history_or_projects: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(history_or_projects, list):
        return [p for p in history_or_projects if isinstance(p, dict)]
    if isinstance(history_or_projects, dict):
        projects = history_or_projects.get("projects") or []
        return [p for p in projects if isinstance(p, dict)]
    return []


def _shortlist(candidates: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Pick the best candidates to enrich, so we do not hit the DB for every search hit."""
    if len(candidates) <= cap:
        return candidates

    def rank(p: dict[str, Any]) -> tuple:
        has_key = 1 if (p.get("key") or "").strip() else 0
        has_photos = 1 if len(p.get("photos") or []) >= MIN_PHOTOS else 0
        appeared = int(p.get("appeared") or 0)
        last_seen = str(p.get("last_seen") or "")
        return (bool(p.get("explicit_project")), has_key, has_photos, appeared, last_seen)

    ranked = sorted(candidates, key=rank, reverse=True)
    return ranked[:cap]


def _apply_filters(
    candidates: list[dict[str, Any]],
    bedrooms: int | None,
    communities: list[str] | None,
    project_keys: list[str] | None,
) -> list[dict[str, Any]]:
    out = candidates

    if project_keys:
        wanted = {_norm(k) for k in project_keys if k}
        out = [
            p
            for p in out
            if _norm(p.get("key")) in wanted or _norm(p.get("name")) in wanted
        ]

    if communities:
        wanted = {_norm(c) for c in communities if c}
        out = [p for p in out if _norm(p.get("community")) in wanted]

    if bedrooms is not None:
        beds = int(bedrooms)
        filtered = []
        for p in out:
            lo = p.get("bedrooms_min")
            hi = p.get("bedrooms_max")
            # Unknown bedrooms: keep; pack will filter sales/rents by bedroom later.
            if lo is None and hi is None:
                filtered.append(p)
                continue
            lo_i = int(lo) if lo is not None else beds
            hi_i = int(hi) if hi is not None else beds
            if lo_i <= beds <= hi_i:
                filtered.append(p)
        out = filtered

    return out


def _enrich(candidate: dict[str, Any], bedrooms: int | None) -> dict[str, Any] | None:
    """Only catalog projects can be complete (sale + rent + handover + photos)."""
    key = (candidate.get("key") or "").strip() or None
    if not key:
        return None
    catalog = _from_catalog(key, bedrooms)
    if catalog is None:
        return None
    catalog["appeared"] = int(candidate.get("appeared") or 1)
    # Keep searched bedroom range for on-screen captions / narration.
    if candidate.get("bedrooms_min") is not None:
        catalog["bedrooms_min"] = candidate.get("bedrooms_min")
    if candidate.get("bedrooms_max") is not None:
        catalog["bedrooms_max"] = candidate.get("bedrooms_max")
    return catalog


def _is_complete(row: dict[str, Any]) -> bool:
    """
    True only when every video field is present and market facts are recent.

    Required: name, community, developer, 4+ photos, recent sale, recent rent,
    handover (or "ready"). Sale and rent dates must fall inside RECENT_MONTHS.
    """
    if not (row.get("name") or "").strip():
        return False
    if not (row.get("community") or "").strip():
        return False
    if not (row.get("developer") or "").strip():
        return False
    if len(row.get("photos") or []) < MIN_PHOTOS:
        return False
    if not (row.get("handover") or "").strip():
        return False

    sale = row.get("recent_sale")
    rent = row.get("rent")
    if not isinstance(sale, dict) or not isinstance(rent, dict):
        return False
    if sale.get("price") is None or not sale.get("date"):
        return False
    if rent.get("annual") is None or not rent.get("date"):
        return False
    if not _is_recent_date(sale.get("date")):
        return False
    if not _is_recent_date(rent.get("date")):
        return False
    return True


def _is_recent_date(value: Any) -> bool:
    if not value:
        return False
    try:
        from datetime import date, datetime, timedelta

        raw = str(value)[:10]
        d = datetime.strptime(raw, "%Y-%m-%d").date()
        return d >= date.today() - timedelta(days=RECENT_MONTHS * 31)
    except ValueError:
        return False


def _from_catalog(key: str, bedrooms: int | None) -> dict[str, Any] | None:
    rows = fetch(
        """
        select
          p.key,
          p.name,
          p.area as community,
          p.master_project as "masterCommunity",
          p.developer,
          p.status_group as status,
          p.location_key as "locationKey",
          p.project_number as "projectNumber",
          p.completion_date::text as "completionDate",
          p.offplan_completion_year as "completionYear",
          p.estimated_completion as "estimatedCompletion",
          p.delivery_status as "deliveryStatus"
        from public.propai_projects p
        where p.key = %s
        limit 1
        """,
        (key,),
    )
    if not rows:
        return None
    project = rows[0]
    location_key = project.get("locationKey")

    if not location_key:
        return None

    photos = _catalog_photos(key)
    if len(photos) < MIN_PHOTOS:
        return None

    name = (project.get("name") or "").strip()
    community = (project.get("masterCommunity") or project.get("community") or "").strip()
    developer = (project.get("developer") or "").strip()
    handover = _handover_label(project)
    if not name or not community or not developer or not handover:
        return None

    stats = fetch(
        """
        select
          coalesce(sales_12m, 0)::int as "sales12m",
          coalesce(
            (select count(*)::int from public.propai_rents r
             where r.location_key_project = x.location_key
               and r.deleted_at is null
               and r.contract_start_date >= current_date - interval '12 months'
               and r.contract_start_date <= current_date),
            0
          ) as "rents12m"
        from public.propai_project_prices x
        where x.location_key = %s
        limit 1
        """,
        (location_key,),
    )
    if stats:
        sales_12m = int(stats[0].get("sales12m") or 0)
        rent_12m = int(stats[0].get("rents12m") or 0)
    else:
        sales_12m = _count_sales_12m(location_key)
        rent_12m = _count_rents_12m(location_key)

    recent_sale = _latest_sale(location_key, bedrooms, project.get("projectNumber"))
    rent = _latest_rent(location_key, bedrooms, project.get("projectNumber"))
    if recent_sale is None or rent is None:
        return None

    return {
        "key": project.get("key"),
        "name": name,
        "community": community,
        "developer": developer,
        "photos": photos[:MAX_PHOTOS_PER_PROJECT],
        "recent_sale": recent_sale,
        "rent": rent,
        "handover": handover,
        "completion_date": project.get("completionDate"),
        "status": project.get("status"),
        "sales_12m": sales_12m,
        "rent_contracts_12m": rent_12m,
        "appeared": 1,
        "source": "catalog",
    }


def _from_saved(candidate: dict[str, Any]) -> dict[str, Any] | None:
    photos = [u for u in (candidate.get("photos") or []) if isinstance(u, str) and u.startswith("http")]
    # Prefer distinct URLs; allow fewer than MIN_PHOTOS only if we have at least 1.
    # Saved answers max out around 4; keep projects with 4, skip empty.
    if len(photos) < MIN_PHOTOS:
        return None

    price = candidate.get("price")
    recent_sale = None
    if price is not None:
        recent_sale = {
            "price": price,
            "date": None,
            "bedrooms": candidate.get("bedrooms_min"),
            "price_per_sqft": None,
        }

    return {
        "key": candidate.get("key"),
        "name": candidate.get("name") or "Project",
        "community": candidate.get("community"),
        "developer": candidate.get("developer"),
        "photos": photos[:MAX_PHOTOS_PER_PROJECT],
        "recent_sale": recent_sale,
        "rent": None,
        "handover": _handover_from_saved(candidate.get("handover_time"), candidate.get("status")),
        "status": candidate.get("status"),
        "sales_12m": 0,
        "rent_contracts_12m": 0,
        "appeared": int(candidate.get("appeared") or 1),
        "source": "saved",
    }


def _catalog_photos(key: str) -> list[str]:
    rows = fetch(
        """
        select object_name as "objectName"
        from public.propai_project_images
        where project_key = %s
        order by position
        limit %s
        """,
        (key, MAX_PHOTOS_PER_PROJECT),
    )
    return [f"{IMAGES_BASE_URL}/{r['objectName']}" for r in rows if r.get("objectName")]


def _count_sales_12m(location_key: str) -> int:
    rows = fetch(
        """
        select count(*)::int as n
        from public.propai_sales
        where location_key_project = %s
          and deleted_at is null
          and instance_date >= current_date - interval '12 months'
          and instance_date <= current_date
        """,
        (location_key,),
    )
    return int(rows[0]["n"]) if rows else 0


def _count_rents_12m(location_key: str) -> int:
    rows = fetch(
        """
        select count(*)::int as n
        from public.propai_rents
        where location_key_project = %s
          and deleted_at is null
          and contract_start_date >= current_date - interval '12 months'
          and contract_start_date <= current_date
        """,
        (location_key,),
    )
    return int(rows[0]["n"]) if rows else 0


def _latest_sale(
    location_key: str, bedrooms: int | None, project_number: str | None = None,
) -> dict[str, Any] | None:
    """Latest sale inside the recent window only. Older sales are ignored."""
    recent = f"interval '{RECENT_MONTHS} months'"
    rows = fetch(
            f"""
            select instance_date::text as date,
                   sale_price::float8 as price,
                   bedrooms::float8 as bedrooms,
                   price_per_sqft::float8 as "pricePerSqft"
            from public.propai_sales
            where (location_key_project = %s or (location_key_project is null and project_number = %s))
              and deleted_at is null
              and sale_price > 0
              and lower(coalesce(outlier_flag, '')) not in ('true', '1')
              and (%s::float8 is null or bedrooms = %s)
              and instance_date >= current_date - {recent}
              and instance_date <= current_date
            order by instance_date desc, updated_at desc nulls last, source, transaction_id
            limit 1
            """,
            (location_key, project_number, bedrooms, bedrooms),
    )
    if not rows:
        return _latest_sale(location_key, None, project_number) if bedrooms is not None else None
    r = rows[0]
    return {
        "price": r.get("price"),
        "date": r.get("date"),
        "bedrooms": r.get("bedrooms"),
        "price_per_sqft": r.get("pricePerSqft"),
    }


def _latest_rent(
    location_key: str, bedrooms: int | None, project_number: str | None = None,
) -> dict[str, Any] | None:
    """Latest registered rent for a contract already in effect."""
    recent = f"interval '{RECENT_MONTHS} months'"
    rows = fetch(
            f"""
            select coalesce(contract_registration_date, contract_start_date)::text as date,
                   contract_start_date::text as start_date,
                   contract_registration_date::text as registration_date,
                   contract_months::float8,
                   annual_amount::float8 as annual,
                   bedrooms::float8 as bedrooms,
                   rent_per_sqft::float8 as "rentPerSqft"
            from public.propai_rents
            where (location_key_project = %s or (location_key_project is null and project_number = %s))
              and deleted_at is null
              and annual_amount > 0
              and lower(coalesce(outlier_flag, '')) not in ('true', '1')
              and lower(coalesce(status, '')) not in ('cancelled', 'canceled', 'rejected', 'void', 'deleted')
              and (%s::float8 is null or bedrooms = %s)
              and coalesce(contract_registration_date, contract_start_date) >= current_date - {recent}
              and coalesce(contract_registration_date, contract_start_date) <= current_date
              and contract_start_date <= current_date
            order by coalesce(contract_registration_date, contract_start_date) desc,
                     contract_start_date desc, updated_at desc nulls last, pai_contract_id, unit_seq
            limit 1
            """,
            (location_key, project_number, bedrooms, bedrooms),
    )
    if not rows:
        return _latest_rent(location_key, None, project_number) if bedrooms is not None else None
    r = rows[0]
    return {
        "annual": r.get("annual"),
        "date": r.get("date"),
        "start_date": r.get("start_date"),
        "registration_date": r.get("registration_date"),
        "contract_months": r.get("contract_months"),
        "bedrooms": r.get("bedrooms"),
        "rent_per_sqft": r.get("rentPerSqft"),
    }


def _handover_label(project: dict[str, Any]) -> str | None:
    status = (project.get("status") or project.get("deliveryStatus") or "").lower()
    if "ready" in status or "secondary" in status:
        return "ready"

    completion = project.get("completionDate")
    if completion:
        # "2027-12-31 ..." -> "Dec 2027"
        try:
            from datetime import date, datetime

            raw = str(completion)[:10]
            d = datetime.strptime(raw, "%Y-%m-%d").date()
            if d <= date.today():
                return "ready"
            return d.strftime("%b %Y")
        except ValueError:
            pass

    year = project.get("completionYear")
    if year:
        try:
            y = int(year)
            from datetime import date

            if y < date.today().year:
                return "ready"
            return str(y)
        except (TypeError, ValueError):
            pass

    estimated = project.get("estimatedCompletion")
    if estimated:
        return str(estimated).strip() or None
    return None


def _handover_from_saved(handover_time: Any, status: Any) -> str | None:
    status_s = (status or "").lower() if isinstance(status, str) else ""
    if "ready" in status_s or "secondary" in status_s:
        return "ready"
    if not handover_time:
        return None
    try:
        from datetime import date, datetime

        raw = str(handover_time)[:10]
        d = datetime.strptime(raw, "%Y-%m-%d").date()
        if d <= date.today():
            return "ready"
        return d.strftime("%b %Y")
    except ValueError:
        return str(handover_time)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


# Quick local check:
#   python -c "from app.Viedo_generation.history import searched_projects; from app.Viedo_generation.pack import build_pack; import json; h=searched_projects('USER_ID'); print(json.dumps(build_pack(h, bedrooms=2, limit=4), indent=2, default=str)[:4000])"
