"""Step 2–3 for prediction: recorded last-12-month sales, never a future AED figure."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.database.postgres import fetch


def _change_pct(now, before) -> float | None:
    if now and before:
        return round(100 * (now - before) / before, 1)
    return None


def _place_keys(place: str) -> list[str]:
    keys = [place]
    bare = re.sub(r"^the\s+", "", place or "", flags=re.I).strip()
    if bare and bare.lower() != (place or "").lower():
        # Sales store "Springs", not "The Springs".
        keys.append(bare)
    try:
        from app.listings.tools import ALIASES

        alias = ALIASES.get((place or "").lower())
        if alias:
            keys.append(alias)
    except Exception:
        pass
    return list(dict.fromkeys(key for key in keys if key))


def _official_keys(place: str) -> list[str]:
    keys = list(_place_keys(place))
    try:
        from app.answers.area_comparison import official_area

        official = official_area(place)
        if official:
            keys.append(official)
    except Exception:
        pass
    return list(dict.fromkeys(key for key in keys if key))


def _sales_place_filter(place: str) -> tuple[str, list]:
    from app.retrieval.place_lock import sales_place_sql

    return sales_place_sql(_place_keys(place), _official_keys(place))


def fetch_place_trend(place: str) -> dict[str, Any] | None:
    """Median sale, AED/sq ft, YoY %, latest date for a Dubai place."""
    lock, lock_params = _sales_place_filter(place)
    if lock == "false":
        return None
    try:
        rows = fetch(
            f"""
            with s as (
              select instance_date, sale_price, price_per_sqft
              from public.propai_sales
              where deleted_at is null
                and coalesce(outlier_flag, '') <> 'True'
                and sale_price > 0
                and instance_date between current_date - interval '24 months' and current_date
                and {lock}
            ),
            cur as (select * from s where instance_date >= current_date - interval '12 months'),
            prev as (select * from s where instance_date < current_date - interval '12 months')
            select
              (select count(*)::int from cur) as sales,
              (select percentile_cont(0.5) within group (order by sale_price)::float8
                 from cur) as median_sale,
              (select percentile_cont(0.5) within group (order by price_per_sqft)::float8
                 from cur where price_per_sqft > 0) as psf,
              (select percentile_cont(0.5) within group (order by sale_price)::float8
                 from prev) as prev_sale,
              (select percentile_cont(0.5) within group (order by price_per_sqft)::float8
                 from prev where price_per_sqft > 0) as prev_psf,
              (select max(instance_date) from cur) as as_of,
              (select sale_price from cur order by instance_date desc limit 1) as latest_price
            """,
            tuple(lock_params),
        )
    except Exception:
        return None
    row = rows[0] if rows else {}
    if not row.get("sales"):
        return None
    as_of = row.get("as_of")
    if hasattr(as_of, "date"):
        as_of = as_of.date()
    change = _change_pct(row.get("median_sale"), row.get("prev_sale"))
    if change is None:
        change = _change_pct(row.get("psf"), row.get("prev_psf"))
    return {
        "place": place,
        "sales_12m": int(row["sales"]),
        "median_sale": int(round(row["median_sale"])) if row.get("median_sale") else None,
        "psf": int(round(row["psf"])) if row.get("psf") else None,
        "change_pct": change,
        "psf_change_pct": _change_pct(row.get("psf"), row.get("prev_psf")),
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else (str(as_of) if as_of else None),
        "latest_price": int(round(row["latest_price"])) if row.get("latest_price") else None,
    }


def fetch_bedroom_median(place: str, bedrooms: int) -> dict[str, Any] | None:
    lock, lock_params = _sales_place_filter(place)
    if lock == "false":
        return None
    try:
        rows = fetch(
            f"""
            select count(*)::int as n,
                   percentile_cont(0.5) within group (order by sale_price)::float8 as median,
                   max(instance_date) as as_of
            from public.propai_sales
            where deleted_at is null
              and coalesce(outlier_flag, '') <> 'True'
              and sale_price > 0
              and instance_date >= current_date - interval '12 months'
              and bedrooms between %s - 0.2 and %s + 0.2
              and {lock}
            """,
            (bedrooms, bedrooms, *lock_params),
        )
    except Exception:
        return None
    row = rows[0] if rows else {}
    if not row.get("n") or not row.get("median"):
        return None
    as_of = row.get("as_of")
    if hasattr(as_of, "date"):
        as_of = as_of.date()
    return {
        "place": place,
        "bedrooms": bedrooms,
        "sales": int(row["n"]),
        "median": int(round(row["median"])),
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else (str(as_of) if as_of else None),
    }


def fetch_areas_near_budget(budget_aed: int, limit: int = 3) -> list[dict]:
    lo, hi = int(budget_aed * 0.7), int(budget_aed * 1.35)
    try:
        rows = fetch(
            """
            select area, median_price::bigint as median_price, sales_12m
            from public.propai_area_stats
            where median_price between %s and %s and sales_12m > 0
            order by abs(median_price - %s)
            limit %s
            """,
            (lo, hi, budget_aed, limit),
        )
    except Exception:
        return []
    return [row for row in rows if row.get("area")]


def _pretty_date(value) -> str | None:
    if not value:
        return None
    text = str(value)[:10]
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        try:
            stamp = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return text
    return f"{stamp.day} {stamp.strftime('%b %Y')}"


def _aed(value) -> str:
    return f"AED {int(value):,}"


def _bed_label(bedrooms: int) -> str:
    if bedrooms <= 0:
        return "studio"
    if bedrooms == 1:
        return "1-bedroom"
    return f"{int(bedrooms)}-bedroom"


def attach_latest_project_sales(projects: list[dict] | None) -> list[dict]:
    """Replace 24-month medians with the newest recorded sale for each project."""
    rows = list(projects or [])
    names = [str(row.get("name") or "").strip().upper() for row in rows if row.get("name")]
    if not names:
        return rows
    try:
        stats = fetch(
            """
            select upper(project_name_prop_ai) as name,
                   max(instance_date) as sold_on,
                   (array_agg(sale_price order by instance_date desc))[1] as price,
                   min(bedrooms) filter (
                     where instance_date >= current_date - interval '12 months'
                       and bedrooms between 0 and 10
                   ) as bedroom_min,
                   max(bedrooms) filter (
                     where instance_date >= current_date - interval '12 months'
                       and bedrooms between 0 and 10
                   ) as bedroom_max
            from public.propai_sales
            where deleted_at is null
              and sale_price > 0
              and coalesce(outlier_flag, '') <> 'True'
              and upper(project_name_prop_ai) = any(%s)
            group by 1
            """,
            (names,),
        )
    except Exception:
        return rows
    by_name = {str(row["name"]): row for row in stats if row.get("name")}
    for project in rows:
        stat = by_name.get(str(project.get("name") or "").strip().upper())
        if not stat:
            continue
        if stat.get("price"):
            project["price"] = int(round(float(stat["price"])))
            project["price_kind"] = "latest_sale"
        if stat.get("sold_on"):
            project["last_sold_on"] = stat["sold_on"]
        if stat.get("bedroom_min") is not None:
            project["bedroom_min"] = stat["bedroom_min"]
        if stat.get("bedroom_max") is not None:
            project["bedroom_max"] = stat["bedroom_max"]
    rows.sort(
        key=lambda row: str(row.get("last_sold_on") or "0000-00-00")[:10],
        reverse=True,
    )
    return rows


def projects_for_place(place: str, projects: list[dict] | None) -> list[dict]:
    """Keep rows whose community is this place. A name like Silver Springs is a different community."""
    wanted = {key.upper() for key in _place_keys(place)}
    return [
        row for row in (projects or [])
        if str(row.get("community") or "").strip().upper() in wanted
    ]


def _project_detail_line(row: dict) -> str:
    name = str(row.get("name") or "This project").strip()
    bits = []
    developer = str(row.get("developer_name") or row.get("developer") or "").strip()
    if developer:
        bits.append(f"by {developer}")
    price = row.get("price")
    try:
        price = int(round(float(price))) if price else None
    except (TypeError, ValueError):
        price = None
    if price and price > 0:
        pretty = _pretty_date(row.get("last_sold_on"))
        kind = str(row.get("price_kind") or "")
        if kind == "latest_sale" or pretty:
            bits.append(f"latest sale AED {price:,}" + (f" on {pretty}" if pretty else ""))
        else:
            label = "recent sale" if kind == "recent_sale" else "from"
            bits.append(f"{label} AED {price:,}")
    lo, hi = row.get("bedroom_min"), row.get("bedroom_max")
    if lo is not None or hi is not None:
        lo = lo if lo is not None else hi
        hi = hi if hi is not None else lo
        if lo == hi:
            count = int(lo)
            bits.append("studio" if count == 0 else ("1 bedroom" if count == 1 else f"{count} bedrooms"))
        else:
            bits.append(f"{int(lo)} to {int(hi)} bedrooms")
    extra = (", " + ", ".join(bits)) if bits else ""
    return f"- **{name}**{extra}"


def format_place_details(
    place: str,
    trend: dict | None,
    projects: list[dict] | None = None,
    total: int | None = None,
) -> str:
    """Full community details from recorded sales and project rows. No model call."""
    shown = [row for row in (projects or []) if row.get("name")]
    count = int(total) if total is not None else len(shown)
    lines = [f"**{place}**"]
    if count:
        lines.append("")
        noun = "project" if count == 1 else "projects"
        if shown and count > len(shown):
            lines.append(f"I found {count} {noun} in {place}. Here are {len(shown)} of them.")
        else:
            lines.append(f"I found {count} {noun} in {place}.")
    if trend and (trend.get("median_sale") or trend.get("psf") or trend.get("sales_12m")):
        pretty = _pretty_date(trend.get("as_of"))
        lines.append("")
        lines.append(
            "Last 12 months of recorded sales"
            + (f" (data through {pretty})" if pretty else "")
            + ":"
        )
        lines.append("")
        if trend.get("median_sale"):
            lines.append(f"- Median sale price: {_aed(trend['median_sale'])}")
        if trend.get("psf"):
            lines.append(f"- Median price per sq ft: AED {int(trend['psf']):,}")
        if trend.get("sales_12m"):
            lines.append(f"- Recorded sales: {int(trend['sales_12m']):,}")
        if trend.get("change_pct") is not None:
            pct = trend["change_pct"]
            direction = "higher" if pct >= 0 else "lower"
            lines.append(f"- About {abs(pct):g}% {direction} than the previous 12 months")
        if trend.get("latest_price"):
            extra = f" on {pretty}" if pretty else ""
            lines.append(f"- Latest recorded sale: {_aed(trend['latest_price'])}{extra}")
        elif pretty:
            lines.append(f"- Latest sale date in this set: {pretty}")
    if shown:
        lines.append("")
        lines.append("Projects on record:")
        lines.append("")
        lines.extend(_project_detail_line(row) for row in shown)
    if len(lines) == 1:
        lines.append("")
        lines.append(f"I don't have recorded sales for {place} in the last 12 months.")
    return "\n".join(lines)


def format_snapshot_answer(
    place: str,
    trend: dict | None,
    projects: list[dict] | None = None,
    bedroom_stats: list[dict] | None = None,
) -> str:
    """Median sale, AED/sq ft, and data date for a named place. Never invent."""
    names = [str(row.get("name") or "").strip() for row in (projects or []) if row.get("name")][:3]
    stats = [row for row in (bedroom_stats or []) if isinstance(row, dict)]
    if stats:
        lines = [f"{place}, recorded sales by bedroom (last 12 months):", ""]
        for row in stats:
            beds = int(row.get("bedrooms") or 0)
            label = _bed_label(beds)
            if row.get("median"):
                line = f"- {label}: median sale {_aed(row['median'])}"
                extra = []
                if row.get("sales"):
                    extra.append(f"{int(row['sales']):,} sales")
                pretty = _pretty_date(row.get("as_of"))
                if pretty:
                    extra.append(f"through {pretty}")
                if extra:
                    line += f" ({', '.join(extra)})"
                lines.append(line)
            else:
                lines.append(f"- {label}: no recorded sales in this set.")
        if names:
            lines.append("")
            lines.append("Example projects from this area are on the cards below.")
        return "\n".join(lines)
    if trend and (trend.get("median_sale") or trend.get("psf")):
        pretty = _pretty_date(trend.get("as_of"))
        lines = [f"{place}, last 12 months of recorded sales" + (f" (data through {pretty})" if pretty else "") + ":"]
        lines.append("")
        if trend.get("median_sale"):
            lines.append(f"- Median sale price: {_aed(trend['median_sale'])}")
        if trend.get("psf"):
            lines.append(f"- Median price per sq ft: AED {int(trend['psf']):,}")
        if pretty:
            lines.append(f"- Latest sale date in this set: {pretty}")
        if names:
            lines.append("")
            lines.append("Example projects from this area are on the cards below.")
        return "\n".join(lines)
    if names:
        return (
            f"{place} projects are on the cards below. "
            "I can quote the latest recorded median as soon as sales for that area load."
        )
    return f"Share another Dubai area and I’ll quote the last-12-month median from recorded sales."


def format_trend_answer(place: str, trend: dict | None, projects: list[dict] | None = None) -> str:
    """Lead with recorded history. Never invent a next-year price. Never print ids."""
    names = [str(row.get("name") or "").strip() for row in (projects or []) if row.get("name")][:3]
    if trend and trend.get("median_sale"):
        pretty = _pretty_date(trend.get("as_of"))
        through = f" through {pretty}" if pretty else ""
        lines = [
            f"Latest recorded picture for {place}{through}:",
            "",
            f"- Median sale: {_aed(trend['median_sale'])}",
        ]
        if trend.get("psf"):
            lines.append(f"- About AED {int(trend['psf']):,} per sq ft")
        if trend.get("change_pct") is not None:
            pct = trend["change_pct"]
            direction = "higher" if pct >= 0 else "lower"
            lines.append(f"- About {abs(pct):g}% {direction} than the year before")
        lines.append("")
        lines.append(
            "That is the last 12 months of recorded sales, not a guaranteed price for next year."
        )
        if names:
            lines.append("Example projects from this area are on the cards below.")
        lines.append(f"I can show recent {place} 1-beds or 2-beds next.")
        return "\n".join(lines)
    if names:
        return (
            f"I can help with {place} from the projects we have. "
            "They are on the cards below.\n\n"
            "Want these, or a nearby area with recorded sales?"
        )
    return (
        f"I can help with {place} from the projects and sales in our data. "
        "Share a budget or bedroom count and I’ll match options."
    )
