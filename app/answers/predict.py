"""Step 2–3 for prediction: recorded last-12-month sales, never a future AED figure."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.database.postgres import fetch


def _change_pct(now, before) -> float | None:
    if now and before:
        return round(100 * (now - before) / before, 1)
    return None


def _place_keys(place: str) -> list[str]:
    keys = [place]
    try:
        from app.answers.area_comparison import official_area

        official = official_area(place)
        if official:
            keys.append(official)
    except Exception:
        pass
    return list(dict.fromkeys(keys))


def fetch_place_trend(place: str) -> dict[str, Any] | None:
    """Median sale, AED/sq ft, YoY %, latest date for a Dubai place."""
    names = [key.upper() for key in _place_keys(place) if key]
    if not names:
        return None
    try:
        rows = fetch(
            """
            with s as (
              select instance_date, sale_price, price_per_sqft
              from public.propai_sales
              where deleted_at is null
                and coalesce(outlier_flag, '') <> 'True'
                and sale_price > 0
                and instance_date between current_date - interval '24 months' and current_date
                and (
                  upper(area_name_prop_ai) = any(%s)
                  or upper(master_project_prop_ai) = any(%s)
                )
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
              (select max(instance_date) from cur) as as_of
            """,
            (names, names),
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
    }


def fetch_bedroom_median(place: str, bedrooms: int) -> dict[str, Any] | None:
    names = [key.upper() for key in _place_keys(place) if key]
    if not names:
        return None
    try:
        rows = fetch(
            """
            select count(*)::int as n,
                   percentile_cont(0.5) within group (order by sale_price)::float8 as median,
                   max(instance_date) as as_of
            from public.propai_sales
            where deleted_at is null
              and coalesce(outlier_flag, '') <> 'True'
              and sale_price > 0
              and instance_date >= current_date - interval '12 months'
              and bedrooms between %s - 0.2 and %s + 0.2
              and (
                upper(area_name_prop_ai) = any(%s)
                or upper(master_project_prop_ai) = any(%s)
              )
            """,
            (bedrooms, bedrooms, names, names),
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


def format_snapshot_answer(place: str, trend: dict | None, projects: list[dict] | None = None) -> str:
    """Median sale, AED/sq ft, and data date for a named place. Never invent."""
    names = [str(row.get("name") or "").strip() for row in (projects or []) if row.get("name")][:3]
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
