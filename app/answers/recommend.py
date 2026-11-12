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
    "more space": "Larger recorded size",
    "more bedrooms": "Wider bedroom range",
    "bedroom fit": "Strong bedroom fit for your request",
    "earlier handover": "Earlier handover timing",
    "readiness": "Strong readiness / availability status",
    "more recent sales activity": "Strong recent sales and transaction activity",
    "active rental market": "Active rental market for this project",
}


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
        return float(value)
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
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
    beds = _num(row.get("beds"), row.get("bedroom_min"))
    bed_max = _num(row.get("bedroom_max"))
    if beds is None and bed_max is not None:
        beds = bed_max
    elif beds is not None and bed_max is not None:
        beds = (beds + bed_max) / 2
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
        "size": size,
        "handover": _handover_stamp(row.get("handover_time") or row.get("handover")),
        "status_rank": _status_rank(row.get("project_status") or row.get("status")),
        "sales_count": _num(row.get("sales_count"), row.get("recent_sales_count")),
        "rent_count": _num(row.get("rent_count"), row.get("rental_count")),
        "median_rent": _num(row.get("median_rent"), row.get("median_annual_rent_aed")),
        "price_kind": row.get("price_kind"),
        "row": row,
    }


def attach_market_activity(projects: list[dict]) -> None:
    """Fill sales and rental activity from Prop-AI when missing. Best-effort."""
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
    if need_sales or need_size:
        try:
            rows = fetch(
                """
                select upper(project_name_prop_ai) as name,
                       count(*)::int as n,
                       percentile_cont(0.5) within group (
                           order by procedure_area_sqft
                       ) filter (where procedure_area_sqft > 0) as median_size
                from propai_sales
                where deleted_at is null and sale_price > 0
                  and instance_date >= current_date - interval '24 months'
                  and upper(project_name_prop_ai) = any(%s)
                group by 1
                """,
                (names,),
            )
        except Exception:
            rows = []
        by_name = {str(row["name"]): row for row in rows if row.get("name")}
        for project in projects:
            key = str(project.get("name") or "").strip().upper()
            row = by_name.get(key)
            if not row:
                continue
            if project.get("sales_count") is None and project.get("recent_sales_count") is None:
                project["sales_count"] = int(row["n"] or 0)
            if need_size and _num(project.get("size"), project.get("size_sqft"), project.get("size_min")) is None:
                if row.get("median_size"):
                    project["size_sqft"] = int(round(float(row["median_size"])))

    need_rent = any(
        project.get("rent_count") is None and project.get("rental_count") is None
        for project in projects
    )
    if not need_rent:
        return
    try:
        rents = fetch(
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


def _bed_fit_scores(beds: list[float | None], wanted: int | None) -> list[float | None]:
    if wanted is None:
        return _rank_key(beds, lower_is_better=False)
    scores = []
    for value in beds:
        if value is None:
            scores.append(None)
        else:
            gap = abs(value - wanted)
            scores.append(max(0.0, 1.0 - gap / max(wanted, 1)))
    if sum(1 for score in scores if score is not None) < 2:
        return [None] * len(beds)
    return scores


def _why_line(factor: str) -> str:
    return _WHY_LABELS.get(factor, factor.replace("_", " ").capitalize())


def rank_projects(
    projects: list[dict],
    *,
    wanted_beds: int | None = None,
    attach_sales: bool = True,
) -> dict[str, Any]:
    """Rank projects. Returns ranked list with scores and a recommendation payload."""
    rows = [project for project in projects if isinstance(project, dict)]
    if attach_sales:
        attach_market_activity(rows)
    items = [features(row) for row in rows]
    if len(items) < 2:
        winner = items[0] if items else None
        return {
            "ranked": items,
            "can_rank": False,
            "winner": winner,
            "alternatives": [],
            "why": [],
            "notes": ["Share one more project to compare side by side."],
            "recommendation": None if not winner else {
                "name": winner["name"],
                "id": winner.get("id"),
                "why": ["A solid option from the recorded project data."],
                "notes": [],
            },
        }

    price_pts = _rank_key([item["price"] for item in items], lower_is_better=True)
    size_pts = _rank_key([item["size"] for item in items], lower_is_better=False)
    bed_pts = _bed_fit_scores([item["beds"] for item in items], wanted_beds)
    hand_pts = _rank_key([item["handover"] for item in items], lower_is_better=True)
    status_pts = _rank_key([item["status_rank"] for item in items], lower_is_better=False)
    sales_pts = _rank_key([item["sales_count"] for item in items], lower_is_better=False)
    rent_pts = _rank_key([item["rent_count"] for item in items], lower_is_better=False)

    factor_sets = {
        "lower price": price_pts,
        "more space": size_pts,
        "bedroom fit" if wanted_beds is not None else "more bedrooms": bed_pts,
        "earlier handover": hand_pts,
        "readiness": status_pts,
        "more recent sales activity": sales_pts,
        "active rental market": rent_pts,
    }
    active_factors = [
        name for name, points in factor_sets.items()
        if any(point is not None for point in points)
    ]
    can_rank = len(active_factors) >= 2

    for index, item in enumerate(items):
        parts = []
        why = []
        for name, points in factor_sets.items():
            point = points[index]
            if point is None:
                continue
            parts.append(point)
            if point >= 0.75:
                why.append(name)
        item["score"] = round(100 * (sum(parts) / len(parts))) if parts else 50
        item["why"] = why
        item["factor_count"] = len(parts)

    items.sort(key=lambda item: (-(item.get("score") or 0), item["name"]))
    if not can_rank:
        return {
            "ranked": items,
            "can_rank": False,
            "winner": items[0] if items else None,
            "alternatives": items[1:4],
            "why": [],
            "notes": [
                "These options are close on the recorded details — useful next step is to compare handover and layout preference."
            ],
            "recommendation": {
                "name": items[0]["name"] if items else None,
                "id": items[0].get("id") if items else None,
                "why": ["A practical starting point from the projects in view."],
                "notes": [
                    "These options are close on the recorded details — useful next step is to compare handover and layout preference."
                ],
                "alternatives": [
                    {"name": item["name"], "id": item.get("id"), "score": item.get("score")}
                    for item in items[1:4]
                ],
            },
        }

    best = items[0]
    tied = [item for item in items if item["score"] == best["score"]]
    if len(tied) > 1:
        names = [item["name"] for item in tied]
        return {
            "ranked": items,
            "can_rank": True,
            "winner": None,
            "alternatives": tied,
            "why": [],
            "notes": [
                f"{names[0]} and {names[1]} both look strong — choose by the layout and community feel you prefer."
            ],
            "recommendation": {
                "name": None,
                "tied": names,
                "why": [
                    "Both lead on a balanced mix of price, size, sales activity and rental activity."
                ],
                "notes": [
                    f"{names[0]} and {names[1]} both look strong — choose by the layout and community feel you prefer."
                ],
            },
        }

    why = best.get("why") or active_factors[:3]
    notes = []
    alts = items[1:4]
    if alts:
        notes.append(
            "Also strong: "
            + ", ".join(item["name"] for item in alts)
            + "."
        )

    return {
        "ranked": items,
        "can_rank": True,
        "winner": best,
        "alternatives": alts,
        "why": why,
        "notes": notes,
        "recommendation": {
            "name": best["name"],
            "id": best.get("id"),
            "score": best.get("score"),
            "why": why,
            "notes": notes,
            "alternatives": [
                {"name": item["name"], "id": item.get("id"), "score": item.get("score")}
                for item in alts
            ],
        },
    }


def _detail_why(item: dict | None, factors: list[str]) -> list[str]:
    """Concrete, positive why lines using the winning project's numbers."""
    item = item or {}
    lines = []
    for factor in factors[:5]:
        if factor == "lower price" and item.get("price"):
            lines.append(f"Attractive recent price at AED {int(item['price']):,}")
        elif factor == "more space" and item.get("size"):
            lines.append(f"Larger recorded size around {int(item['size']):,} sqft")
        elif factor in {"more bedrooms", "bedroom fit"} and item.get("beds") is not None:
            beds = item["beds"]
            if beds == 0:
                lines.append("Useful studio layout in the recorded range")
            elif beds == 1:
                lines.append("Solid 1-bedroom fit in the recorded range")
            else:
                lines.append(f"Flexible bedroom range around {beds:g} bedrooms")
        elif factor == "earlier handover" and item.get("handover"):
            try:
                stamp = datetime.utcfromtimestamp(item["handover"]).strftime("%b %Y")
            except Exception:
                stamp = "an earlier date"
            lines.append(f"Earlier handover timing ({stamp})")
        elif factor == "more recent sales activity" and item.get("sales_count"):
            lines.append(
                f"Strong sales and transaction activity ({int(item['sales_count'])} recent sales)"
            )
        elif factor == "active rental market" and item.get("rent_count"):
            lines.append(
                f"Active rental market ({int(item['rent_count'])} rental contracts)"
            )
        elif factor == "readiness":
            lines.append("Strong readiness / availability status")
        else:
            lines.append(_why_line(factor))
    if not lines:
        lines.append("Strong balance of price, size, sales activity and rental activity")
    return lines


def format_footer(result: dict, *, intro: str | None = None) -> str:
    """Positive Acot Recommend block for answers."""
    lines = []
    if intro:
        lines.append(intro)
        lines.append("")
    rec = result.get("recommendation") or {}
    winner = result.get("winner")
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
        lines.append("**Acot Recommend:** These projects are ready to compare in more detail.")
        lines.append("**Why:**")
        lines.append("- Each option brings useful price, size or market activity signals.")
        return "\n".join(lines)

    lines.append(f"**Acot Recommend:** {name}")
    lines.append("**Why this is the strongest match:**")
    why = rec.get("why") or result.get("why") or []
    for item in _detail_why(winner, why):
        lines.append(f"- {item}")
    alts = rec.get("alternatives") or []
    if alts:
        alt_names = ", ".join(
            str(alt.get("name") or "") for alt in alts[:3] if alt.get("name")
        )
        if alt_names:
            lines.append(f"- Also strong: {alt_names}.")
    return "\n".join(lines)


def format_shortlist(result: dict, *, question: str | None = None, assumptions: str | None = None) -> str:
    """Ranked list + Acot Recommend footer for invest / which-is-best questions."""
    ranked = result.get("ranked") or []
    lines = []
    if assumptions:
        lines.append(assumptions)
        lines.append("")
    lines.append(
        "Best matches for investing and buying, based on price, size, sales and rentals:"
    )
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
        lines.append(" — ".join(bits))
    lines.append("")
    lines.append(format_footer(result))
    return "\n".join(lines).rstrip()


def format_compare_table(projects: list[dict], result: dict) -> str:
    """Side-by-side style text for named compare when not using structured snapshots."""
    items = result.get("ranked") or [features(row) for row in projects]
    by_name = {str(item["name"]).lower(): item for item in items}
    ordered = []
    for row in projects:
        key = str(row.get("name") or "").lower()
        if key in by_name and by_name[key] not in ordered:
            ordered.append(by_name[key])
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
        kind = item.get("price_kind") or ""
        label = "recent sale" if kind == "recent_sale" else "price"
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
        try:
            return datetime.utcfromtimestamp(stamp).strftime("%b %Y")
        except Exception:
            return "—"

    def count_cell(item, key):
        value = item.get(key)
        return "—" if value is None else str(int(value))

    rows = [
        header,
        sep,
        "| Price | " + " | ".join(cell(price_cell(item)) for item in ordered) + " |",
        "| Size | " + " | ".join(
            cell(f"{int(item['size']):,} sqft" if item.get("size") else "—") for item in ordered
        ) + " |",
        "| Bedrooms | " + " | ".join(cell(beds_cell(item)) for item in ordered) + " |",
        "| Recent sales | " + " | ".join(cell(count_cell(item, "sales_count")) for item in ordered) + " |",
        "| Rental contracts | " + " | ".join(cell(count_cell(item, "rent_count")) for item in ordered) + " |",
        "| Handover | " + " | ".join(cell(hand_cell(item)) for item in ordered) + " |",
        "| Score | " + " | ".join(cell(f"{item.get('score', 50)}%") for item in ordered) + " |",
    ]
    intro = f"Side-by-side comparison of {', '.join(names)} on price, size, sales and rentals:"
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
    row = row or {}
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
        points.append(f"{int(sales)} recent sales")
    hand = _handover_label(row)
    if hand:
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
    try:
        return datetime.utcfromtimestamp(stamp).strftime("%b %Y")
    except Exception:
        return None


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
) -> str:
    items = ranked.get("ranked") or []
    winner = ranked.get("winner") or (items[0] if items else None)
    shown = items[:3]
    if not winner:
        return intro.rstrip(",") + "."
    why = _detail_why(winner, winner.get("why") or ranked.get("why") or [])[:2]
    lines = [intro.rstrip(",") + ".", ""]
    for index, item in enumerate(shown, start=1):
        row = item.get("row") or item
        lines.append(_project_bullet(index, {**row, "name": item.get("name") or row.get("name")}))
    lines.append("")
    lines.append(f"{tag_project(winner)} is the strongest match on the recorded price, size and sales activity.")
    for point in why:
        lines.append(f"- {point}")
    if relaxed:
        lines.append(relaxed)
    else:
        lines.append("Share a bedroom tweak or a nearby area and I’ll tighten this list.")
    return "\n".join(lines)


def format_open_best(ranked: dict) -> str:
    items = (ranked.get("ranked") or [])[:3]
    if not items:
        return (
            "Share your budget and preferred area and I’ll match buy options from our data."
        )
    lines = [
        "A strong starting point in Dubai depends on budget and whether you want to live there or invest.",
        "",
        "From our project data, these are practical buy matches right now:",
        "",
    ]
    for index, item in enumerate(items, start=1):
        row = item.get("row") or item
        lines.append(_project_bullet(index, {**row, "name": item.get("name") or row.get("name")}))
    lines.append("")
    lines.append("What’s your budget, and is this to live in or to rent out?")
    return "\n".join(lines)


def format_open_areas(areas: list[dict]) -> str:
    if not areas:
        return (
            "Share your budget and preferred area and I’ll match buy options from our data."
        )
    lines = [
        "A strong starting point in Dubai depends on budget and whether you want to live there or invest.",
        "",
        "From recorded sales in the last 12 months, these areas currently look strong:",
        "",
    ]
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
    names = [str(row.get("name") or "").strip() for row in projects[:3] if row.get("name")]
    areas = [str(row.get("area") or "").title() for row in nearby[:3] if row.get("area")]
    label = _bed_label(bedrooms)
    lines = []
    if median and median.get("median"):
        centre = f"AED {int(median['median']):,}"
        lines.append(
            f"{label.capitalize()} sales in {place} recently centre around {centre} (last 12 months)."
        )
        lines.append("")
        if budget_aed >= median["median"] * 0.92:
            lines.append(f"AED {budget_aed:,} sits in that band.")
        else:
            lines.append(
                f"AED {budget_aed:,} sits below that typical {place} level, so nearby areas can be a better fit."
            )
        if names:
            lines.append("Closest options we have are on the cards below.")
        if areas and budget_aed < median["median"] * 0.92:
            lines.append(
                f"AED {budget_aed:,} also lines up with recorded prices in "
                + " and ".join(areas)
                + "."
            )
        lines.append("")
        lines.append(f"Want those {label}s, or {place} homes closest to this budget?")
        return "\n".join(lines)
    if names:
        return (
            f"Here are {place} options in our data near AED {budget_aed:,}. "
            "They are on the cards below.\n\n"
            "Want me to filter by bedrooms?"
        )
    if areas:
        return (
            f"AED {budget_aed:,} lines up with recorded prices in "
            + " and ".join(areas)
            + f". I can show {label}s there now."
        )
    return (
        f"Share a bedroom count or a nearby area and I’ll match options around AED {budget_aed:,}."
    )


def format_split_answer(studios: list[dict], villas: list[dict], budget_aed: int | None) -> str:
    budget = f" around AED {budget_aed:,}" if budget_aed else ""
    lines = [
        "Those are two different searches, so here are two short lists from our data.",
        "",
        f"Studios{budget}:",
    ]
    if studios:
        for index, row in enumerate(studios[:2], start=1):
            lines.append(_project_bullet(index, row))
    else:
        lines.append("- I can go deeper on studios once you pick this list.")
    lines.append("")
    lines.append("4-bedroom beach villas:")
    if villas:
        for index, row in enumerate(villas[:2], start=1):
            lines.append(_project_bullet(index, row))
    else:
        lines.append("- I can go deeper on villas once you pick this list.")
    lines.append("")
    lines.append("Which list should I go deeper on?")
    return "\n".join(lines)
