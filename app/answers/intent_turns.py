"""Fast turns for predict and recommend (no LLM). Search/agent path stays unchanged."""
from __future__ import annotations

import re
from typing import Any

from app.answers.intent_route import classify_intent, family_area_focus, prior_anchor_question
from app.answers.predict import (
    fetch_areas_near_budget,
    fetch_bedroom_median,
    fetch_place_trend,
    format_snapshot_answer,
    format_trend_answer,
)
from app.answers.recommend import (
    format_afford_answer,
    fetch_family_areas,
    format_family_areas,
    format_open_areas,
    format_open_best,
    format_ranked_picks,
    format_ready_offplan,
    fetch_ready_offplan_stats,
    format_yield_areas,
    format_split_answer,
    rank_projects,
    wanted_bedrooms,
    within_months,
)


def _summary(projects: list[dict], extra: dict | None = None) -> dict:
    payload = {
        "community": [],
        "projects": projects,
        "sub_communities": [],
        "listings": [],
        "brokers": [],
        "filter_no_match": False,
        "filter_evidence_projects": [],
        "project_total": len(projects),
    }
    if extra:
        payload.update(extra)
    return payload


def _shown_labels(conversation_memory) -> set[str]:
    names: set[str] = set()
    for item in getattr(conversation_memory, "active_candidates", None) or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "community", "area", "official_area"):
            value = item.get(key)
            if value:
                names.add(str(value).strip().lower())
    for turn in getattr(conversation_memory, "turns", None) or []:
        if not isinstance(turn, dict):
            continue
        for value in turn.get("candidate_names") or []:
            if value:
                names.add(str(value).strip().lower())
    return {name for name in names if name}


def _turn(
    question: str,
    answer: str,
    projects: list[dict],
    question_type: str = "search",
    standalone_question: str | None = None,
) -> dict:
    summary = _summary(projects)
    return {
        "question": question,
        "standalone_question": standalone_question or question,
        "question_type": question_type,
        "structured_summary": summary,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": question_type,
            "structured_data": {
                "community": [],
                "projects": projects,
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _search(
    filters: dict,
    limit: int = 6,
    skip_names: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    skip = {str(name).strip().lower() for name in (skip_names or set()) if name}
    if skip or limit > 10:
        from app.retrieval.propai_catalog import agent_projects

        raw: dict[str, Any] = {}
        if filters.get("communities"):
            raw["communities"] = filters["communities"]
        if filters.get("bedrooms") is not None:
            raw["bedrooms"] = filters["bedrooms"]
        if filters.get("max_price") is not None:
            raw["max_price"] = filters["max_price"]
        if filters.get("min_price") is not None:
            raw["min_price"] = filters["min_price"]
        if filters.get("handover_year") is not None:
            raw["handover_year"] = filters["handover_year"]
        rows, _ = agent_projects(raw, limit=min(limit + len(skip) + 4, 24), sort="cheapest")
        matches = [
            row for row in rows
            if str(row.get("name") or "").strip().lower() not in skip
        ][:limit]
        return matches, []

    from app.listings.tools import search_projects

    raw: dict[str, Any] = {"limit": limit, "sort": "cheapest"}
    if filters.get("communities"):
        raw["communities"] = filters["communities"]
    if filters.get("bedrooms") is not None:
        raw["bedrooms"] = filters["bedrooms"]
    if filters.get("max_price") is not None:
        raw["max_price_aed"] = filters["max_price"]
    if filters.get("handover_year") is not None:
        raw["handover_year"] = filters["handover_year"]
    if filters.get("property_types"):
        raw["property_types"] = filters["property_types"]
    result = search_projects(raw)
    matches = [card["row"] for card in (result.cards or []) if isinstance(card, dict) and card.get("row")]
    closest = []
    for row in (result.output or {}).get("closest_matches") or []:
        closest.append({
            "id": row.get("id"),
            "name": row.get("name"),
            "price": row.get("starting_price_aed") or row.get("price"),
            "community": row.get("community"),
            "handover_time": row.get("handover"),
            "status": row.get("status"),
            "bedroom_min": None,
            "bedroom_max": None,
        })
    return matches, closest


def predict_turn(question: str, conversation_memory=None, route: dict | None = None) -> dict | None:
    route = route or classify_intent(question, conversation_memory)
    if route.get("intent") != "predict":
        return None
    place = route.get("subject") or "Dubai"
    trend = fetch_place_trend(place)
    projects, closest = [], []
    try:
        projects, closest = _search({"communities": [place]}, limit=3)
    except Exception:
        projects, closest = [], []
    shown = projects or closest
    if route.get("kind") == "snapshot":
        answer = format_snapshot_answer(place, trend, shown)
        return _turn(question, answer, shown, question_type="fact")
    answer = format_trend_answer(place, trend, shown)
    return _turn(question, answer, shown, question_type="prediction")


def recommend_data_turn(question: str, conversation_memory=None, route: dict | None = None) -> dict | None:
    route = route or classify_intent(question, conversation_memory)
    if route.get("intent") != "recommend":
        return None
    kind = route.get("kind")
    if kind in {None, "shortlist"}:
        return None
    if not route.get("source_question"):
        route = dict(route)
        route["source_question"] = (
            prior_anchor_question(conversation_memory, question) or question
        )
    if kind == "family_areas":
        return _family_areas_turn(question, route, conversation_memory)
    if kind == "yield_areas":
        return _yield_areas_turn(question, route, conversation_memory)
    if kind == "ready_offplan":
        return _ready_offplan_turn(question, route, conversation_memory)
    if kind == "open_best":
        return _open_best_turn(question, route, conversation_memory)
    if kind == "afford":
        return _afford_turn(question, route, conversation_memory)
    if kind == "split":
        return _split_turn(question, route)
    if kind == "filtered":
        return _filtered_turn(question, route, conversation_memory)
    return None


def _open_best_turn(question: str, route: dict | None = None, conversation_memory=None) -> dict:
    skip = _shown_labels(conversation_memory) if (route or {}).get("more") else set()
    areas = []
    try:
        from app.listings.tools import top_areas_by_sales

        areas = top_areas_by_sales(10 if skip else 5)
        if skip:
            areas = [
                row for row in areas
                if str(row.get("area") or "").strip().lower() not in skip
            ][:5]
    except Exception:
        areas = []
    projects: list[dict] = []
    for area in areas[:3]:
        name = area.get("area")
        if not name:
            continue
        try:
            rows, _ = _search({"communities": [name]}, limit=1)
        except Exception:
            rows = []
        for row in rows:
            if row not in projects:
                projects.append(row)
    if not projects:
        matches, closest = _search({}, limit=5)
        projects = matches or closest
    if areas:
        answer = format_open_areas(
            areas,
            question=question,
            more=bool((route or {}).get("more")),
        )
    else:
        ranked = rank_projects(projects[:8], wanted_beds=wanted_bedrooms(question), attach_sales=False)
        answer = format_open_best(ranked, question=question)
    turn = _turn(
        question,
        answer,
        projects[:5],
        question_type="search",
        standalone_question=(route or {}).get("source_question") or question,
    )
    if areas:
        turn["structured_summary"]["community"] = [
            {
                "name": row.get("area"),
                "official_area": row.get("official_area"),
                "signals": {
                    "price_psf": row.get("median_price_per_sqft_aed"),
                    "sales_12m": row.get("sales_12m"),
                },
            }
            for row in areas
            if row.get("area")
        ]
    return turn


def _yield_areas_turn(question: str, route: dict | None = None, conversation_memory=None) -> dict:
    skip = _shown_labels(conversation_memory) if (route or {}).get("more") else set()
    place = (route or {}).get("subject")
    areas = []
    try:
        from app.listings.tools import top_areas_by_rent

        areas = top_areas_by_rent(10 if skip else 5)
        if skip:
            areas = [
                row for row in areas
                if str(row.get("area") or "").strip().lower() not in skip
            ][:5]
    except Exception:
        areas = []
    if (route or {}).get("more") and not areas:
        answer = (
            "Those were the main rental-activity areas I have. "
            "Share a budget and I’ll match buy options from the same data."
        )
        return _turn(
            question,
            answer,
            [],
            question_type="search",
            standalone_question=(route or {}).get("source_question") or question,
        )
    projects: list[dict] = []
    for area in areas[:5]:
        name = area.get("area")
        if not name:
            continue
        try:
            rows, _ = _search({"communities": [name]}, limit=1)
        except Exception:
            rows = []
        for row in rows:
            if row not in projects:
                projects.append(row)
        if len(projects) >= 5:
            break
    answer = format_yield_areas(areas, place=place)
    if (route or {}).get("more") and areas:
        answer = answer.replace(
            "From recorded rents, these Dubai areas currently have the most rental contracts:",
            "More areas with recorded rental contracts:",
            1,
        )
    turn = _turn(
        question,
        answer,
        projects[:5],
        question_type="search",
        standalone_question=(route or {}).get("source_question") or question,
    )
    if areas:
        turn["structured_summary"]["community"] = [
            {
                "name": row.get("area"),
                "official_area": row.get("official_area"),
                "signals": {
                    "rent_contracts": row.get("rent_contracts"),
                    "median_yearly_rent_aed": row.get("median_yearly_rent_aed"),
                    "median_sale_price_aed": row.get("median_sale_price_aed"),
                    "sales_12m": row.get("sales_12m"),
                },
            }
            for row in areas
            if row.get("area")
        ]
    return turn


def _project_handover_year(row: dict) -> int | None:
    raw = row.get("handover_time") or row.get("handover")
    if not raw:
        return None
    year = getattr(raw, "year", None)
    if year is not None:
        return int(year)
    match = re.search(r"(19|20)\d{2}", str(raw))
    return int(match.group(0)) if match else None


def _is_ready_project(row: dict, this_year: int) -> bool:
    year = _project_handover_year(row)
    if year is not None:
        return year <= this_year
    status = str(row.get("status") or row.get("project_status") or "")
    return bool(re.search(r"complet|ready|existing|finished", status, re.I))


def _is_offplan_project(row: dict, this_year: int) -> bool:
    year = _project_handover_year(row)
    if year is not None:
        return year > this_year
    status = str(row.get("status") or row.get("project_status") or "")
    return bool(re.search(r"off[\s-]?plan|on sale|upcoming|presale|announced", status, re.I))


def _project_pool(filters: dict, limit: int = 8, sort: str = "cheapest") -> list[dict]:
    from app.retrieval.propai_catalog import agent_projects

    raw: dict[str, Any] = {}
    if filters.get("communities"):
        raw["communities"] = filters["communities"]
    rows, _ = agent_projects(raw, limit=limit, sort=sort)
    return rows or []


def _ready_offplan_turn(question: str, route: dict | None = None, conversation_memory=None) -> dict:
    from datetime import date

    skip = _shown_labels(conversation_memory) if (route or {}).get("more") else set()
    place = (route or {}).get("subject")
    stats = {}
    try:
        stats = fetch_ready_offplan_stats(place)
    except Exception:
        stats = {}
    filters: dict[str, Any] = {}
    if place:
        filters["communities"] = [place]
    this_year = date.today().year
    try:
        early = _project_pool(filters, limit=8, sort="earliest_handover")
        late = _project_pool(filters, limit=8, sort="latest_handover")
    except Exception:
        early, late = [], []
    ready = [
        row for row in early
        if _is_ready_project(row, this_year)
        and str(row.get("name") or "").strip().lower() not in skip
    ][:2]
    offplan = [
        row for row in late
        if _is_offplan_project(row, this_year)
        and str(row.get("name") or "").strip().lower() not in skip
    ][:2]
    shown = ready + [row for row in offplan if row not in ready]
    try:
        answer = format_ready_offplan(stats, ready, offplan, place=place)
    except (OSError, OverflowError, ValueError, TypeError):
        answer = format_ready_offplan(stats, [], [], place=place)
    if (route or {}).get("more"):
        answer = answer.replace(
            "From recorded sales in the last 12 months:",
            "More ready and off-plan examples from the same split:",
            1,
        )
    return _turn(
        question,
        answer,
        shown[:5],
        question_type="comparison",
        standalone_question=(route or {}).get("source_question") or question,
    )


def _usable_family_project(row: dict) -> bool:
    raw = row.get("handover_time") or row.get("handover")
    if not raw:
        return True
    year = getattr(raw, "year", None)
    if year is None:
        match = re.search(r"(19|20)\d{2}", str(raw))
        year = int(match.group(0)) if match else None
    if year is not None and year < 2016:
        return False
    return True


def _family_areas_turn(question: str, route: dict | None = None, conversation_memory=None) -> dict:
    focus = (route or {}).get("focus") or family_area_focus(question)
    skip = _shown_labels(conversation_memory) if (route or {}).get("more") else set()
    areas = []
    try:
        areas = fetch_family_areas(10 if skip else 5, focus=focus)
        if skip:
            areas = [
                row for row in areas
                if str(row.get("area") or "").strip().lower() not in skip
            ][:5]
    except Exception:
        areas = []
    projects: list[dict] = []
    villa_first = bool(focus.get("want_villas") or focus.get("avoid_towers") or focus.get("kids"))
    for area in areas[:5]:
        name = area.get("area")
        if not name:
            continue
        filters: dict[str, Any] = {"communities": [name], "bedrooms": 3}
        if villa_first:
            filters["property_types"] = ["VILLA"]
        try:
            rows, closest = _search(filters, limit=3)
        except Exception:
            rows, closest = [], []
        chosen = next((row for row in (rows or closest) if _usable_family_project(row)), None)
        if chosen is None and villa_first:
            try:
                rows, closest = _search({"communities": [name], "bedrooms": 3}, limit=3)
            except Exception:
                rows, closest = [], []
            chosen = next((row for row in (rows or closest) if _usable_family_project(row)), None)
        if chosen and chosen not in projects:
            projects.append(chosen)
        if len(projects) >= 5:
            break
    if areas:
        answer = format_family_areas(areas, focus=focus)
    else:
        matches, closest = _search({"bedrooms": 3}, limit=5)
        projects = [row for row in (matches or closest) if _usable_family_project(row)][:5]
        ranked = rank_projects(projects[:8], wanted_beds=3, attach_sales=False)
        answer = format_ranked_picks(
            ranked,
            intro=(
                "Families usually need more bedrooms. "
                "These 3-bedroom options in our data are a practical starting point,"
            ),
        )
    if (route or {}).get("more") and areas:
        answer = "More family communities from the same ranking:\n\n" + "\n".join(
            answer.split("\n", 1)[-1].lstrip("\n").split("\n")
        )
    turn = _turn(
        question,
        answer,
        projects[:5],
        question_type="search",
        standalone_question=(route or {}).get("source_question") or question,
    )
    if areas:
        turn["structured_summary"]["community"] = [
            {
                "name": row.get("area"),
                "official_area": row.get("official_area"),
                "signals": {
                    "existing_villas": row.get("existing_villas"),
                    "family_sales": row.get("existing_family_sales") or row.get("family_sales"),
                    "villa_townhouse_sales": row.get("villa_townhouse_sales"),
                    "median_family_sale": row.get("median_family_sale"),
                },
            }
            for row in areas
            if row.get("area")
        ]
    return turn


def _afford_turn(question: str, route: dict, conversation_memory=None) -> dict:
    place = route.get("subject") or "Dubai"
    budget = int(route.get("budget_aed") or 0)
    beds = int(route.get("bedrooms") or 1)
    skip = _shown_labels(conversation_memory) if route.get("more") else set()
    median = fetch_bedroom_median(place, beds)
    matches, closest = [], []
    try:
        filters = {"communities": [place], "bedrooms": beds}
        if budget:
            filters["max_price"] = max(budget, int((median or {}).get("median") or budget))
        matches, closest = _search(filters, limit=4, skip_names=skip)
    except Exception:
        matches, closest = [], []
    nearby = []
    if budget and median and budget < median["median"] * 0.92:
        nearby = fetch_areas_near_budget(budget)
    shown = matches or closest
    answer = format_afford_answer(
        place, budget or 0, median, shown, nearby, bedrooms=beds
    )
    if route.get("more"):
        answer = answer.replace("For ", "More options for ", 1) if shown else (
            f"Those were the main {beds}-bedroom matches in {place} around that budget. "
            "I can raise the budget or check a nearby community."
        )
    return _turn(
        question,
        answer,
        shown,
        standalone_question=route.get("source_question") or question,
    )


def _split_turn(question: str, route: dict) -> dict:
    budget = route.get("budget_aed")
    studios, _ = _search(
        {"bedrooms": 0, "max_price": budget, "property_types": ["Apartments"]},
        limit=3,
    )
    villas, villa_close = _search(
        {"bedrooms": 4, "property_types": ["Villas"]},
        limit=3,
    )
    if not villas:
        villas = villa_close
    answer = format_split_answer(studios, villas, budget)
    return _turn(question, answer, (studios + villas)[:6])


def _filtered_turn(question: str, route: dict, conversation_memory=None) -> dict:
    place = route.get("subject")
    skip = _shown_labels(conversation_memory) if route.get("more") else set()
    filters: dict[str, Any] = {}
    if place:
        filters["communities"] = [place]
    if route.get("bedrooms") is not None:
        filters["bedrooms"] = route["bedrooms"]
    if route.get("budget_aed"):
        filters["max_price"] = route["budget_aed"]
    matches, closest = _search(filters, limit=6, skip_names=skip)
    months = route.get("months")
    exact = matches
    relaxed = None
    if months and matches:
        soon = [row for row in matches if within_months(row, int(months))]
        if soon:
            exact = soon
        else:
            relaxed = "I can keep the 3-month window and widen the area if you want."
            exact = matches
    rows = exact or closest
    bits = []
    if route.get("bedrooms") is not None:
        bits.append(f"a {route['bedrooms']}-bedroom")
    if place:
        bits.append(f"in {place}")
    if route.get("budget_aed"):
        bits.append(f"at up to AED {int(route['budget_aed']):,}")
    if months and not relaxed:
        bits.append("with an early handover")
    if route.get("more") and not rows:
        need = " ".join(bits) if bits else "this search"
        answer = (
            f"I already showed the closest matches for {need}. "
            "Share a nearby area or a slightly higher budget and I’ll keep going."
        )
        return _turn(
            question,
            answer,
            [],
            question_type="comparison",
            standalone_question=route.get("source_question") or question,
        )
    ranked = rank_projects(rows, wanted_beds=route.get("bedrooms"))
    if route.get("more"):
        intro = "More " + " ".join(bits) + "," if bits else "More matching projects,"
        if not relaxed:
            relaxed = "These are further options for the same request, beyond the ones already on screen."
    else:
        intro = "For " + " ".join(bits) + "," if bits else "From the matching projects,"
    answer = format_ranked_picks(ranked, intro=intro, relaxed=relaxed)
    ordered = []
    by_name = {str(row.get("name") or "").lower(): row for row in rows}
    for item in ranked.get("ranked") or []:
        row = by_name.get(str(item.get("name") or "").lower())
        if row and row not in ordered:
            ordered.append(row)
    return _turn(
        question,
        answer,
        ordered or rows,
        question_type="comparison",
        standalone_question=route.get("source_question") or question,
    )
