"""Fast turns for predict and recommend (no LLM). Search/agent path stays unchanged."""
from __future__ import annotations

from typing import Any

from app.answers.intent_route import classify_intent
from app.answers.predict import (
    fetch_areas_near_budget,
    fetch_bedroom_median,
    fetch_place_trend,
    format_snapshot_answer,
    format_trend_answer,
)
from app.answers.recommend import (
    format_afford_answer,
    format_open_areas,
    format_open_best,
    format_ranked_picks,
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


def _turn(question: str, answer: str, projects: list[dict], question_type: str = "search") -> dict:
    summary = _summary(projects)
    return {
        "question": question,
        "standalone_question": question,
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


def _search(filters: dict, limit: int = 6) -> tuple[list[dict], list[dict]]:
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
    if kind == "open_best":
        return _open_best_turn(question)
    if kind == "afford":
        return _afford_turn(question, route)
    if kind == "split":
        return _split_turn(question, route)
    if kind == "filtered":
        return _filtered_turn(question, route)
    return None


def _open_best_turn(question: str) -> dict:
    areas = []
    try:
        from app.listings.tools import top_areas_by_sales

        areas = top_areas_by_sales(5)
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
        answer = format_open_areas(areas)
    else:
        ranked = rank_projects(projects[:8], wanted_beds=wanted_bedrooms(question), attach_sales=False)
        answer = format_open_best(ranked)
    turn = _turn(question, answer, projects[:5], question_type="search")
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


def _afford_turn(question: str, route: dict) -> dict:
    place = route.get("subject") or "Dubai"
    budget = int(route.get("budget_aed") or 0)
    beds = int(route.get("bedrooms") or 1)
    median = fetch_bedroom_median(place, beds)
    matches, closest = [], []
    try:
        filters = {"communities": [place], "bedrooms": beds}
        if budget:
            filters["max_price"] = max(budget, int((median or {}).get("median") or budget))
        matches, closest = _search(filters, limit=4)
    except Exception:
        matches, closest = [], []
    nearby = []
    if budget and median and budget < median["median"] * 0.92:
        nearby = fetch_areas_near_budget(budget)
    shown = matches or closest
    answer = format_afford_answer(
        place, budget or 0, median, shown, nearby, bedrooms=beds
    )
    return _turn(question, answer, shown)


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


def _filtered_turn(question: str, route: dict) -> dict:
    place = route.get("subject")
    filters: dict[str, Any] = {}
    if place:
        filters["communities"] = [place]
    if route.get("bedrooms") is not None:
        filters["bedrooms"] = route["bedrooms"]
    if route.get("budget_aed"):
        filters["max_price"] = route["budget_aed"]
    matches, closest = _search(filters, limit=6)
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
    ranked = rank_projects(rows, wanted_beds=route.get("bedrooms"))
    bits = []
    if route.get("bedrooms") is not None:
        bits.append(f"a {route['bedrooms']}-bedroom")
    if place:
        bits.append(f"in {place}")
    if route.get("budget_aed"):
        bits.append(f"at up to AED {int(route['budget_aed']):,}")
    if months and not relaxed:
        bits.append("with an early handover")
    intro = "For " + " ".join(bits) + "," if bits else "From the matching projects,"
    answer = format_ranked_picks(ranked, intro=intro, relaxed=relaxed)
    ordered = []
    by_name = {str(row.get("name") or "").lower(): row for row in rows}
    for item in ranked.get("ranked") or []:
        row = by_name.get(str(item.get("name") or "").lower())
        if row and row not in ordered:
            ordered.append(row)
    return _turn(question, answer, ordered or rows, question_type="comparison")
