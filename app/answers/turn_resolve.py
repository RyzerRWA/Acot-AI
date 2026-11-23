"""Phase 5–7: understand a turn into action + slots (not phrase routes in run.py).

Preferred path: LLM JSON resolve (any wording → same schema).
Offline fallback: classify_intent + slot parsers (no new demo-sentence regexes).
Phase 7: confidence + clarify when understanding is too weak to execute safely.

Fast SQL / tools still execute; this module only decides what to run.
"""
from __future__ import annotations

import json
from datetime import date
import re
from typing import Any, Callable

from app.answers.clarify import enrich_resolution_from_state, score_confidence
from app.answers.intent_route import (
    classify_intent,
    extract_place,
    goal_for_route,
    is_incomplete_home_browse,
    parse_bedrooms,
    parse_budget_aed,
)
from app.memory.dialogue_state import normalize_dialogue_state

_ACTIONS = frozenset({
    "data_freshness",
    "handovers",
    "rent_stat",
    "area_market",
    "area_rank",
    "top_projects",
    "project_detail",
    "guidance",
    "search",
    "project_count",
    "sales_list",
    "sales_count",
    "sales_stat",
    "developer_rank",
    "attribute",
    "recommend",
    "agents",
    "predict",
    "general",
    "clarify",
    "small_talk",
})

_SALES_SIGNAL = re.compile(
    r"\b(?:sales?|sold|transactions?|deals?)\b",
    re.I,
)
# Price extreme / aggregate intent (after typo repair). Not a synonym bank for
# "transaction" — context decides whether this is a sales_stat.
_SALES_STAT_SIGNAL = re.compile(
    r"\b(?:cheapest|lowest|least\s+expensive|highest|most\s+expensive|priciest|"
    r"average|median|mean|minimum|maximum)\b",
    re.I,
)
_WHICH_AMONG_SHOWN = re.compile(
    r"\b(?:which\s+one|which\s+(?:project|of\s+(?:these|those|them))|"
    r"among\s+(?:these|those)|(?:of|from)\s+(?:these|those|them))\b",
    re.I,
)


def _stat_kind_from_text(text: str) -> str | None:
    match = _SALES_STAT_SIGNAL.search(text or "")
    if not match:
        return None
    word = match.group(0).lower()
    if word in {"highest", "priciest", "maximum"} or word.startswith("most"):
        return "highest"
    if word in {"average", "median", "mean"}:
        return "average"
    return "cheapest"


def _prior_sales_thread(conversation_memory) -> bool:
    if conversation_memory is None:
        return False
    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state() or {}
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    return str((state or {}).get("action") or "") in {"sales_list", "sales_count"}


def _on_screen_project_count(conversation_memory) -> int:
    if conversation_memory is None:
        return 0
    items = getattr(conversation_memory, "active_candidates", None) or []
    return sum(
        1
        for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    )


def _has_shown_or_sale_projects(conversation_memory) -> bool:
    if _on_screen_project_count(conversation_memory) >= 1:
        return True
    if conversation_memory is None:
        return False
    stubs = getattr(conversation_memory, "last_sale_projects", None) or []
    return any(isinstance(item, dict) and item.get("name") for item in stubs)


def _is_shown_attribute_ask(question: str | None, conversation_memory=None) -> bool:
    """Developer/price/handover about the current shortlist (with or without 'these')."""
    from app.retrieval.hybrid.question_text import (
        is_attribute_phrase,
        is_group_attribute_question,
    )

    text = question or ""
    if is_group_attribute_question(text):
        return True
    if not is_attribute_phrase(text):
        return False
    if not _has_shown_or_sale_projects(conversation_memory):
        return False
    # Real project browses / agents / sales stay on their own paths.
    if re.search(
        r"\b(?:show|find|list)\s+(?:me\s+)?(?:\d+\s+)?(?:projects?|apartments?|"
        r"villas?|townhouses?|flats?)\b",
        text,
        re.I,
    ):
        return False
    if _AGENTS_SIGNAL.search(text) or _SALES_SIGNAL.search(text):
        return False
    return True
_PROJECT_COUNT_SIGNAL = re.compile(
    r"\bhow many\b.*\bprojects?\b|\bproject count\b|\bnumber of projects\b",
    re.I,
)
_SALE_COUNT_SIGNAL = re.compile(
    r"\bhow many\b.*\b(?:sold|sales?|transactions?)\b|"
    r"\b(?:sold|sales?)\b.*\bhow many\b",
    re.I,
)
_SEARCH_SIGNAL = re.compile(
    r"\b(?:show|find|list|browse|get)\b|"
    r"\b(?:projects?|propert(?:y|ies)|apartments?|villas?|townhouses?|flats?)\b",
    re.I,
)
_AGENTS_SIGNAL = re.compile(r"\b(?:agents?|brokers?)\b", re.I)
_RENT_WORD = re.compile(r"\b(?:rent|rents|rental\s+(?:price|cost|rate)s?|to\s+rent|lease|yearly\s+rent)\b", re.I)
_HANDOVER_WORD = re.compile(r"\b(?:handovers?|handed\s+over|hand(?:ing)?\s+over|delivered|completed|completions?)\b", re.I)
_RECENT_WORD = re.compile(r"\b(?:recent(?:ly)?|latest|lately|newly|just|last)\b", re.I)
_UPCOMING_WORD = re.compile(r"\b(?:upcoming|next|soon(?:est)?|nearest|coming|future)\b", re.I)
_POINTS_AT_SHOWN = re.compile(r"\b(?:these|those|them|their|each|this\s+project|that\s+project)\b", re.I)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_month_year(text: str) -> tuple[int | None, int | None]:
    month = None
    year = None
    lower = (text or "").lower()
    for name, value in _MONTHS.items():
        if re.search(rf"\b{re.escape(name)}\b", lower):
            month = value
            break
    year_match = re.search(r"\b(20\d{2})\b", text or "")
    if year_match:
        year = int(year_match.group(1))
    if month is None and year is None:
        relative = re.search(r"\b(last|previous|this|current)\s+month\b", lower)
        if relative:
            today = date.today()
            back = 1 if relative.group(1) in ("last", "previous") else 0
            index = today.year * 12 + today.month - 1 - back
            year, month = divmod(index, 12)
            month += 1
    return month, year


# Strip sale/browse filler so leftover tokens like "AC3" can resolve as a place/project.
_SALES_PLACE_FILLER = re.compile(
    r"\b(?:show|me|tell|find|list|give|the|top|best|latest|recent|last|newest|"
    r"recorded|please|can|you|could|what|are|is|any|all|how|many|"
    r"transactions?|transcations?|sales?|sold|details?|info|information|about|"
    r"for|in|at|of|dubai|apartments?|bedroom|bedrooms?|studio|studios?)\b",
    re.I,
)


def sales_place_from_text(text: str | None) -> str | None:
    """Area or project named in a sales/transaction ask (including short codes)."""
    asked = text or ""
    try:
        from app.retrieval.propai_catalog import (
            place_named,
            project_named_in_question,
            resolve_dubai_place,
        )

        # Prefer a full project title ("The Woods at Sobha Sanctuary") over a
        # shorter master/community leftover ("Sobha Sanctuary").
        project = project_named_in_question(asked)
        if project:
            return project
        named = extract_place(asked) or place_named(asked)
        if named and str(named).strip().lower() != "dubai":
            return named
        leftover = " ".join(_SALES_PLACE_FILLER.sub(" ", asked).split()).strip(" .?")
        if len(leftover) < 3:
            return None
        return resolve_dubai_place(leftover)
    except Exception:
        named = extract_place(asked)
        if named and str(named).strip().lower() != "dubai":
            return named
        return None


def has_explicit_sales_place_token(text: str | None) -> bool:
    """True when the ask leaves a place/project token after sale-filler stripping."""
    leftover = " ".join(_SALES_PLACE_FILLER.sub(" ", text or "").split()).strip(" .?")
    return len(leftover) >= 3


def _coerce_place_scoped_recommend(
    resolution: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Best projects in Marina must stay place-scoped — never citywide open_best."""
    out = dict(resolution)
    place = out.get("place") or extract_place(question)
    if place:
        place = str(place).strip() or None
    if not place or place.lower() == "dubai":
        return out
    out["place"] = place
    if out.get("action") == "recommend" and out.get("kind") in {
        None, "", "open_best", "open_buy",
    }:
        out["kind"] = "filtered"
    # open_buy goal is only for citywide "no area yet" answers.
    if out.get("goal") == "open_buy" and out.get("kind") == "filtered":
        out["goal"] = "invest" if not out.get("budget_aed") else "budget"
    return out


def _apply_agents_override(resolution: dict[str, Any], question: str) -> dict[str, Any]:
    """'Top agents for Dubai Hills' must never become a project browse."""
    text = question or ""
    if not _AGENTS_SIGNAL.search(text):
        return resolution
    if _SALES_SIGNAL.search(text):
        return resolution
    # Projects/properties wording without agents already returned above.
    out = dict(resolution or {})
    out["action"] = "agents"
    if not out.get("place"):
        out["place"] = extract_place(text)
    out["kind"] = None
    out["goal"] = None
    out["confidence"] = max(float(out.get("confidence") or 0), 0.9)
    return out


def _apply_conflict_split(resolution: dict[str, Any], question: str) -> dict[str, Any]:
    """Studio + villa / multi-bed at once → recommend split, never a single search."""
    from app.answers.intent_route import is_split_constraints, parse_budget_aed

    if not is_split_constraints(question):
        return resolution
    out = dict(resolution)
    out["action"] = "recommend"
    out["kind"] = "split"
    out["goal"] = "open_buy"
    out["conflict"] = True
    # Do not invent a place or pick one bedroom — that was the live bug.
    out["place"] = None
    out["beds"] = None
    if out.get("budget_aed") is None:
        out["budget_aed"] = parse_budget_aed(question)
    out["clarify_ask"] = None
    out["confidence"] = max(float(out.get("confidence") or 0.85), 0.9)
    return out


def _clean_resolution(data: dict[str, Any], question: str) -> dict[str, Any]:
    action = str(data.get("action") or "general").strip().lower()
    if action not in _ACTIONS:
        action = "general"
    beds = data.get("beds")
    try:
        beds = int(beds) if beds is not None else None
    except (TypeError, ValueError):
        beds = None
    budget = data.get("budget_aed")
    try:
        budget = int(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None
    month = data.get("month")
    year = data.get("year")
    try:
        month = int(month) if month is not None else None
    except (TypeError, ValueError):
        month = None
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    goal = data.get("goal")
    if goal is not None:
        goal = str(goal).strip().lower() or None
    place = data.get("place")
    if place is not None:
        place = str(place).strip() or None
    kind = data.get("kind")
    if kind is not None:
        kind = str(kind).strip().lower() or None
    clarify_ask = data.get("clarify_ask")
    if clarify_ask is not None:
        clarify_ask = str(clarify_ask).strip() or None
    # sales_stat details come from the model, not from phrase regexes.
    stat = data.get("stat")
    if stat is not None:
        stat = str(stat).strip().lower() or None
        stat = {
            "cheapest": "cheapest", "lowest": "cheapest", "min": "cheapest",
            "highest": "highest", "most_expensive": "highest", "max": "highest",
            "average": "average", "median": "average", "mean": "average",
        }.get(stat)
    project = data.get("project")
    if project is not None:
        project = str(project).strip() or None
    developer = data.get("developer")
    if developer is not None:
        developer = str(developer).strip() or None
    months = data.get("months")
    try:
        months = max(1, min(int(months), 60)) if months is not None else None
    except (TypeError, ValueError):
        months = None
    confidence = data.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    cleaned = _coerce_place_scoped_recommend(
        {
            "action": action,
            "goal": goal,
            "place": place,
            "beds": beds,
            "budget_aed": budget,
            "month": month,
            "year": year,
            "ready_only": bool(data.get("ready_only")),
            "prefer_cheaper": bool(data.get("prefer_cheaper")),
            "kind": kind,
            "confidence": confidence,
            "clarify_ask": clarify_ask,
            "source_question": question,
            "source": data.get("source") or "llm",
            "route": data.get("route"),
            "stat": stat,
            "project": project,
            "developer": developer,
            "months": months,
        },
        question,
    )
    # Deterministic: conflicting product types beat a wrong LLM "search".
    # Agents/brokers beat a wrong LLM "search" for the same place.
    return _apply_agents_override(_apply_conflict_split(cleaned, question), question)


def offline_resolve_turn(
    question: str | None,
    conversation_memory=None,
) -> dict[str, Any] | None:
    """Map a turn to action+slots using classifiers/parsers — not run.py phrase lists."""
    try:
        from app.retrieval.hybrid.question_text import repair_typed_question

        text = repair_typed_question((question or "").strip())
    except Exception:
        text = (question or "").strip()
    if not text:
        return None

    # Price extreme over sales evidence — driven by meaning + dialogue context,
    # not by expanding synonym regexes for every typo of "transaction".
    # Typos are repaired above; prior sales_list makes "which one is cheapest?"
    # enough without repeating the word "sale/transaction".
    stat = _stat_kind_from_text(text)
    in_sales_thread = _prior_sales_thread(conversation_memory)
    comparing = bool(_WHICH_AMONG_SHOWN.search(text)) and (
        in_sales_thread or _on_screen_project_count(conversation_memory) >= 2
    )
    if (
        stat
        and (
            _SALES_SIGNAL.search(text)
            or in_sales_thread
            or comparing
        )
        and not _SALE_COUNT_SIGNAL.search(text)
        and not _AGENTS_SIGNAL.search(text)
    ):
        parsed_beds = parse_bedrooms(text)
        return _clean_resolution(
            {
                "action": "sales_stat",
                "place": extract_place(text),
                "beds": parsed_beds[0] if parsed_beds else None,
                "stat": stat,
                "confidence": 0.92,
                "source": "offline",
            },
            text,
        )

    route = classify_intent(text, conversation_memory)
    place = route.get("subject") if route.get("subject") != "on_screen" else None
    place = place or extract_place(text)
    beds = route.get("bedrooms")
    if isinstance(beds, list):
        beds = beds[0] if beds else None
    if beds is None:
        parsed = parse_bedrooms(text)
        beds = parsed[0] if parsed else None
    budget = route.get("budget_aed")
    if budget is None:
        budget = parse_budget_aed(text)
    month, year = _parse_month_year(text)
    ready_only = bool(re.search(r"\b(?:ready(?:\s+only)?|only\s+ready)\b", text, re.I))

    if route.get("intent") == "recommend":
        return _clean_resolution(
            {
                "action": "recommend",
                "goal": goal_for_route(route),
                "place": place,
                "beds": beds,
                "budget_aed": budget,
                "ready_only": ready_only,
                "kind": route.get("kind"),
                "route": route,
                "source": "offline",
            },
            text,
        )

    if route.get("intent") == "predict":
        return _clean_resolution(
            {
                "action": "predict",
                "place": place,
                "beds": beds,
                "kind": route.get("kind"),
                "route": route,
                "source": "offline",
            },
            text,
        )

    if _AGENTS_SIGNAL.search(text) and not _SALES_SIGNAL.search(text):
        return _clean_resolution(
            {
                "action": "agents",
                "place": place,
                "source": "offline",
            },
            text,
        )

    # "Average rent for a 1-bed in JVC": rent, never a sale statistic. Near-certain, so it also holds when
    # the model is unavailable (rate limits) and only these rules answer.
    if _RENT_WORD.search(text) and place and str(place).strip().lower() != "dubai" and not _AGENTS_SIGNAL.search(text):
        lowered = text.lower()
        rent_stat = ("cheapest" if re.search(r"\b(?:cheap|cheapest|lowest|minimum)\b", lowered)
                     else "highest" if re.search(r"\b(?:highest|most expensive|maximum|priciest)\b", lowered)
                     else "average")
        parsed_beds = parse_bedrooms(text)
        return _clean_resolution(
            {"action": "rent_stat", "place": place, "beds": parsed_beds[0] if parsed_beds else None,
             "stat": rent_stat, "confidence": 0.92, "source": "offline"},
            text,
        )

    # "Which projects were handed over recently?" / "upcoming handovers in JVC": a ranking, not an attribute
    # of the cards on screen ("their handover dates" still is one).
    if _HANDOVER_WORD.search(text) and not _POINTS_AT_SHOWN.search(text):
        when = "upcoming" if _UPCOMING_WORD.search(text) else "recent" if _RECENT_WORD.search(text) else None
        if when:
            return _clean_resolution(
                {"action": "handovers", "kind": when, "place": place, "confidence": 0.92, "source": "offline"},
                text,
            )

    # "Who is the developer of each of these projects?" / "developer details in JVC"
    # with cards on screen — never clarify or re-browse the community.
    if _is_shown_attribute_ask(text, conversation_memory):
        # Near-certain only when the ask points at the shown set ("these", "their", "each") or names
        # a project attribute. A bare "date" ("till what date is your info updated?") is left to the LLM.
        pointed = re.search(
            r"\b(?:these|those|them|they|their|each|every|all\s+of|handover|developers?|builders?|"
            r"completion|payment\s+plans?|service\s+charges?|bedrooms?|sizes?|prices?|brochures?)\b",
            text,
            re.I,
        )
        return _clean_resolution(
            {
                "action": "attribute",
                "place": place,
                "confidence": 0.92 if pointed else 0.85,
                "source": "offline",
            },
            text,
        )

    if _PROJECT_COUNT_SIGNAL.search(text) and not _SALE_COUNT_SIGNAL.search(text):
        return _clean_resolution(
            {
                "action": "project_count",
                "place": place or "Dubai",
                "source": "offline",
            },
            text,
        )

    if _SALE_COUNT_SIGNAL.search(text):
        return _clean_resolution(
            {
                "action": "sales_count",
                "place": place or sales_place_from_text(text),
                "beds": beds,
                "month": month,
                "year": year,
                "source": "offline",
            },
            text,
        )

    if _SALES_SIGNAL.search(text) and (
        re.search(r"\b(?:recent|latest|last|show|list|transactions?)\b", text, re.I)
        or place
        or sales_place_from_text(text)
    ):
        return _clean_resolution(
            {
                "action": "sales_list",
                "place": place or sales_place_from_text(text),
                "beds": beds,
                "month": month,
                "year": year,
                "source": "offline",
            },
            text,
        )

    # Incomplete "2 bed apartments" with no area/budget → let clarify path ask.
    if is_incomplete_home_browse(text):
        return _clean_resolution(
            {
                "action": "clarify",
                "beds": beds,
                "source": "offline",
            },
            text,
        )

    if _SEARCH_SIGNAL.search(text) and (place or beds is not None or budget is not None):
        return _clean_resolution(
            {
                "action": "search",
                "place": place,
                "beds": beds,
                "budget_aed": budget,
                "ready_only": ready_only,
                "source": "offline",
            },
            text,
        )

    return None


async def resolve_turn(
    question: str | None,
    conversation_memory=None,
    *,
    llm_generate: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Understand the user turn as action + slots.

    Returns None when understanding is too weak — caller falls through.
    """
    try:
        from app.retrieval.hybrid.question_text import repair_typed_question

        text = repair_typed_question((question or "").strip())
    except Exception:
        text = (question or "").strip()
    if not text:
        return None

    state = {}
    if conversation_memory is not None:
        if hasattr(conversation_memory, "get_dialogue_state"):
            state = conversation_memory.get_dialogue_state() or {}
        else:
            state = getattr(conversation_memory, "dialogue_state", None) or {}
    state = normalize_dialogue_state(state)

    # Named project compares ("Compare Serenz and Aura Heights") belong to the
    # hybrid comparison path — never invent a recommend clarify for area/beds.
    try:
        from app.retrieval.hybrid.question_text import comparison_pair

        if comparison_pair(text):
            return None
    except Exception:
        pass

    # Latency: skip LLM when offline already has a high-confidence executable
    # action. Never short-circuit a sticky sales_list over a price-extreme ask —
    # that was the "regex dump again" failure mode; prefer sales_stat / LLM.
    offline_fast = offline_resolve_turn(text, conversation_memory)
    offline_fast = enrich_resolution_from_state(offline_fast, state)
    if offline_fast:
        offline_fast["source"] = "offline"
        offline_fast["confidence"] = score_confidence(offline_fast, text)
        action = str(offline_fast.get("action") or "")
        kind = str(offline_fast.get("kind") or "").lower()
        # On-screen shortlist/compare is handled by specialized turns — never
        # treat it as an executable sticky afford/filtered recommend.
        if (
            action == "recommend"
            and kind == "shortlist"
            and _on_screen_project_count(conversation_memory) >= 2
        ):
            return None
        sticky_sales_trap = (
            action in {"sales_list", "sales_count"}
            and _stat_kind_from_text(text) is not None
        )
        if (
            not sticky_sales_trap
            and action
            in {
                "project_count",
                "sales_list",
                "sales_count",
                "sales_stat",
                "recommend",
                "search",
                "developer_rank",
                "small_talk",
                "clarify",
                "predict",
                "attribute",
                "agents",
            }
            # Keyword rules only win when near-certain (explicit stat words, greetings).
            # At 0.8 they answered "how much do homes sell for in X" with a project list
            # and "sales volume" with a sales list, so rewordings got different answers.
            # The meaning-based LLM step below handles everything else; offline stays the fallback.
            and (llm_generate is None or float(offline_fast.get("confidence") or 0) >= 0.9)
        ):
            return offline_fast

    # Compare / best-of-shown with live cards: do not let the LLM invent a
    # sticky afford/search re-run — specialized compare/shortlist turns handle it.
    if _on_screen_project_count(conversation_memory) >= 2:
        route = classify_intent(text, conversation_memory)
        if (
            route.get("intent") == "recommend"
            and route.get("kind") == "shortlist"
            and str(route.get("subject") or "").lower() == "on_screen"
        ):
            return None
        if re.search(r"\bcompar(?:e|ison|ing)\b", text, re.I):
            return None
        if re.search(
            r"\btell(?:\s+me)?(?:\s+the)?\s+best\b|"
            r"\bwhich (?:one |of these )?(?:is|are) (?:the )?best\b",
            text,
            re.I,
        ):
            return None

    llm_general = None
    if llm_generate is not None:
        prompt = f"""You are ACOT's turn understanding layer for Dubai real estate.
Do NOT answer the user. Resolve the message into JSON only.
Understand MEANING + prior dialogue state. Do not rely on exact keywords.

Today is {date.today().isoformat()}. Resolve relative dates against it ("last month" = the previous
calendar month, so set month and year).

Prior dialogue state (may be empty):
{json.dumps(state, default=str)}

User message:
{text}

Return ONLY JSON with keys:
- action: one of data_freshness, handovers, rent_stat, area_market, area_rank, top_projects, project_detail, guidance, search, project_count, sales_list, sales_count, sales_stat, developer_rank, attribute, recommend, agents, predict, general, clarify, small_talk
- goal: family|budget|afford|invest|ready|compare|open_buy or null
- place: Dubai community/area name or null (use canonical names like "Jumeirah Village Circle" for JVC when clear)
- beds: integer or null (studio=0)
- budget_aed: integer AED or null
- month: 1-12 or null
- year: e.g. 2026 or null
- ready_only: boolean
- prefer_cheaper: boolean
- kind: handovers: recent | upcoming. area_market: yield | roi | psf | trend | invest | profile. area_rank: cheapest_rent | priciest_rent |
  cheapest_buy | priciest_buy | best_yield. project_count: off_plan | ready when asked. Otherwise optional recommend kind (filtered|afford|family_areas|open_best|cheapest|shortlist|split|ready_offplan|yield_areas) or null
- stat: for sales_stat or rent_stat: cheapest | highest | average, else null
- developer: a developer the user named (e.g. "Danube", "Emaar"), else null
- project: a specific project/building name the user named (e.g. "Hanover Square"), else null
- months: lookback window in months when the user gave one ("last 12 months" = 12, "last year" = 12), else null
- confidence: number from 0 to 1 (how sure you are of action+slots)
- clarify_ask: one short question for the user when action is clarify, else null

Rules (meaning + state — never require exact wording):
- Same meaning, any wording / typos → same action.
- What homes cost / sell for / go for / the going rate / typical price in a place, with no ask to
  list or show options ("how much does a flat cost in X", "what do apartments in X go for") → sales_stat, stat average
- how recent / up to date / latest the data is, any wording or grammar ("which data do you have latest?"),
  or today's date / what day it is → data_freshness. Handover dates, developers or prices of projects are
  attribute (or search), never data_freshness.
- RENT (rent, rental price, lease, yearly rent) for a place → rent_stat with stat (average unless cheapest/highest) and beds.
  Never sales_stat for rent.
- Which AREA is cheapest / most expensive to rent or buy, or gives the best rental yield → area_rank with kind, beds if given
- For ONE named area → area_market with kind:
  yield = rental yield only (rent as a share of price);
  roi = ROI / return on investment / total return / how much I make (rent + price growth) — NOT the same as yield;
  psf = price per sq ft; trend = prices going up or down; invest = "is X a good investment";
  profile = "tell me about X", "what is X like", living in X, overview, or schools / metro / amenities /
  lifestyle / pros and cons of ONE named area
- Top / best-selling / most popular projects by sales → top_projects (year and place when given)
- Price, handover, developer, size or details of ONE named project ("price of Sobha Central") → project_detail, project=name
- How many projects a developer has → project_count with developer; "off-plan" / "ready" → kind off_plan / ready
- Handovers in a named year ("handing over in 2027", "completed in 2025") → handovers with year (kind upcoming for a
  future year, recent for a past one)
- Buying process, fees, visas, mortgages, who can buy, laws, general Dubai property knowledge → guidance
- Local life we hold no data for — any place or service in or near an area (hotels, malls, restaurants, cafes,
  gyms, hospitals, clinics, supermarkets, beaches, parks, mosques, things to do), "nearest / closest / biggest /
  best X", or metro / schools near projects or a place → guidance (place when named). It is answered from the
  web, never refused
- Predict / forecast / future price / "will prices go up" → predict (place from the message or the prior state)
- A place the user names that you do not recognise: set place to exactly what they typed, never swap in
  a place from the prior state
- What data or sources you use / where your data comes from → data_freshness
- Anything not about Dubai real estate or living in Dubai (weather, sport, news, coding) → general
- Which projects were handed over / completed / delivered recently or most recently → handovers, kind recent
- Which projects hand over next / soon / upcoming handovers → handovers, kind upcoming (place if one is named)
- how many projects / developments → project_count
- sales volume / number of transactions → sales_count
- how many sold / sale count → sales_count with place/beds/month/year when given
- list/show recent sales → sales_list (need a place; clarify if none)
- Asking for the cheapest / lowest / highest / average / "least I'd pay" / "which one is cheapest"
  among a project, area, OR the projects already shown after a sales answer
  → sales_stat (NOT another sales_list). Set stat. Leave project null when comparing "which one / these".
- If prior action is sales_list or sales_count and the user compares or asks a price extreme
  about those results → sales_stat, never re-list the same transactions.
- top / best / ranking of developers by sales → developer_rank
- developer / price / handover / bedrooms of these / those / each on-screen project → attribute
- show/list apartments/projects in X → search
- buy/best/family/afford/invest/recommend → recommend + goal
- best projects in a named area → recommend, kind filtered, place set — NOT open_best
- open_best only with no community and no budget (citywide where-to-buy)
- Compare all / compare these / tell me the best / which is best about options already
  shown → recommend, kind=shortlist, place=null (on-screen). Do NOT re-run a prior
  afford/filtered recommend with sticky place+budget.
- Named project compares ("Compare Serenz by Danube and Aura Heights", "A vs B")
  are NOT recommend/clarify. Leave them for the comparison retriever (action general
  is wrong too — prefer not inventing slots; if unsure use action general with
  confidence < 0.5 so the hybrid compare path can run).
- agents/brokers → agents
- greetings only → small_talk
- incomplete home browse (beds/type, no area, no budget) → clarify
- Conflicting products in one ask (studio AND villa, or studio AND 3/4-bed at the same time)
  → recommend, kind=split, place=null, beds=null. Do NOT invent a community from prior chat.
  Do NOT pick only one bedroom and run search.
- unclear otherwise → clarify with confidence < 0.7
- Never invent places. Prefer one clear question over guessing.
"""
        try:
            raw = await llm_generate(prompt)
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("action"):
                data["source"] = "llm"
                resolved = _clean_resolution(data, text)
                resolved = enrich_resolution_from_state(resolved, state)
                # Prefer the model's own confidence; only fill if missing.
                if resolved.get("confidence") is None:
                    resolved["confidence"] = score_confidence(resolved, text)
                # LLM understood the turn — do NOT fall back to regex offline.
                if resolved["action"] != "general":
                    return resolved
                # general → allow offline only as a last resort below
                llm_general = resolved
        except Exception:
            pass

    # Cold fallback when no LLM, or LLM failed / returned general / low offline confidence.
    if offline_fast:
        return offline_fast
    offline = offline_resolve_turn(text, conversation_memory)
    offline = enrich_resolution_from_state(offline, state)
    if offline:
        offline["confidence"] = score_confidence(offline, text)
        offline["source"] = "offline"
    # The LLM saw no property task and no rule matched: say so, so callers send it to the agent
    # instead of the default project list (metro questions were answered with 10 random projects).
    return offline or llm_general


def state_updates_from_resolution(resolution: dict[str, Any] | None) -> dict[str, Any]:
    """Dialogue-state fields to persist from a turn resolution."""
    if not resolution:
        return {}
    updates = {
        "action": resolution.get("action"),
        "goal": resolution.get("goal"),
        "kind": resolution.get("kind"),
        "place": resolution.get("place"),
        "beds": resolution.get("beds"),
        "budget_aed": resolution.get("budget_aed"),
        "month": resolution.get("month"),
        "year": resolution.get("year"),
        "ready_only": bool(resolution.get("ready_only")),
        "prefer_cheaper": bool(resolution.get("prefer_cheaper")),
        "exclude_names": [],
        "offset": 0,
        "source_question": resolution.get("source_question"),
        "focus": resolution.get("focus") if isinstance(resolution.get("focus"), dict) else None,
    }
    if resolution.get("limit") is not None:
        updates["limit"] = resolution.get("limit")
    return updates
