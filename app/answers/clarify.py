"""Phase 7: ask before guessing when understanding or slots are weak.

Industry pattern (RAG clarify / confidence thresholds): answer only when
evidence+slots are enough; otherwise one short clarifying question.
"""
from __future__ import annotations

from typing import Any

from app.answers.intent_route import is_incomplete_home_browse

# Align with common bot thresholds (~0.70): below this → clarify, don't invent.
CLARIFY_CONFIDENCE = 0.70

_OPEN_GOALS = frozenset({"open_buy", "family", "invest"})
_OPEN_KINDS = frozenset({"open_best", "family_areas", "yield_areas", "open_buy"})


def _as_confidence(value: Any, default: float = 0.85) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _citywide_dubai_ask(text: str | None) -> bool:
    """True when the user named Dubai as the market, not a prior community."""
    import re

    from app.answers.intent_route import extract_place

    asked = text or ""
    if not re.search(r"\bdubai\b", asked, re.I):
        return False
    named = extract_place(asked)
    return not named or str(named).strip().lower() == "dubai"


def enrich_resolution_from_state(
    resolution: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Fill missing place/beds/budget from dialogue state so follow-ups stay sharp.

    Never paste a sticky place onto a conflicting / split ask — that invents a
    community the user did not name and turns a clarify/split into a wrong search.
    Never paste a sticky community (e.g. JVC) onto an explicit Dubai-wide browse.
    """
    if not resolution:
        return None
    out = dict(resolution)
    state = state or {}
    kind = str(out.get("kind") or "").lower()
    # Split / conflict asks are citywide product choices, not place follow-ups.
    if kind == "split" or out.get("conflict"):
        return out
    text = str(out.get("source_question") or "")
    action = str(out.get("action") or "")
    sales_action = action in {"sales_list", "sales_count", "sales_stat"}
    citywide = _citywide_dubai_ask(text)

    if sales_action:
        # Sales SQL ignores budget; sticky AED from a prior browse only
        # pollutes clarify ("Understood … under AED 5M").
        sales_place = _sales_place_hint(text)
        if sales_place:
            out["place"] = sales_place
        elif has_explicit_sales_token(text):
            # "sales in AC3" must not re-list sticky JVC when AC3 is unresolved.
            out["place"] = None
        elif citywide:
            out["place"] = None
        elif not out.get("place") and state.get("place"):
            # Continue an active sales thread only. Do not paste browse/recommend
            # place onto a bare "transaction details" ask — on-screen cards are
            # handled separately when that context is still live.
            if str(state.get("action") or "") in {"sales_list", "sales_count"}:
                out["place"] = state["place"]
    elif action == "agents":
        # Same gate as sales: bare "top agents" must not inherit browse/recommend
        # place. Live on-screen cards are attached later; agents-thread place sticks.
        if citywide:
            out["place"] = "Dubai"
        elif not out.get("place") and state.get("place"):
            if str(state.get("action") or "") == "agents":
                out["place"] = state["place"]
    elif citywide:
        # "projects in Dubai …" is citywide — drop sticky JVC/Marina/etc.
        out["place"] = None
    elif not out.get("place") and state.get("place"):
        out["place"] = state["place"]

    if out.get("beds") is None and state.get("beds") is not None:
        # Keep beds for short follow-ups / same-place paging.
        # Drop them when the user switches sales subject (JVC → AC3) or starts a
        # Dubai-wide browse without naming bedrooms — sticky 1-bed emptied AC3.
        sticky_place = str(state.get("place") or "").strip().lower()
        current_place = str(out.get("place") or "").strip().lower()
        sales_place_changed = bool(
            sales_action
            and current_place
            and sticky_place
            and current_place != sticky_place
            and not _beds_mentioned(text)
        )
        # A fresh recommend in a new place, or after another kind of task
        # (sales/search/clarify), must not inherit a stale studio/1-bed slot.
        fresh_recommend_place_change = bool(
            action == "recommend"
            and not _beds_mentioned(text)
            and (
                str(state.get("action") or "") != "recommend"
                or (
                    current_place
                    and sticky_place
                    and current_place != sticky_place
                )
            )
        )
        if sales_place_changed:
            pass
        elif fresh_recommend_place_change:
            pass
        elif citywide and not _beds_mentioned(text):
            pass
        else:
            out["beds"] = state["beds"]
    if out.get("budget_aed") is None and state.get("budget_aed") is not None:
        # Never paste sticky AED onto sales/agents. Also never paste a prior
        # browse budget onto a fresh recommend in a new place (e.g. "best
        # project in Dubai Marina for family" after a 50k studio search).
        sticky_place = str(state.get("place") or "").strip().lower()
        current_place = str(out.get("place") or "").strip().lower()
        place_changed = bool(
            current_place
            and sticky_place
            and current_place != sticky_place
        )
        prior_action = str(state.get("action") or "")
        if sales_action or action == "agents":
            pass
        elif action == "recommend" and prior_action != "recommend":
            pass
        elif action == "recommend" and place_changed:
            pass
        elif action == "search" and place_changed and not _budget_mentioned(text):
            pass
        else:
            out["budget_aed"] = state["budget_aed"]
    if out.get("goal") is None and state.get("goal"):
        out["goal"] = state["goal"]
    # Incomplete browse after a locked place → search, don't re-ask.
    if out.get("action") == "clarify" and out.get("place"):
        place = str(out["place"]).strip().lower()
        if place and place != "dubai":
            out["action"] = "search"
            out["confidence"] = max(_as_confidence(out.get("confidence"), 0.5), 0.85)
    return out


def _beds_mentioned(text: str | None) -> bool:
    from app.answers.intent_route import parse_bedrooms

    return bool(parse_bedrooms(text or ""))


def _budget_mentioned(text: str | None) -> bool:
    from app.answers.intent_route import parse_budget_aed

    return parse_budget_aed(text or "") is not None


def _sales_place_hint(text: str | None) -> str | None:
    try:
        from app.answers.turn_resolve import sales_place_from_text

        return sales_place_from_text(text)
    except Exception:
        return None


def has_explicit_sales_token(text: str | None) -> bool:
    try:
        from app.answers.turn_resolve import has_explicit_sales_place_token

        return bool(has_explicit_sales_place_token(text))
    except Exception:
        return False


def score_confidence(
    resolution: dict[str, Any] | None,
    question: str | None = None,
) -> float:
    """Confidence that we can execute without disappointing the user."""
    if not resolution:
        return 0.0
    if resolution.get("confidence") is not None:
        base = _as_confidence(resolution.get("confidence"), 0.5)
    else:
        base = 0.85 if resolution.get("source") == "llm" else 0.8

    action = str(resolution.get("action") or "")
    if action == "clarify":
        return min(base, 0.45)
    if action in {"small_talk"}:
        return max(base, 0.9)
    if action == "general":
        return min(base, 0.55)

    place = resolution.get("place")
    beds = resolution.get("beds")
    budget = resolution.get("budget_aed")
    goal = str(resolution.get("goal") or "")
    kind = str(resolution.get("kind") or "")

    if action == "search":
        # Place already locked (message or dialogue state) → executable.
        if place and str(place).strip().lower() != "dubai":
            return max(base, 0.8)
        if not place and is_incomplete_home_browse(
            question or resolution.get("source_question")
        ):
            return min(base, 0.4)
        if not place and beds is None and budget is None:
            return min(base, 0.35)
        if place or beds is not None or budget is not None:
            return max(base, 0.8)

    if action in {"sales_list", "sales_count"}:
        if not place:
            return min(base, 0.4)
        return max(base, 0.85)

    if action == "project_count":
        return max(base, 0.85)

    if action == "recommend":
        if goal in _OPEN_GOALS or kind in _OPEN_KINDS:
            return max(base, 0.8)
        try:
            from app.retrieval.hybrid.question_text import comparison_pair

            if comparison_pair(question or resolution.get("source_question")):
                return max(base, 0.85)
        except Exception:
            pass
        if place or budget is not None or beds is not None:
            return max(base, 0.8)
        return min(base, 0.4)

    if action == "predict":
        return max(base, 0.85)
    if action == "agents":
        if place and str(place).strip():
            return max(base, 0.85)
        return min(base, 0.4)

    return base


def should_clarify(
    resolution: dict[str, Any] | None,
    question: str | None = None,
) -> bool:
    """True when we must ask one question instead of executing a weak guess."""
    if not resolution:
        return False
    action = str(resolution.get("action") or "")
    if action == "clarify":
        return True
    if action in {"small_talk", "predict", "general"}:
        return False

    asked = question or resolution.get("source_question")
    # Named project compares never clarify for area/beds/budget — hybrid compare runs.
    try:
        from app.retrieval.hybrid.question_text import comparison_pair

        if comparison_pair(asked):
            return False
    except Exception:
        pass

    # Explicit model/offline confidence below threshold always asks first.
    raw_confidence = resolution.get("confidence")
    if raw_confidence is not None and _as_confidence(raw_confidence) < CLARIFY_CONFIDENCE:
        return True

    confidence = score_confidence(resolution, question)
    if confidence < CLARIFY_CONFIDENCE:
        return True

    place = resolution.get("place")
    beds = resolution.get("beds")
    budget = resolution.get("budget_aed")
    goal = str(resolution.get("goal") or "")
    kind = str(resolution.get("kind") or "")

    if action == "search":
        has_area = bool(place and str(place).strip().lower() != "dubai")
        if has_area:
            return False
        if is_incomplete_home_browse(question or resolution.get("source_question")):
            return True
        if not place and beds is None and budget is None:
            return True
    if action in {"sales_list", "sales_count"} and not place:
        return True
    if action == "agents" and not place:
        return True
    if action == "recommend":
        if goal in _OPEN_GOALS or kind in _OPEN_KINDS:
            return False
        if not place and budget is None and beds is None:
            return True
    return False


def clarify_question_text(
    resolution: dict[str, Any] | None,
    question: str | None = None,
) -> str:
    """Understood restatement + one short ask — never invent projects or prices."""
    from app.answers.explain_contract import build_understood, prepend_understood

    beds = (resolution or {}).get("beds")
    place = (resolution or {}).get("place")
    budget = (resolution or {}).get("budget_aed")
    action = str((resolution or {}).get("action") or "")
    text = question or (resolution or {}).get("source_question") or ""

    intent = "clarify"
    if action in {"sales_list", "sales_count"}:
        intent = "sales"
        # Only echo a budget the user named on this sales ask — never sticky AED.
        try:
            from app.answers.intent_route import parse_budget_aed

            budget = parse_budget_aed(text)
        except Exception:
            budget = None
    elif action == "recommend":
        intent = "recommend"
    understood = build_understood(
        place=place if isinstance(place, str) else None,
        beds=beds if isinstance(beds, int) else None,
        budget_aed=int(budget) if isinstance(budget, (int, float)) else None,
        intent=intent,
        goal=str((resolution or {}).get("goal") or "") or None,
    )

    if isinstance(resolution, dict):
        custom = str(resolution.get("clarify_ask") or "").strip()
        if custom:
            return prepend_understood(custom, understood)

    if action in {"sales_list", "sales_count"} or (
        not place and ("sale" in text.lower() or "sold" in text.lower() or "transaction" in text.lower())
    ):
        ask = (
            "Which Dubai area or project should I check sales for? "
            "For example JVC, Business Bay, or Dubai Marina."
        )
        return prepend_understood(ask, understood)

    if action == "agents" or (
        not place and ("agent" in text.lower() or "broker" in text.lower())
    ):
        ask = (
            "Which Dubai community should I find licensed agents for? "
            "For example JVC, Business Bay, Dubai Marina, or Palm Jumeirah."
        )
        return prepend_understood(ask, understood)

    if action == "recommend":
        ask = (
            "To recommend well I need an area, bedrooms, or a budget. "
            "Which can you share first?"
        )
        return prepend_understood(ask, understood)

    if beds is not None or is_incomplete_home_browse(text):
        bed_bit = ""
        if beds == 0:
            bed_bit = "studio "
        elif isinstance(beds, int):
            bed_bit = f"{beds}-bedroom "
        ask = (
            f"Which Dubai community should I search for {bed_bit}homes? "
            "You can also share a budget if you have one."
        )
        return prepend_understood(ask, understood)

    ask = (
        "I want to get this right — are you browsing projects, checking recent sales, "
        "or looking for a recommendation? Share an area if you have one."
    )
    return prepend_understood(ask, understood)


def clarify_turn_payload(
    question: str | None,
    resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fast structured reply shape used by run._understood_turn."""
    answer = clarify_question_text(resolution, question)
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "information",
        "structured_summary": {
            "projects": [],
            "community": [],
            "sub_communities": [],
            "understood_kind": "clarify",
            "clarify": True,
        },
        # Top-level documents required by _finalize_acot_turn / build_frontend_response.
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "information",
            "structured_data": {
                "projects": [],
                "community": [],
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
        "understood_action": "clarify",
        "understood_confidence": score_confidence(resolution, question),
    }
