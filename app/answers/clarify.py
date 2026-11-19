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


def enrich_resolution_from_state(
    resolution: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Fill missing place/beds/budget from dialogue state so follow-ups stay sharp.

    Never paste a sticky place onto a conflicting / split ask — that invents a
    community the user did not name and turns a clarify/split into a wrong search.
    """
    if not resolution:
        return None
    out = dict(resolution)
    state = state or {}
    kind = str(out.get("kind") or "").lower()
    # Split / conflict asks are citywide product choices, not place follow-ups.
    if kind == "split" or out.get("conflict"):
        return out
    if not out.get("place") and state.get("place"):
        out["place"] = state["place"]
    if out.get("beds") is None and state.get("beds") is not None:
        out["beds"] = state["beds"]
    if out.get("budget_aed") is None and state.get("budget_aed") is not None:
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
        if place or budget is not None or beds is not None:
            return max(base, 0.8)
        return min(base, 0.4)

    if action in {"agents", "predict"}:
        return max(base, 0.85)

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
    if action in {"small_talk", "agents", "predict", "general"}:
        return False

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
