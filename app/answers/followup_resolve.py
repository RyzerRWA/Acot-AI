"""Phase 2/3: resolve short follow-ups from dialogue_state.

Entry is gated by an active dialogue_state.action — not by a growing list of
full-question phrase routes. Slot updates come from:
1) a small LLM JSON resolve when available (preferred; no hardcoded sentences)
2) generic deixis + slot parsers as offline fallback, always gated by state

Research note: LLM dialogue-state / function-call DST (e.g. FnCTOD, ParsingDST)
treats follow-ups as state updates + mode, not new regex routes per wording.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from app.answers.intent_route import (
    extract_place,
    family_area_focus,
    is_show_more,
    parse_bedrooms,
    parse_budget_aed,
    wants_more_results,
)
from app.memory.dialogue_state import normalize_dialogue_state, patch_dialogue_state


_REF_HINT = re.compile(
    r"\b(?:more|them|those|these|they|it|same|again|also|instead|cheaper|ready|only|"
    r"yes|ok|okay|sure|please|list|see|what about|how about|why|explain|good)\b",
    re.I,
)
# Deixis / meta-ask about the current result set (state-gated, not sentence routes).
_EXPLAIN_HINT = re.compile(
    r"\b(?:why|explain|reason|because|good(?:\s+for)?|fit|suit|makes?\s+sense|"
    r"strong(?:est)?|recommend(?:ation)?)\b|"
    r"\b(?:these|those|them|they|this|that)\b",
    re.I,
)
_NEW_TASK = re.compile(
    r"\b(?:show|find|list)\s+(?:me\s+)?(?:projects?|propert(?:y|ies)|apartments?|"
    r"villas?|townhouses?|agents?|brokers?)\b|"
    r"\bhow many\b|\bwho are\b|\bcompare\b|"
    r"\brecent\s+(?:sales?|transactions?)\b|"
    # Price extremes are a new ask; sales-thread context decides sales_stat.
    r"\b(?:cheapest|lowest|highest|most\s+expensive|average|median|"
    r"least\s+expensive|priciest)\b|"
    r"\bwhich\s+one\b",
    re.I,
)
# Catalog freshness / max sale date — never sticky-explain a prior recommend.
_DATA_FRESHNESS_ASK = re.compile(
    r"(?:"
    r"today'?s\s+date|"
    r"what(?:'s|\s+is)\s+today|"
    r"current\s+date|"
    r"latest\s+sale\s+date|"
    r"how\s+(?:fresh|current|up[\s-]?to[\s-]?date)\s+(?:is\s+)?(?:your\s+)?data|"
    r"how\s+recent\s+(?:is\s+)?(?:your\s+|the\s+)?(?:sales?\s+)?data\b|"
    r"what(?:'s|\s+is)\s+the\s+(?:most\s+)?recent\s+data\b|"
    r"(?:most\s+)?recent.{0,40}?data.{0,80}?\b(?:sales?|dubai)\b|"
    r"(?:most\s+)?recent.{0,40}?(?:dubai\s+)?sales?\s+data\b|"
    r"latest.{0,40}?data.{0,80}?\b(?:sales?|dubai)\b|"
    r"data\s+(?:we|you)\s+have.{0,60}?\bsales?\b|"
    r"sales?\s+data\b.{0,60}?\b(?:most\s+)?recent\b"
    r")",
    re.I | re.S,
)


def _is_attribute_about_shown(question: str | None) -> bool:
    """Developer/price/handover of 'these' is a new ask, not a sales_list continue."""
    try:
        from app.retrieval.hybrid.question_text import is_group_attribute_question

        return bool(is_group_attribute_question(question or ""))
    except Exception:
        return False


def looks_like_state_followup(question: str | None, state: dict[str, Any] | None) -> bool:
    """Gate: ask the follow-up LLM whenever a task is active.

    Understanding (continue vs new) is the LLM's job — not this function.
    """
    current = normalize_dialogue_state(state)
    if not current.get("action"):
        return False
    return bool((question or "").strip())


_SPLIT_STUDIO = re.compile(r"^\s*(?:the\s+)?studios?\s*[.!]?\s*$", re.I)
_SPLIT_VILLA = re.compile(r"^\s*(?:the\s+)?villas?\s*[.!]?\s*$", re.I)


def _active_split_task(state: dict[str, Any]) -> bool:
    if str(state.get("kind") or "").lower() == "split":
        return True
    try:
        from app.answers.intent_route import is_split_constraints

        return bool(is_split_constraints(state.get("source_question")))
    except Exception:
        return False


def _family_focus_is_new_ask(question: str, state: dict[str, Any]) -> bool:
    """Kids / not-towers / villa refinements are a new family ranking, not a re-run."""
    if state.get("action") != "recommend":
        return False
    kind = str(state.get("kind") or "").lower()
    prior = state.get("source_question") or ""
    if kind != "family_areas" and "famil" not in prior.lower():
        return False
    asked = family_area_focus(question)
    if not any(asked.get(key) for key in ("kids", "avoid_towers", "want_villas", "want_3bed")):
        return False
    # Must look like a family-areas question, not a filtered buy in one community.
    if not re.search(
        r"\b(?:famil|kids?|children|communit(?:y|ies)|areas?|neighbourhoods?|neighborhoods?)\b",
        question or "",
        re.I,
    ):
        return False
    prior_focus = family_area_focus(prior)
    carried = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    for key in ("kids", "avoid_towers", "want_villas", "want_3bed", "not_studios"):
        if asked.get(key) and not (prior_focus.get(key) or carried.get(key)):
            return True
    return False


def _split_path_followup(question: str, state: dict[str, Any]) -> dict[str, Any] | None:
    """After a studio+villa split, 'studios' / 'villas' picks one path to deepen."""
    if state.get("action") != "recommend" or not _active_split_task(state):
        return None
    text = (question or "").strip()
    if _SPLIT_STUDIO.match(text) or re.fullmatch(r"(?:go\s+)?(?:deeper\s+on\s+)?studios?", text, re.I):
        budget = state.get("budget_aed")
        source = (
            f"studios under AED {int(budget):,}"
            if budget is not None
            else "studios"
        )
        return {
            "kind": "continue",
            "more": False,
            "mode": "patch",
            "action": "recommend",
            "patch": {
                "beds": 0,
                "kind": "filtered",
                "source_question": source,
                "goal": "budget" if budget is not None else state.get("goal"),
            },
        }
    if _SPLIT_VILLA.match(text) or re.fullmatch(r"(?:go\s+)?(?:deeper\s+on\s+)?villas?", text, re.I):
        from app.answers.intent_route import parse_bedrooms

        prior = state.get("source_question") or ""
        beds = parse_bedrooms(prior) or []
        villa_beds = next((b for b in beds if isinstance(b, int) and b >= 3), 4)
        budget = state.get("budget_aed")
        source = (
            f"{villa_beds}-bedroom villas under AED {int(budget):,}"
            if budget is not None
            else f"{villa_beds}-bedroom villas"
        )
        return {
            "kind": "continue",
            "more": False,
            "mode": "patch",
            "action": "recommend",
            "patch": {
                "beds": villa_beds,
                "kind": "filtered",
                "source_question": source,
                "goal": "budget" if budget is not None else state.get("goal"),
            },
        }
    return None


def _offline_resolve(question: str, state: dict[str, Any]) -> dict[str, Any]:
    """Slot patch without LLM — parsers + deixis only, gated by active state."""
    text = question or ""
    split_path = _split_path_followup(text, state)
    if split_path is not None:
        return split_path
    # "Most recent Dubai sales data?" is a catalog-freshness ask, not explain/rerun.
    if _DATA_FRESHNESS_ASK.search(text):
        return {
            "kind": "new",
            "more": False,
            "mode": "rerun",
            "action": state.get("action"),
            "patch": {},
        }
    if _family_focus_is_new_ask(text, state):
        return {
            "kind": "new",
            "more": False,
            "mode": "rerun",
            "action": state.get("action"),
            "patch": {},
        }
    if _is_attribute_about_shown(text):
        return {
            "kind": "new",
            "more": False,
            "mode": "rerun",
            "action": state.get("action"),
            "patch": {},
        }
    # After a sales answer, a price extreme / "which one" compare is a new
    # sales_stat — never continue/re-dump the sticky sales_list.
    if str(state.get("action") or "") in {"sales_list", "sales_count"}:
        try:
            from app.answers.turn_resolve import _stat_kind_from_text
            from app.retrieval.hybrid.question_text import repair_typed_question

            repaired = repair_typed_question(text)
        except Exception:
            repaired = text
        if _stat_kind_from_text(repaired) or re.search(
            r"\b(?:which\s+one|which\s+(?:of\s+)?(?:these|those|them)|among\s+(?:these|those))\b",
            repaired,
            re.I,
        ):
            return {
                "kind": "new",
                "more": False,
                "mode": "rerun",
                "action": state.get("action"),
                "patch": {},
            }
    more = bool(
        is_show_more(text)
        or wants_more_results(text)
        or re.search(
            r"\b(?:i\s+)?(?:want|need|give\s+me|show\s+me)?\s*more\b|"
            r"\bmore\s+(?:please|pls|options?|results?|sales?|projects?)?\b",
            text,
            re.I,
        )
    )
    # A full new search/sales ask is not a follow-up, even with active state.
    if (
        _NEW_TASK.search(text)
        and not more
        and not re.search(r"\b(?:them|those|these|same)\b", text, re.I)
    ):
        return {
            "kind": "new",
            "more": False,
            "mode": "rerun",
            "action": state.get("action"),
            "patch": {},
        }
    patch: dict[str, Any] = {}
    place = extract_place(text)
    if place:
        patch["place"] = place
    beds = parse_bedrooms(text)
    if beds:
        patch["beds"] = beds[0]
    budget = parse_budget_aed(text)
    if budget is not None:
        patch["budget_aed"] = budget
    if re.search(r"\b(?:ready(?:\s+only)?|only\s+ready)\b", text, re.I):
        patch["ready_only"] = True
    if re.search(r"\b(?:cheaper|cheapest|more affordable|lower budget)\b", text, re.I):
        patch["prefer_cheaper"] = True
    # Asking why/how the current picks fit → explain, do not re-search.
    explain = (
        state.get("action") in {"recommend", "search"}
        and bool(_EXPLAIN_HINT.search(text))
        and not place
        and not beds
        and budget is None
        and not more
    )
    # Bare acknowledgements with an active task → continue same action.
    continue_same = (
        more
        or explain
        or bool(_REF_HINT.search(text))
        or len(text.split()) <= 4
    )
    action = state.get("action")
    # Selective overwrite (SOM-DST / Diable): "same … 2 bedrooms" keeps
    # sales_list + place/limit and only updates beds.
    sales_slot_tweak = (
        action in {"sales_list", "sales_count"}
        and bool(beds or place)
        and (
            bool(_REF_HINT.search(text))
            or len(re.findall(r"[a-z0-9]+", text.lower())) <= 8
        )
        and not _NEW_TASK.search(text)
    )
    if sales_slot_tweak:
        continue_same = True
        action = "sales_list"
        # New bedroom/place filter → restart at the first page.
        if beds or place:
            patch["offset"] = 0
    show_list = more or bool(
        re.search(r"\b(?:show|list|see|them|those|details?)\b", text, re.I)
    )
    # After a count, "ok show me" / "list them" means list the counted set.
    if action == "sales_count" and show_list:
        action = "sales_list"
    if action == "project_count" and show_list:
        action = "search"
        continue_same = True
    mode = "explain" if explain else ("page" if more else ("patch" if patch else "rerun"))
    return {
        "kind": "continue" if continue_same or patch else "new",
        "more": more,
        "mode": mode,
        "action": action,
        "patch": patch,
    }


async def resolve_followup(
    question: str | None,
    state: dict[str, Any] | None,
    *,
    llm_generate: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """
    Return {kind, more, action, patch} to continue the active task, or None.

    kind: continue | new
    """
    current = normalize_dialogue_state(state)
    if not looks_like_state_followup(question, current):
        return None

    text = (question or "").strip()
    # Deterministic: after a split, "studios"/"villas" picks a path (don't re-ask LLM).
    split_path = _split_path_followup(text, current)
    if split_path is not None:
        return split_path
    # Catalog freshness must leave the sticky recommend/search thread.
    if _DATA_FRESHNESS_ASK.search(text):
        return None
    # Family kids / not-towers refinements must re-rank — never replay the prior list.
    if _family_focus_is_new_ask(text, current):
        return None

    # Latency: offline-first for clear continues / new asks — skip a full LLM hop.
    offline = _offline_resolve(text, current)
    if offline.get("kind") == "new":
        return None
    if offline.get("kind") == "continue" and (
        offline.get("more")
        or offline.get("patch")
        or offline.get("mode") in {"page", "patch", "explain"}
        or (
            offline.get("mode") == "rerun"
            and len(text.split()) <= 6
        )
    ):
        return offline

    if llm_generate is not None:
        prompt = f"""You resolve follow-ups for ACOT (Dubai property assistant).
Use dialogue state, not fixed sentence patterns.

Active task state (JSON):
{json.dumps(current, default=str)}

Latest user message:
{text}

Return ONLY JSON with keys:
- kind: "continue" if this continues the active task, else "new"
- more: true if they want more/next page of the same results
- mode: one of "explain", "page", "patch", "rerun"
  - explain = ask why/how the current on-screen picks fit (do NOT re-search)
  - page = show more of the same result set
  - patch = same task with updated slots (place/beds/budget/ready/cheaper)
  - rerun = continue same action with current slots
- action: one of search, project_count, sales_list, sales_count, recommend, agents, clarify (or same as state)
- patch: object with any of place, beds, budget_aed, ready_only, prefer_cheaper, month, year
  Use null for fields not mentioned. beds is an integer (0=studio).

Rules:
- References to these/those/them/the list + why/good/explain/family fit → continue, mode=explain
- After project_count or sales_count, "show me" / "list them" → continue; action becomes search or sales_list
- "ok show me", "yes", "show me more", "list them" → continue (page or rerun)
- "what about Marina?", "only ready", "cheaper" → continue with patch
- After sales_list: "same give in 2 bedrooms" / "same for 2 beds" / "now 3 bedroom" → continue,
  mode=patch, action=sales_list, patch only beds (carry place + prior last-N limit)
- After a split recommend (studio AND villa): reply "studios" / "villas" → continue, mode=patch,
  action=recommend, patch beds=0 for studios or beds=3/4 for villas, and set kind to filtered
  (do not re-run the conflict split)
- After family_areas: a NEW ranking ask (kids, not towers, villas/3-bed stock) → kind=new
  Example: "Best communities for a family with kids, not just the busiest towers."
  Do NOT continue/rerun the previous general family list.
- After sales_list/sales_count: any compare or price extreme about those results
  ("which one is cheapest", "lowest among these", "least I'd pay…") → kind=new
  Never continue/re-list the same transactions. Understanding maps that to sales_stat.
- Attribute asks about the shown set are NOT sales continues: kind=new
  Examples: "Who is the developer of each of these projects?", "their handover dates",
  "budget for each project", "what are their prices?"
- A full new ask like "Show me villas in Palm" with no link to the active task → new
- Do not invent places. Only patch what the user said. Carry over unmentioned slots.
"""
        try:
            # Ambiguous follow-ups only — clear page/patch/new already returned above.
            raw = await llm_generate(prompt)
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("kind") in {"continue", "new"}:
                patch = data.get("patch") if isinstance(data.get("patch"), dict) else {}
                # Drop nulls so patch_dialogue_state keeps prior slots.
                clean_patch = {
                    key: value
                    for key, value in patch.items()
                    if value is not None and key in {
                        "place", "beds", "budget_aed", "ready_only",
                        "prefer_cheaper", "month", "year", "action", "limit",
                        "offset", "kind", "source_question",
                    }
                }
                action = data.get("action") or current.get("action")
                if (
                    data.get("more")
                    or re.search(r"\b(?:show|list|see|them|those)\b", text, re.I)
                ):
                    if action == "sales_count":
                        action = "sales_list"
                    elif action == "project_count":
                        action = "search"
                mode = str(data.get("mode") or "").strip().lower()
                if mode not in {"explain", "page", "patch", "rerun"}:
                    if data.get("more"):
                        mode = "page"
                    elif clean_patch:
                        mode = "patch"
                    else:
                        mode = "rerun"
                # Fresh question → caller runs the main LLM turn resolver next.
                if data["kind"] == "new":
                    return None
                return {
                    "kind": data["kind"],
                    "more": bool(data.get("more")),
                    "mode": mode,
                    "action": action,
                    "patch": clean_patch,
                }
        except Exception:
            pass

    # Offline only when no LLM or the LLM call failed.
    return _offline_resolve(text, current)


def apply_followup(
    state: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge resolution into state. None if not a continue."""
    if not resolution or resolution.get("kind") != "continue":
        return None
    updates = dict(resolution.get("patch") or {})
    if resolution.get("action"):
        updates["action"] = resolution["action"]
    if resolution.get("more"):
        # Caller advances offset; mark action for sales paging.
        if updates.get("action") == "sales_count" or (
            normalize_dialogue_state(state).get("action") == "sales_count"
            and not updates.get("action")
        ):
            updates["action"] = "sales_list"
    return patch_dialogue_state(state, updates)
