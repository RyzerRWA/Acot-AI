"""Step 1: classify a turn as predict, recommend, or neither.

Predict is checked before recommend so “best investment next year” stays a trend
question. Search, stats, agents, and small talk stay neither.
"""
from __future__ import annotations

import re
from typing import Any

from app.listings.tools import ALIASES
from app.retrieval.propai_catalog import known_dubai_label

_PREDICT = re.compile(
    r"\b(?:"
    r"next year|next month|in the future|"
    r"will (?:the )?prices?(?:\s+\w+){0,6}\s+be|"
    r"will (?:the )?price|"
    r"will (?:it|they) (?:go |rise|fall|appreciate)|"
    r"forecast|predict(?:ion)?s?|outlook|"
    r"expected (?:price|rent)|"
    r"go up|go down|appreciat(?:e|ion)"
    r")\b",
    re.I,
)

_AFFORD = re.compile(
    r"\b(?:"
    r"enough for|"
    r"is (?:aed|dhs)?\s*[\d,.]+\s*(?:k|m|million|thousand)?\s+enough|"
    r"can i (?:buy|afford|get)|"
    r"afford"
    r")\b",
    re.I,
)

_OPEN_BEST = re.compile(
    r"\b(?:"
    r"best (?:property|project)(?: in dubai)?|"
    r"which (?:one |of these )?(?:is |are )?(?:the )?best property|"
    r"tell me which one is the best"
    r")\b",
    re.I,
)

_SHORTLIST = re.compile(
    r"\b(?:"
    r"which (?:one |of these |of those )?(?:is|are) (?:the )?best|"
    r"which of these|"
    r"recommend (?:one|one of these|the best|among these)|"
    r"which (?:of these )?(?:should|would) i|"
    r"what should i (?:pick|choose|buy)"
    r")\b",
    re.I,
)

_SUGGEST_PICK = re.compile(
    r"\b(?:recommend|suggest)\b.{0,40}\b(?:one|best|buy|invest|property|project)\b|"
    r"\b(?:should i buy|worth buying|good buy|where should i (?:invest|buy))\b",
    re.I,
)

_WANT_FILTER = re.compile(
    r"\b(?:i want|i need|looking for|i'?m looking)\b",
    re.I,
)

_BUY = re.compile(r"\b(?:buy|buying|to buy|purchase|off-?plan|for sale)\b", re.I)
_RENT = re.compile(r"\b(?:rent|rental|to rent|yearly rent)\b", re.I)
_STUDIO = re.compile(r"\bstudios?\b", re.I)
_VILLA = re.compile(r"\bvillas?\b", re.I)
_FOUR_BED = re.compile(r"\b4\s*-?\s*(?:bed(?:room)?s?|br)\b", re.I)
_BED = re.compile(r"\b(\d+)\s*-?\s*(?:bed(?:room)?s?|br)\b", re.I)
_BUDGET = re.compile(
    r"(?:aed|dhs)\s*([\d,.]+)\s*(k|m|million|thousand)?|"
    r"\b([\d,.]+)\s*(k|m|million)\b",
    re.I,
)
_MONTHS = re.compile(r"\b(?:in|within)\s+(\d+)\s+months?\b", re.I)
_STATS = re.compile(
    r"\b(?:average|median|per sq|sq ?ft|last 12 months|how many|"
    r"show (?:me )?(?:communities|properties|projects|transactions)|"
    r"list (?:all )?agents)\b",
    re.I,
)
_SNAPSHOT = re.compile(
    r"\b(?:median sale|median price|average price|avg price|"
    r"price per sq|per sq ?ft|sq ?ft|"
    r"latest data date|data date|last 12 months)\b",
    re.I,
)
_FAMILY = re.compile(r"\bfamil(?:y|ies)\b", re.I)
_FAMILY_AREAS = re.compile(
    r"\b(?:"
    r"(?:which|what|best|top|good|ideal|suit(?:able)?)\b.{0,48}"
    r"\b(?:areas?|communities|neighbourhoods?|neighborhoods?|places?)\b.{0,32}\bfamil"
    r"|\bfamil(?:y|ies)\b.{0,32}\b(?:areas?|communities|neighbourhoods?|neighborhoods?|places?)"
    r"|good place for (?:a )?famil"
    r"|\bwhere\b.{0,48}\bfamil"
    r")",
    re.I,
)
_FAMILY_KIDS = re.compile(r"\b(?:kids?|children)\b", re.I)
_FAMILY_AVOID_TOWERS = re.compile(
    r"\b(?:not (?:just )?(?:the )?(?:busiest )?towers?|busiest towers?|"
    r"not apartments?|apartment towers?)\b",
    re.I,
)
_FAMILY_VILLAS = re.compile(r"\b(?:villas?|townhouses?|town houses?|low-?rise)\b", re.I)
_FAMILY_NOT_STUDIOS = re.compile(r"\bnot studios?\b", re.I)
_FAMILY_3BED = re.compile(r"\b3\s*-?\s*bed", re.I)
_YIELD = re.compile(
    r"\b(?:rental\s+yields?|gross\s+yields?|yields?|rois?|return on investment)\b",
    re.I,
)
_READY_HOME = re.compile(
    r"\b(?:ready(?:[\s-]?to[\s-]?(?:move|buy|live))?|completed|existing homes?)\b",
    re.I,
)
_OFFPLAN = re.compile(r"\boff[\s-]?plan\b", re.I)
_SHOW_MORE = re.compile(
    r"^\s*(?:(?:please|kindly)\s+)?"
    r"(?:(?:can|could|would)\s+you\s+)?"
    r"(?:show|give|list|find|send)\s+(?:me\s+)?more"
    r"(?:\s+(?:of\s+)?(?:these|those|them|options?|projects?|properties|"
    r"results|listings|communities|areas?))?"
    r"(?:\s+please)?"
    r"\s*[.!]?\s*$",
    re.I,
)
_SHOW_MORE_SHORT = re.compile(
    r"^\s*(?:any\s+)?more"
    r"(?:\s+(?:please|options?|projects?|properties|results|"
    r"like\s+(?:these|this|those)))?"
    r"\s*[.!]?\s*$",
    re.I,
)


def family_area_focus(question: str | None) -> dict[str, bool]:
    """What the family-area question actually asked for."""
    text = question or ""
    avoid_towers = bool(_FAMILY_AVOID_TOWERS.search(text))
    want_villas = bool(_FAMILY_VILLAS.search(text))
    kids = bool(_FAMILY_KIDS.search(text))
    return {
        "kids": kids,
        "avoid_towers": avoid_towers,
        "want_villas": want_villas or avoid_towers,
        "not_studios": bool(_FAMILY_NOT_STUDIOS.search(text)),
        "want_3bed": bool(_FAMILY_3BED.search(text)),
    }


def parse_budget_aed(question: str | None) -> int | None:
    text = question or ""
    match = _BUDGET.search(text)
    if not match:
        return None
    raw = (match.group(1) or match.group(3) or "").replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return None
    suffix = (match.group(2) or match.group(4) or "").lower()
    if suffix in {"k", "thousand"}:
        number *= 1_000
    elif suffix in {"m", "million"}:
        number *= 1_000_000
    if number < 10_000:
        return None
    return int(round(number))


def parse_months(question: str | None) -> int | None:
    match = _MONTHS.search(question or "")
    return int(match.group(1)) if match else None


def extract_place(question: str | None) -> str | None:
    text = question or ""
    lowered = text.lower()
    for key in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            return ALIASES[key]
    from app.retrieval.propai_catalog import _static_dubai_names

    hits = []
    for key, proper in _static_dubai_names().items():
        if len(key) < 4:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            hits.append((len(key), proper))
    if hits:
        hits.sort(reverse=True)
        return hits[0][1]
    return known_dubai_label(text)


def is_split_constraints(question: str | None) -> bool:
    text = question or ""
    studio = bool(_STUDIO.search(text))
    other = bool(_VILLA.search(text) or _FOUR_BED.search(text))
    return studio and other


def _on_screen(conversation_memory) -> bool:
    items = getattr(conversation_memory, "active_candidates", None) or []
    projects = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    return len(projects) >= 2


def is_show_more(question: str | None) -> bool:
    """True when the turn only asks for more of the previous search."""
    text = (question or "").strip()
    if not text:
        return False
    return bool(_SHOW_MORE.match(text) or _SHOW_MORE_SHORT.match(text))


def prior_anchor_question(conversation_memory, current: str | None = None) -> str | None:
    """Last real search question, skipping 'show me more' itself."""
    seen = {(current or "").strip().lower()}
    candidates: list[str] = []
    for value in (
        getattr(conversation_memory, "last_standalone_question", None),
        getattr(conversation_memory, "last_user_question", None),
    ):
        if value:
            candidates.append(str(value))
    for turn in reversed(getattr(conversation_memory, "turns", None) or []):
        if not isinstance(turn, dict):
            continue
        for key in ("standalone_question", "user_question"):
            value = turn.get(key)
            if value:
                candidates.append(str(value))
    for raw in candidates:
        text = raw.strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        if is_show_more(text):
            continue
        return text
    return None


def classify_intent(question: str | None, conversation_memory=None) -> dict[str, Any]:
    """Return intent, optional kind, and subject. No database calls."""
    text = question or ""
    if is_show_more(text):
        prior = prior_anchor_question(conversation_memory, text)
        if prior:
            nested = classify_intent(prior, conversation_memory)
            kind = nested.get("kind")
            if nested.get("intent") == "recommend" and kind not in {None, "shortlist"}:
                continued = dict(nested)
                continued["more"] = True
                continued["source_question"] = prior
                return continued
        return {"intent": "neither", "kind": "show_more", "subject": None}
    place = extract_place(text)
    if place and _SNAPSHOT.search(text) and not _PREDICT.search(text):
        return {
            "intent": "predict",
            "kind": "snapshot",
            "subject": place,
        }
    if _PREDICT.search(text) and not _STATS.search(text):
        return {
            "intent": "predict",
            "kind": "trend",
            "subject": place,
        }
    if is_split_constraints(text):
        return {
            "intent": "recommend",
            "kind": "split",
            "subject": place,
            "budget_aed": parse_budget_aed(text),
        }
    if _AFFORD.search(text) and (place or parse_budget_aed(text)):
        beds = _BED.search(text)
        return {
            "intent": "recommend",
            "kind": "afford",
            "subject": place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": int(beds.group(1)) if beds else 1,
        }
    whole_market = (place or "").lower() in {"", "dubai"}
    if whole_market and _FAMILY.search(text) and _FAMILY_AREAS.search(text):
        focus = family_area_focus(text)
        focus["no_yield"] = bool(_YIELD.search(text))
        return {
            "intent": "recommend",
            "kind": "family_areas",
            "subject": None,
            "focus": focus,
        }
    if _YIELD.search(text):
        return {
            "intent": "recommend",
            "kind": "yield_areas",
            "subject": None if whole_market else place,
        }
    if _READY_HOME.search(text) and _OFFPLAN.search(text):
        return {
            "intent": "recommend",
            "kind": "ready_offplan",
            "subject": None if whole_market else place,
        }
    open_city = bool(_OPEN_BEST.search(text) or (
        _SUGGEST_PICK.search(text)
        and whole_market
        and not _BED.search(text)
        and parse_budget_aed(text) is None
    ))
    if open_city and not _on_screen(conversation_memory):
        return {
            "intent": "recommend",
            "kind": "open_best",
            "subject": None,
        }
    if _SHORTLIST.search(text) and _on_screen(conversation_memory):
        return {
            "intent": "recommend",
            "kind": "shortlist",
            "subject": "on_screen",
        }
    rental_only = bool(_RENT.search(text)) and not _BUY.search(text)
    if not rental_only and (
        _SUGGEST_PICK.search(text)
        or (
            _WANT_FILTER.search(text)
            and (_BUY.search(text) or _BED.search(text) or parse_budget_aed(text))
            and place
        )
    ):
        beds = _BED.search(text)
        return {
            "intent": "recommend",
            "kind": "filtered",
            "subject": place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": int(beds.group(1)) if beds else None,
            "months": parse_months(text),
        }
    return {"intent": "neither", "kind": None, "subject": place}
