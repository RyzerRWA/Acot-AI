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


def classify_intent(question: str | None, conversation_memory=None) -> dict[str, Any]:
    """Return intent, optional kind, and subject. No database calls."""
    text = question or ""
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
