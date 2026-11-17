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

_CHEAPEST = re.compile(
    r"\b(?:"
    r"cheapest|"
    r"most affordable|"
    r"lowest[- ]priced?|"
    r"least expensive|"
    r"cheap(?:est)? to live|"
    r"affordable to live"
    r")\b",
    re.I,
)
_CHEAPEST_SCOPE = re.compile(
    r"\b(?:communit(?:y|ies)|areas?|projects?|propert(?:y|ies)|homes?|"
    r"apartments?|to live|to buy|in dubai)\b",
    re.I,
)
_CHEAPEST_SHOWN = re.compile(
    r"^\s*(?:which(?: one| of (?:these|those|them))?|what)(?: is| are)?(?: the)? cheapest\b",
    re.I,
)
_OPEN_BEST = re.compile(
    r"\b(?:"
    r"best (?:propert(?:y|ies)|projects?)(?: in dubai)?|"
    r"which (?:one |of these )?(?:is |are )?(?:the )?best propert|"
    r"tell me which one is the best|"
    r"best areas?(?: to (?:buy|invest|live))?|"
    r"where (?:should|can) i (?:invest|buy|live|look)|"
    r"where to (?:invest|buy|live)|"
    r"where are people buying|"
    r"which area has the most sales|"
    r"busiest areas|"
    r"good investment"
    r")\b",
    re.I,
)

_SHORTLIST = re.compile(
    r"\b(?:"
    r"which (?:one |of these |of those )?(?:is|are) (?:the )?(?:best|better)|"
    r"which (?:project|property) is (?:the )?(?:best|better)|"
    r"tell(?: me)? which (?:project|property|one) is (?:the )?(?:best|better)|"
    r"which of these|"
    r"which one|"
    r"what do you think|"
    r"why (?:these|those|this|them)|"
    r"is (?:that|this|it) good|"
    r"are (?:these|those|they) good|"
    r"explain(?:\s+more)?|"
    r"recommend (?:one|one of these|the best|among these)|"
    r"which (?:of these )?(?:should|would) i|"
    r"what should i (?:pick|choose|buy)|"
    r"best (?:project|property) to buy"
    r")\b",
    re.I,
)

_SUGGEST_PICK = re.compile(
    r"\b(?:recommend|suggest)\b.{0,40}\b(?:one|best|buy|invest|property|project)\b|"
    r"\b(?:should i buy|worth buying|good buy|where should i (?:invest|buy|live))\b|"
    r"\b(?:show me (?:the )?best (?:propert|project)|best (?:propert(?:y|ies)|projects))",
    re.I,
)

_WANT_FILTER = re.compile(
    r"\b(?:i want|i need|looking for|i'?m looking)\b",
    re.I,
)
_SHOW_HOME = re.compile(
    r"\b(?:show|find|list|get)\s+(?:me\s+)?(?:the\s+)?(?:best\s+)?"
    r"(?:apartments?|flats?|villas?|townhouses?)\b|"
    r"\b(?:best\s+)?(?:apartments?|flats?|villas?|townhouses?)\s+for\s+(?:the\s+)?family\b|"
    r"\bmy budget\b.{0,60}\b(?:apartments?|flats?|villas?|homes?|family)\b",
    re.I,
)
_CASUAL_CHALLENGE = re.compile(
    r"\b(?:"
    r"so you(?:'re| are) telling|"
    r"are you saying|"
    r"you(?:'re| are) telling|"
    r"you mean|"
    r"does that mean|"
    r"is there (?:really )?(?:no|nothing)|"
    r"no (?:options?|homes?|apartments?|projects?|budget)|"
    r"nothing (?:at|under|in|for)|"
    r"no budget"
    r")\b",
    re.I,
)
_HOME = re.compile(r"\b(?:apartments?|flats?|homes?|propert(?:y|ies)|units?)\b", re.I)
_HOUSEHOLD = re.compile(
    r"\bfamily of\s+(a couple|couple|one|two|three|four|five|six|seven|eight|\d+)\b"
    r"|\b(?:for|with)\s+(a couple|couple|one|two|three|four|five|six|seven|eight|\d+)\s+"
    r"(?:members?|people|persons?|adults?)\b"
    r"|\b(a couple|couple|one|two|three|four|five|six|seven|eight|\d+)\s+"
    r"(?:members?|people|persons?|adults?)\b",
    re.I,
)
_HOUSEHOLD_WORDS = {
    "a couple": 2, "couple": 2, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}

_BUY = re.compile(r"\b(?:buy|buying|to buy|purchase|off-?plan|for sale)\b", re.I)
_RENT = re.compile(r"\b(?:rent|rental|to rent|yearly rent)\b", re.I)
_STUDIO = re.compile(r"\bstudios?\b", re.I)
_VILLA = re.compile(r"\bvillas?\b", re.I)
_FOUR_BED = re.compile(r"\b4\s*-?\s*(?:bed(?:room)?s?|br)\b", re.I)
_BED = re.compile(r"\b(\d+)\s*-?\s*(?:bed(?:room)?s?|br)\b", re.I)
_BED_NAMED = re.compile(
    r"\b(studio|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*-?\s*"
    r"(?:bed(?:room)?s?|br)\b",
    re.I,
)
_BED_NAMED_VALUES = {
    "studio": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_BUDGET = re.compile(
    r"(?:aed|dhs)\s*([\d,.]+)\s*(k|m|million|thousand)?|"
    r"\b([\d,.]+)\s*(k|m|million)\b|"
    r"(?:my\s+)?budget\s*(?:is|:|=)?\s*(?:aed|dhs)?\s*([\d,.]+)\s*(k|m|million|thousand)?",
    re.I,
)
_ONLY_TYPE = re.compile(
    r"\bonly\s+(villas?|townhouses?|apartments?|flats?)\b|"
    r"\b(villas?|townhouses?|apartments?|flats?)\s+only\b|"
    r"\b(villas?|townhouses?)\s+under\b",
    re.I,
)
_PLACE_NARROW = re.compile(
    r"\b(?:what about|how about|(?:look(?:ing)?|only) in)\b|"
    r"\bonly\b|"
    r"\b(?:and\s+)?in\b|"
    r"\b(?:show(?:\s+me)?|list|find)\s+(?:the\s+)?"
    r"(?:projects?|villas?|apartments?|them|those)?\s*in\b",
    re.I,
)
_OTHER_OPTIONS = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|please|pls|and)\s+)*"
    r"(?:(?:show|give|find|get)\s+(?:me\s+)?)?"
    r"(?:any\s+)?(?:other|more|different)\s+(?:options?|ones?|projects?|properties|choices?|results)|"
    r"(?:something|anything)\s+else|"
    r"any\s+others?"
    r"(?:\s+please|\s+pls)?"
    r"\s*[.!]?\s*$",
    re.I,
)
_TOO_EXPENSIVE = re.compile(
    r"\b(?:too\s+(?:expensiv\w*|pric\w*|costly)|out\s+of\s+(?:my\s+)?budget|"
    r"can'?t\s+afford|lower\s+please|i\s+don'?t\s+like\s+(?:these|those|them))\b",
    re.I,
)
_SHOW_IN_PLACE = re.compile(
    r"\b(?:show(?:\s+me)?|list|find)\s+(?:the\s+)?"
    r"(?:projects?|villas?|apartments?|them|those)?\s*in\s+(.+)$",
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
    r"typical.{0,48}\bsales?\b|"
    r"price per sq|per sq ?ft|sq ?ft|"
    r"latest data date|data date|last 12 months)\b",
    re.I,
)
_FAMILY = re.compile(r"\bfamil(?:y|ies)\b", re.I)
_FAMILY_AREAS = re.compile(
    r"\b(?:"
    r"(?:which|what|best|top|good|ideal|suit(?:able)?)\b.{0,48}"
    r"\b(?:areas?|communit(?:y|ies)|neighbourhoods?|neighborhoods?|places?)\b.{0,40}\bfamil"
    r"|\bfamil(?:y|ies)\b.{0,40}\b(?:areas?|communit(?:y|ies)|neighbourhoods?|neighborhoods?|places?)"
    r"|(?:looking for|want|need).{0,40}\b(?:best|top|good)\b.{0,40}"
    r"\b(?:areas?|communit(?:y|ies)|neighbourhoods?|neighborhoods?|places?)\b.{0,40}\bfamil"
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
# Core "show me more" lead. Trailing tweaks (cheap / best / high / typo words) are allowed
# for short follow-ups; a new place lock ("show me more in Dubai Marina") is not.
_SHOW_MORE_CORE = re.compile(
    r"^\s*(?:(?:please|kindly|pls|ok(?:ay)?)\s+)?"
    r"(?:(?:can|could|would)\s+you\s+)?"
    r"(?:show|give|list|find|send)\s+(?:\w{1,4}\s+)?more\b",
    re.I,
)
_SHOW_MORE_SHORT = re.compile(
    r"^\s*(?:any\s+)?more"
    r"(?:\s+(?:please|options?|projects?|properties|results|"
    r"like\s+(?:these|this|those)|"
    r"and\b.*|"
    r"(?:chea?p|best|high|premium|top)\w*))?"
    r"\s*[.!]?\s*$",
    re.I,
)
_CHEAP_TWEAK = re.compile(
    r"\b(?:chea?p(?:er|est)?|affordable|lower\s+budget)\b",
    re.I,
)
_PREMIUM_TWEAK = re.compile(
    r"\b(?:best|high(?:er|est)?|premium|top|luxury|pricier)\b",
    re.I,
)
_CHEAP_FOLLOW = re.compile(
    r"^\s*(?:(?:please|kindly|pls|ok(?:ay)?|and|&)\s+)*"
    r"(?:show(?:\s+\w{1,4})?\s+)?"
    r"(?:chea?p(?:er|est)?|more\s+affordable|lower\s+budget)"
    r"(?:\s+(?:ones?|options?|communities|areas?|projects?|please|pls))?"
    r"\s*[.!]?\s*$",
    re.I,
)
_PREMIUM_FOLLOW = re.compile(
    r"^\s*(?:(?:please|kindly|pls|ok(?:ay)?|and|&)\s+)*"
    r"(?:show(?:\s+\w{1,4})?\s+)?"
    r"(?:best|high(?:er|est)?|premium|top|luxury|more\s+expensive|pricier)"
    r"(?:\s+(?:ones?|options?|communities|areas?|projects?|please|pls))?"
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
    groups = match.groups()
    raw = ""
    suffix = ""
    for index in range(0, len(groups), 2):
        if groups[index]:
            raw = groups[index].replace(",", "")
            suffix = (groups[index + 1] or "").lower() if index + 1 < len(groups) else ""
            break
    try:
        number = float(raw)
    except ValueError:
        return None
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


def parse_bedrooms(question: str | None) -> list[int]:
    """Every bedroom count named in the question, in order, unique."""
    found: list[int] = []
    for match in _BED_NAMED.finditer(question or ""):
        token = match.group(1).lower()
        if token in _BED_NAMED_VALUES:
            beds = _BED_NAMED_VALUES[token]
        else:
            beds = int(token)
        if 0 <= beds <= 10 and beds not in found:
            found.append(beds)
    if (
        0 not in found
        and _STUDIO.search(question or "")
        and not _FAMILY_NOT_STUDIOS.search(question or "")
        and not (_VILLA.search(question or "") or _FOUR_BED.search(question or ""))
    ):
        found.append(0)
    return found


_PLACE_DETAIL = re.compile(
    r"\b(?:give(?:\s+me)?(?:\s+the)?\s+full\s+details?|full\s+details?|"
    r"tell\s+me\s+about|what(?:'s| is)|details?\s+(?:of|about|for|on))\b",
    re.I,
)
_DETAIL_TAIL = re.compile(
    r"(?:give(?:\s+me)?(?:\s+the)?\s+)?(?:full\s+details?|tell(?:\s+me)?\s+about|"
    r"details?\s+(?:of|about|for|on))\s+(.+)$",
    re.I,
)


def _place_named_in_text(place: str, text: str) -> bool:
    """True when the resolved place actually appears, or an alias of it does."""
    if not place or not text:
        return False
    lowered = text.lower()
    if re.search(rf"(?<![a-z0-9]){re.escape(place.lower())}(?![a-z0-9])", lowered):
        return True
    return any(
        ALIASES.get(key) == place
        and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered)
        for key in ALIASES
    )


def place_detail_subject(question: str | None, conversation_memory=None) -> str | None:
    """A known community in 'tell me about' / 'full details', not a project name."""
    text = question or ""
    if not _PLACE_DETAIL.search(text):
        return None
    if re.search(r"\b(?:compar\w*|vs\.?|versus|agents?|brokers?)\b", text, re.I):
        return None
    if project_detail_subject(text, conversation_memory):
        return None
    place = extract_place(text)
    if not place or place.lower() == "dubai":
        return None
    if not _place_named_in_text(place, text):
        return None
    return place


def project_detail_subject(question: str | None, conversation_memory=None) -> str | None:
    """A project named in a full-detail follow-up, preferring cards already on screen."""
    text = question or ""
    if not _PLACE_DETAIL.search(text):
        return None
    if re.search(r"\b(?:compar\w*|vs\.?|versus|agents?|brokers?)\b", text, re.I):
        return None
    # "budget details about the budget for each project" is a group attribute ask,
    # not a request for a project literally named "budget for each project".
    from app.retrieval.hybrid.question_text import is_group_attribute_question

    if is_group_attribute_question(text):
        return None
    shown = _shown_label(text, conversation_memory)
    if shown:
        for item in getattr(conversation_memory, "active_candidates", None) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name and name.lower() == shown.lower():
                return name
    matched = _DETAIL_TAIL.search(text)
    raw = (matched.group(1) if matched else "").strip(" .?!,;:'\"")
    raw = re.sub(r"^(?:the\s+)?(?:project\s+)?", "", raw, flags=re.I).strip()
    if not raw or len(raw) < 3:
        return None
    if re.search(
        r"\b(?:budget|budgets|price|prices|cost|costs|handover|bedrooms?)\b",
        raw,
        re.I,
    ) and re.search(
        r"\b(?:each|every|all|these|those|their|them)\b",
        text,
        re.I,
    ):
        return None
    place = extract_place(raw) or extract_place(text)
    if (
        place
        and place.lower() != "dubai"
        and _place_named_in_text(place, text)
    ):
        return None
    return raw


def _property_type_from_text(text: str | None) -> str | None:
    """Villa / townhouse / apartment when the question names a home type."""
    typed = _ONLY_TYPE.search(text or "")
    word = ""
    if typed:
        word = (typed.group(1) or typed.group(2) or typed.group(3) or "").lower()
    elif _VILLA.search(text or ""):
        word = "villa"
    elif re.search(r"\b(?:apartments?|flats?)\b", text or "", re.I):
        word = "apartment"
    elif re.search(r"\btownhouses?\b", text or "", re.I):
        word = "townhouse"
    if "villa" in word:
        return "villa"
    if "town" in word:
        return "townhouse"
    if "apartment" in word or "flat" in word:
        return "apartment"
    return None


def _shown_label(question: str | None, conversation_memory=None) -> str | None:
    """Longest on-screen name that appears in this follow-up."""
    if conversation_memory is None:
        return None
    text = (question or "").lower()
    labels: list[str] = []
    for item in getattr(conversation_memory, "active_candidates", None) or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "community", "sub_community", "area", "official_area"):
            value = item.get(key)
            if value:
                labels.append(str(value).strip())
    labels.sort(key=len, reverse=True)
    seen: set[str] = set()
    for name in labels:
        key = name.lower()
        if key in seen or len(name) < 4:
            continue
        seen.add(key)
        if key in text:
            return name
    return None


def _place_after_in(question: str | None) -> str | None:
    """'show me in Urban Life' → Urban Life, even when it is not a known alias."""
    matched = _SHOW_IN_PLACE.search(question or "")
    if not matched:
        return None
    chunk = re.sub(r"\s+", " ", matched.group(1)).strip(" .?!,;:'\"")
    if not chunk or len(chunk) < 4:
        return None
    return extract_place(chunk) or chunk


def extract_place(question: str | None) -> str | None:
    text = question or ""
    lowered = text.lower()
    for key in sorted(ALIASES, key=len, reverse=True):
        if not re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}[\s-]+(?:[0-9]+|ii|iii|iv)\b", lowered):
            continue
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
    """True when the turn asks for more of the previous search.

    Allows short trailing tweaks ("and cheap", "and best and high", typos)
    and casual phrasing ("any other options", "something else").
    Rejects a new place lock like "show me more in Dubai Marina".
    """
    text = (question or "").strip()
    if not text:
        return False
    if _OTHER_OPTIONS.match(text) or _SHOW_MORE_SHORT.match(text):
        return True
    if not _SHOW_MORE_CORE.search(text):
        return False
    # A new "in <place>" lock is a fresh narrow, not plain show-more.
    if re.search(r"\bin\b", text, re.I) and extract_place(text):
        return False
    # Keep this as a short follow-up, not a full new question.
    return len(text.split()) <= 12


def prefers_cheaper(question: str | None) -> bool:
    """True when the turn asks for cheaper / more affordable options (typos included)."""
    text = question or ""
    return bool(_CHEAP_TWEAK.search(text) or _TOO_EXPENSIVE.search(text))


def prefers_premium(question: str | None) -> bool:
    """True when the turn asks for best / higher / premium options."""
    text = question or ""
    if prefers_cheaper(text):
        return False
    return bool(_PREMIUM_TWEAK.search(text))


_MORE_IN_QUESTION = re.compile(
    r"show additional options|not the same list again|"
    r"show(?:\s+\w{1,4})?\s+more|(?:any\s+)?more\s+"
    r"(?:options?|projects?|properties|results)|"
    r"(?:any|other)\s+(?:options?|ones?)|"
    r"(?:something|anything)\s+else",
    re.I,
)


def wants_more_results(question: str | None) -> bool:
    """True for a 'show me more' turn, including the rewritten follow-up."""
    if is_show_more(question):
        return True
    if _OTHER_OPTIONS.match((question or "").strip()):
        return True
    return bool(_MORE_IN_QUESTION.search(question or ""))


def _looks_like_new_home_search(question: str | None) -> bool:
    """True for a fresh budget / apartment / family ask, not a short list tweak."""
    text = question or ""
    if not text:
        return False
    budget = parse_budget_aed(text)
    home = bool(_HOME.search(text) or _SHOW_HOME.search(text) or _FAMILY.search(text))
    if budget and home and len(text.split()) >= 6:
        return True
    if _SHOW_HOME.search(text) and (budget or _FAMILY.search(text)):
        return True
    return False


def _continue_last_recommend(question: str | None, conversation_memory=None) -> dict[str, Any] | None:
    """Reuse the last recommend thread for show-more / cheap / best follow-ups."""
    if conversation_memory is None:
        return None
    text = (question or "").strip()
    if not text:
        return None
    # A full new home search must not stay stuck on the previous candidate list.
    if _looks_like_new_home_search(text):
        return None
    more = is_show_more(text) or wants_more_results(text)
    cheap = prefers_cheaper(text) or bool(_CHEAP_FOLLOW.match(text))
    premium = prefers_premium(text) or bool(_PREMIUM_FOLLOW.match(text))
    if not (more or cheap or premium):
        return None
    # Tweak-only follow-ups must look like a continuation, not a brand-new search.
    if (cheap or premium) and not more:
        if not (_CHEAP_FOLLOW.match(text) or _PREMIUM_FOLLOW.match(text)):
            if not re.search(r"\b(?:ones?|options?|communities|areas?|these|those|them)\b", text, re.I):
                if len(text.split()) > 4:
                    return None
    prior = prior_anchor_question(conversation_memory, text)
    if not prior:
        return None
    nested = classify_intent(prior, None)
    kind = nested.get("kind")
    if nested.get("intent") != "recommend" or kind in {None, "shortlist"}:
        return None
    continued = dict(nested)
    continued["source_question"] = prior
    if more:
        continued["more"] = True
    if cheap:
        continued["prefer_cheaper"] = True
        continued.pop("prefer_premium", None)
        if kind == "family_areas":
            continued["more"] = True
    elif premium:
        continued["prefer_premium"] = True
        continued.pop("prefer_cheaper", None)
        if kind == "family_areas":
            continued["more"] = True
    return continued


def _named_on_screen(conversation_memory) -> bool:
    items = getattr(conversation_memory, "active_candidates", None) or []
    return any(
        isinstance(item, dict) and item.get("name")
        for item in items
    )


def reuse_last_search(question: str | None, conversation_memory=None) -> bool:
    """True when this turn should continue the last project or listing list."""
    if conversation_memory is None:
        return False
    if not (is_show_more(question) or wants_more_results(question)):
        return False
    if _named_on_screen(conversation_memory) or _on_screen(conversation_memory):
        return True
    return bool(prior_anchor_question(conversation_memory, question))


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


def household_size(question: str | None) -> int | None:
    """People named in the question, such as '2 members' or 'family of 4'."""
    match = _HOUSEHOLD.search(question or "")
    if not match:
        return None
    raw = next(group for group in match.groups() if group).lower().strip()
    if raw in _HOUSEHOLD_WORDS:
        return _HOUSEHOLD_WORDS[raw]
    return int(raw)


def _prior_budget_aed(conversation_memory, current: str | None = None) -> int | None:
    """Budget from this turn, else from the last real search question."""
    budget = parse_budget_aed(current)
    if budget:
        return budget
    prior = prior_anchor_question(conversation_memory, current)
    return parse_budget_aed(prior) if prior else None


def continue_recommend(question: str | None, conversation_memory=None) -> dict[str, Any] | None:
    """Budget / villa / area / cheaper / premium / casual challenge stay on the thread."""
    if conversation_memory is None:
        return None
    text = question or ""
    prior = prior_anchor_question(conversation_memory, text)
    if not prior:
        return None
    budget = parse_budget_aed(text)
    typed = _ONLY_TYPE.search(text)
    cheap = prefers_cheaper(text) or bool(_CHEAP_FOLLOW.match(text))
    premium = prefers_premium(text) or bool(_PREMIUM_FOLLOW.match(text))
    challenge = bool(_CASUAL_CHALLENGE.search(text))
    place = (
        _shown_label(text, conversation_memory)
        or extract_place(text)
        or _place_after_in(text)
    )
    place_narrow = bool(place and _PLACE_NARROW.search(text))
    # Short place-only follow-ups: "and in marina?", "jvc?", "dubai hills?"
    if (
        place
        and not place_narrow
        and not budget
        and not typed
        and not _looks_like_new_home_search(text)
        and len(text.split()) <= 6
        and not re.search(r"\b(?:bed|apartment|villa|buy|rent|compare)\b", text, re.I)
    ):
        place_narrow = True
    if challenge and place:
        nested = classify_intent(prior, None)
        inherited = budget or parse_budget_aed(prior) or nested.get("budget_aed")
        beds = parse_bedrooms(text) or parse_bedrooms(prior) or (
            [nested.get("bedrooms")] if nested.get("bedrooms") is not None else []
        )
        people = household_size(text) or household_size(prior) or nested.get("household")
        if not beds and (_FAMILY.search(prior) or people):
            beds = [bedrooms_for_household(people or 4) or 3]
        ptype = _property_type_from_text(text) or _property_type_from_text(prior) or nested.get("property_type")
        if not ptype and re.search(r"\bapartments?\b", prior, re.I):
            ptype = "apartment"
        # Casual "so nothing at my budget in Business Bay?" → check that place at the budget.
        route = {
            "intent": "recommend",
            "kind": "afford" if inherited else "filtered",
            "subject": place,
            "budget_aed": inherited,
            "bedrooms": beds[0] if beds else (2 if ptype == "apartment" else 3),
            "household": people,
            "source_question": prior,
        }
        if ptype:
            route["property_type"] = ptype
        if not inherited:
            route["kind"] = "filtered"
        return route
    if not (budget or typed or place_narrow or cheap or premium):
        return None
    nested = classify_intent(prior, None)
    if nested.get("intent") != "recommend" or nested.get("kind") in {None, "shortlist"}:
        return None
    continued = dict(nested)
    continued["source_question"] = prior
    continued.pop("more", None)
    if budget:
        continued["budget_aed"] = budget
    if cheap:
        continued["prefer_cheaper"] = True
        continued.pop("prefer_premium", None)
        if continued.get("kind") == "family_areas":
            continued["more"] = True
    elif premium:
        continued["prefer_premium"] = True
        continued.pop("prefer_cheaper", None)
        if continued.get("kind") == "family_areas":
            continued["more"] = True
    ptype = _property_type_from_text(text)
    if ptype:
        focus = dict(continued.get("focus") or {})
        if ptype == "villa":
            focus["want_villas"] = True
        continued["property_type"] = ptype
        continued["focus"] = focus
    if place_narrow:
        continued["subject"] = place
    return continued


def bedrooms_for_household(count: int | None) -> int | None:
    """Studio for 1 person, 1 bedroom for 2, then one more bedroom per two people."""
    if count is None or count < 1:
        return None
    if count == 1:
        return 0
    return min(4, (count + 1) // 2)


_NO_PREFERENCE_REPLY = re.compile(
    r"^\s*(?:i\s+)?(?:don'?t\s+know|do\s+not\s+know|no\s+idea|not\s+sure|"
    r"anything|anywhere|wherever|any(?:\s+area|\s+community|\s+place)?|"
    r"surprise\s+me|no\s+preference|whatever|idk|dunno|"
    r"i\s+don'?t\s+have\s+(?:any\s+)?(?:idea|preference))\s*[.!]?\s*$",
    re.I,
)
_HOME_BROWSE_HINT = re.compile(
    r"\b(?:apartments?|flats?|villas?|townhouses?|penthouses?|"
    r"bedrooms?|beds?|studio|studios)\b",
    re.I,
)


def is_no_preference_reply(question: str | None) -> bool:
    """True when the user declines to pick an area after a clarify ask."""
    text = (question or "").strip()
    if not text or len(text) > 80:
        return False
    return bool(_NO_PREFERENCE_REPLY.match(text))


def is_incomplete_home_browse(question: str | None) -> bool:
    """Beds/type ask with no Dubai community and no budget — needs one clarify ask."""
    text = (question or "").strip()
    if not text:
        return False
    if parse_budget_aed(text):
        return False
    place = extract_place(text)
    if place and place.lower() != "dubai":
        return False
    has_beds = bool(parse_bedrooms(text))
    has_home = bool(
        _HOME_BROWSE_HINT.search(text)
        or _HOME.search(text)
        or _SHOW_HOME.search(text)
        or _WANT_FILTER.search(text)
    )
    if not (has_beds or has_home):
        return False
    if not (
        _WANT_FILTER.search(text)
        or _SHOW_HOME.search(text)
        or re.search(r"\b(?:looking|want|need|show|find|list)\b", text, re.I)
    ):
        return False
    # Invest / open-buy keeps search-first, not clarify.
    if _OPEN_BEST.search(text) or _FAMILY.search(text):
        return False
    return True


def classify_intent(question: str | None, conversation_memory=None) -> dict[str, Any]:
    """Return intent, optional kind, and subject. No database calls."""
    text = question or ""
    continued = _continue_last_recommend(text, conversation_memory)
    if continued is not None:
        return continued
    if is_show_more(text):
        return {"intent": "neither", "kind": "show_more", "subject": None}
    place = extract_place(text)
    if (
        conversation_memory is not None
        and (not place or place.lower() == "dubai")
        and re.search(
            r"\b(?:which|suitable|best for|cheapest|most affordable)\b",
            text,
            re.I,
        )
        and not _FAMILY_AREAS.search(text)
        and not _OPEN_BEST.search(text)
    ):
        prior = prior_anchor_question(conversation_memory, text)
        inherited = extract_place(prior) if prior else None
        if inherited and inherited.lower() not in {"", "dubai"}:
            place = inherited
    if place and _SNAPSHOT.search(text) and not _PREDICT.search(text):
        route = {
            "intent": "predict",
            "kind": "snapshot",
            "subject": place,
        }
        beds = parse_bedrooms(text)
        if beds:
            route["bedrooms"] = beds
        return route
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
        named_beds = parse_bedrooms(text)
        return {
            "intent": "recommend",
            "kind": "afford",
            "subject": place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": named_beds[0] if named_beds else 1,
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
    if (
        _FAMILY.search(text)
        and not _FAMILY_AREAS.search(text)
        and re.search(r"\b(?:project|propert(?:y|ies)|suitable|which)\b", text, re.I)
    ):
        prior = prior_anchor_question(conversation_memory, text)
        locked = None if whole_market else place
        if not locked:
            locked = extract_place(prior) if prior else None
        if locked and str(locked).lower() not in {"", "dubai"}:
            named = parse_bedrooms(text)
            people = household_size(text) or 4
            return {
                "intent": "recommend",
                "kind": "filtered",
                "subject": locked,
                "bedrooms": named[0] if named else 3,
                "household": people,
                "source_question": prior or text,
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
    rental_only = bool(_RENT.search(text)) and not _BUY.search(text)
    if not rental_only and _CHEAPEST.search(text):
        scoped = bool(_CHEAPEST_SCOPE.search(text))
        if _on_screen(conversation_memory) and not scoped:
            return {
                "intent": "recommend",
                "kind": "shortlist",
                "subject": "on_screen",
            }
        if scoped or _CHEAPEST_SHOWN.match(text):
            return {
                "intent": "recommend",
                "kind": "cheapest",
                "subject": None if whole_market else place,
            }
    people = household_size(text)
    named_beds = parse_bedrooms(text)
    home_for_people = (
        people is not None
        and not (bool(_RENT.search(text)) and not _BUY.search(text))
        and _HOME.search(text)
        and not _FAMILY_AREAS.search(text)
        and not named_beds
    )
    if home_for_people:
        return {
            "intent": "recommend",
            "kind": "filtered",
            "subject": None if whole_market else place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": bedrooms_for_household(people),
            "household": people,
        }
    open_city = bool(
        (_OPEN_BEST.search(text) or _SUGGEST_PICK.search(text))
        and whole_market
        and not named_beds
        and parse_budget_aed(text) is None
    )
    # Cards on screen only divert a "which of these" shortlist, not a new city-wide where-to-buy.
    if open_city and not (_on_screen(conversation_memory) and _SHORTLIST.search(text)):
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
    typed_home = _ONLY_TYPE.search(text)
    if (
        not rental_only
        and typed_home
        and place
        and not whole_market
    ):
        ptype = _property_type_from_text(text)
        people = household_size(text)
        route = {
            "intent": "recommend",
            "kind": "filtered",
            "subject": place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": named_beds[0] if named_beds else None,
            "household": people,
            "months": parse_months(text),
            "property_type": ptype,
        }
        if ptype == "villa":
            route["focus"] = {"want_villas": True}
        return route
    if not rental_only and (
        _SUGGEST_PICK.search(text)
        or (
            _WANT_FILTER.search(text)
            and (_BUY.search(text) or named_beds or parse_budget_aed(text))
            and place
        )
    ):
        people = household_size(text)
        ptype = _property_type_from_text(text)
        route = {
            "intent": "recommend",
            "kind": "filtered",
            "subject": place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": named_beds[0] if named_beds else None,
            "household": people,
            "months": parse_months(text),
        }
        if ptype:
            route["property_type"] = ptype
            if ptype == "villa":
                route["focus"] = {"want_villas": True}
        return route
    # Citywide / family apartment asks: "My budget is 1.5M, show me the best apartments for the family"
    if not rental_only and (
        _SHOW_HOME.search(text)
        or (
            _FAMILY.search(text)
            and _HOME.search(text)
            and (parse_budget_aed(text) or named_beds)
        )
    ):
        people = household_size(text) or (4 if _FAMILY.search(text) else None)
        ptype = _property_type_from_text(text)
        beds = named_beds[0] if named_beds else None
        # Family apartment asks default to 3 beds unless the user named a count.
        if beds is None and _FAMILY.search(text):
            beds = 3
        elif beds is None and people:
            beds = bedrooms_for_household(people)
        route = {
            "intent": "recommend",
            "kind": "filtered",
            "subject": None if whole_market else place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": beds,
            "household": people,
            "months": parse_months(text),
        }
        if ptype:
            route["property_type"] = ptype
            if ptype == "villa":
                route["focus"] = {"want_villas": True}
        return route
    followed = continue_recommend(text, conversation_memory)
    if followed:
        return followed
    ptype = _property_type_from_text(text)
    if not rental_only and ptype in {"villa", "townhouse"}:
        people = household_size(text)
        route = {
            "intent": "recommend",
            "kind": "filtered",
            "subject": None if whole_market else place,
            "budget_aed": parse_budget_aed(text),
            "bedrooms": named_beds[0] if named_beds else None,
            "household": people,
            "months": parse_months(text),
            "property_type": ptype,
        }
        if ptype == "villa":
            route["focus"] = {"want_villas": True}
        return route
    return {"intent": "neither", "kind": None, "subject": place}
