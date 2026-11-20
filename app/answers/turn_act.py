"""Turn-act gate: casual speech vs continue vs new task.

Production dialogue managers classify the *act* before intent/slots so
acknowledgements never trigger SQL, recommend, or sticky shortlist re-ranks.
"""
from __future__ import annotations

import re
from typing import Any

# Pure social / ack lines — no place, beds, budget, or data verbs.
_CASUAL_FULL = re.compile(
    r"^\s*(?:"
    r"(?:ok(?:ay)?|alright|sure)(?:\s*,?\s*|\s+)(?:that(?:'s| is)\s+)?(?:cool|great|perfect|nice|good|awesome)?|"
    r"(?:that(?:'s| is)\s+)?(?:cool|great|perfect|nice|awesome|amazing)|"
    r"sounds?\s+good|"
    r"got\s+it|"
    r"ok(?:ay)?|cool|great|perfect|nice|awesome|amazing|"
    r"thanks?(?:\s+you)?|thank\s+you|thx|ty|"
    r"cheers|alright|sure|yep|yeah|"
    r"bye|goodbye|see\s+ya|"
    r"hi+|hey+|hello+"
    r")(?:\s*[.!])?\s*$",
    re.I,
)

# If any of these appear, the line is a real ask (or slot fill), not casual.
_NOT_CASUAL = re.compile(
    r"\b(?:"
    r"show|find|list|search|more|compare|recommend|suggest|best|top|"
    r"projects?|propert(?:y|ies)|apartments?|villas?|townhouses?|flats?|"
    r"agents?|brokers?|transactions?|sales?|sold|budget|aed|bedroom|bedrooms|"
    r"handover|developer|price|prices|rent|buy|invest|"
    r"these|those|them|which|marina|jvc|jlt|downtown|palm|bay|"
    r"family|studio|studios"
    r")\b",
    re.I,
)


def is_casual_speech(question: str | None) -> bool:
    """True when the whole utterance is social/ack and not a data request."""
    text = (question or "").strip()
    if not text:
        return False
    lowered = text.lower()
    # Thanks / bye with polite filler still casual unless a data ask is attached.
    if re.search(r"\b(?:thanks?|thank you|thx|ty|cheers)\b", lowered):
        if not re.search(
            r"\b(?:show|find|list|projects?|agents?|sales?|bedroom|marina|jvc|"
            r"recommend|suggest|compare|transactions?)\b",
            lowered,
        ):
            return True
    if _NOT_CASUAL.search(text):
        return False
    if _CASUAL_FULL.match(text):
        return True
    words = re.findall(r"[a-z']+", lowered)
    if not words or len(words) > 6:
        return False
    casual_words = {
        "ok", "okay", "alright", "sure", "yes", "yep", "yeah",
        "cool", "nice", "great", "perfect", "awesome", "amazing",
        "thanks", "thank", "you", "thx", "ty", "cheers",
        "got", "it", "that", "thats", "that's", "is", "good",
        "sounds", "bye", "goodbye", "hi", "hey", "hello",
        "hmm", "hmmm", "ah", "oh", "um", "uh",
        "a", "lot", "so", "much", "for", "the", "help", "your",
    }
    return all(word in casual_words for word in words)


def classify_turn_act(
    question: str | None,
    state: dict[str, Any] | None = None,
) -> str:
    """Return casual | continue_candidate | new_task_candidate.

    Only *casual* is a hard stop for this phase. Other labels are hints for
    callers; intent/slot resolve still decides continue vs new task.
    """
    if is_casual_speech(question):
        return "casual"
    text = question or ""
    # Short deixis / paging with an active task → likely continue.
    if state and state.get("action"):
        if re.search(
            r"\b(?:more|these|those|them|same|again|instead|what about|how about)\b",
            text,
            re.I,
        ) or len(re.findall(r"[a-z0-9]+", text.lower())) <= 4:
            return "continue_candidate"
    return "new_task_candidate"


def casual_kind(question: str | None) -> str:
    """Map casual speech to small_talk answer kind."""
    text = (question or "").lower()
    if re.search(r"\b(?:thanks?|thank you|thx|ty|cheers)\b", text):
        return "thanks"
    if re.search(r"\b(?:bye|goodbye|see ya)\b", text):
        return "bye"
    if re.fullmatch(r"\s*(?:hi+|hey+|hello+)\s*[.!?]?\s*", text or ""):
        return "greeting"
    return "ack"
