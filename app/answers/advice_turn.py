"""Questions to answer, not searches to run: calculations on the buyer's own numbers, laws and rules,
privacy, trust and personal-finance worries, challenges to an earlier figure, and "what have I told you".

ACOT QA (Nov 2026) found these turned into project lists or agent cards: a 70k rent on 1m got a project
ranking instead of 7%, "check the math" opened a listing search, a privacy question got agents. This
route runs before every search path and answers with one model call over the conversation so far.
"""
from __future__ import annotations

import re

# An explicit ask for homes or agents: the search paths own these.
_LISTING_ASK = re.compile(
    r"\b(?:show|find|list|search|recommend|suggest|give me|which)\b[\s\S]{0,50}?"
    r"\b(?:projects?|propert(?:y|ies)|apartments?|flats?|units?|villas?|townhouses?|homes?|listings?|options|areas?|communit(?:y|ies))\b",
    re.I,
)
_AGENT_ASK = re.compile(
    r"\b(?:find|show|connect|introduce|recommend|suggest|give|list)\s+(?:me\s+|us\s+)?(?:with\s+)?(?:an?\s+|the\s+|some\s+|\d+\s+)?"
    r"(?:\w+\s+){0,2}?(?:agents?|brokers?)\b|"
    r"\b(?:top|best|good)\s+(?:\w+\s+)?(?:agents?|brokers?)\b|"
    r"\b(?:need|want)\s+(?:an?\s+|to\s+(?:speak|talk)\s+(?:to|with)\s+an?\s+)(?:agent|broker)\b",
    re.I,
)

# Money and rates only: "2 bed" or "25 years" alone are not a calculation.
_MONEY = re.compile(
    r"(?:aed\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|k\b|m\b|mn\b|million\b|thousand\b)|aed\s*\d[\d,]*|\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b",
    re.I,
)
_CALC_METRIC = re.compile(
    r"\b(?:yields?|noi|net operating|cash[\s-]*on[\s-]*cash|roi|returns?|mortgage|monthly payments?|payment|"
    r"instal(?:l)?ments?|interest|amorti[sz]\w*|ltv|debt service|service charges?|per\s*sq\.?\s*ft|psf|price per|"
    r"vacancy|break[\s-]*even|math|maths|reserve|net|income|appreciation|capital gain|exit price)\b|/\s*sq\.?\s*ft",
    re.I,
)
_CALC_CUE = re.compile(
    r"\b(?:calculate|compute|work out|check|recheck|stress|compare (?:the )?(?:yields?|returns?)|"
    r"what(?:'s| is| are| would be| will be)\s+(?:my|the|its|their)|give (?:me )?(?:the )?(?:gross|net)?\s*(?:yield|noi)|"
    r"how much (?:is|are|would|will|do|does)|explain what|how (?:does|do|would|will)\b[\s\S]{0,40}\baffect)\b",
    re.I,
)
_REGULATED = re.compile(
    r"\b(?:laws?|legal(?:ly)?|regulat\w*|rera|brn|escrow|title deed|oqood|spa|tenancy contract|sale contract|"
    r"purchase agreement|ejari|refunds?|compensation|handover is \w+ late|months late|delay(?:ed)? handover|"
    r"complain\w*|disputes?|my rights|power of attorney|evict\w*|lender clearance|advertis\w*|permits?)\b",
    re.I,
)
_PERSONAL = re.compile(
    r"\b(?:privacy|privately|private research|my (?:phone )?number|personal (?:data|details)|contact details|"
    r"bait[\s-]*and[\s-]*switch|scam|fraud|distrust|guarantee[ds]?|guaranteeing|emergency fund|savings|"
    r"should i wait|worried|afraid|fear|crash|liquid|downside case|information is private|private (?:information|data)|"
    r"verify (?:it|this|that|the claim)|before any referral|before deciding to speak|no calls)\b",
    re.I,
)
_CHALLENGE = re.compile(
    r"\b(?:your (?:example|answer|figure|number|calculation|math)|you said|you told me|that(?:'s| is) wrong|"
    r"check (?:the|your|my) math|is that (?:right|correct)|correct (?:the|your|that|this)|recheck|re-check)\b",
    re.I,
)
_RECALL = re.compile(
    r"\b(?:running brief|(?:repeat|recap|summari[sz]e|confirm)\b[\s\S]{0,30}\b(?:brief|requirements|needs|criteria)|"
    r"what (?:budget|requirements?|criteria) have i|what have i (?:told|given) you|"
    r"recalled facts|from assumptions|my (?:family )?brief|the brief|revised cap|what changed|what else stays|"
    r"areas are left|shortlist criteria|update the (?:shortlist|criteria|brief)|restate)\b",
    re.I,
)


# "under 1.2m", "between 1.5m and 2m": a search filter, not an input to calculate with.
_FILTER_AMOUNT = re.compile(
    r"\bbetween\s+(?:aed\s*)?[\d.,]+\s*[a-z%]*\s+and\s+(?:aed\s*)?[\d.,]+\s*[a-z%]*|"
    r"\b(?:under|below|above|over|less than|more than|up to|max(?:imum)?|budget(?: of| is)?)\s+(?:aed\s*)?[\d.,]+\s*[a-z%]*",
    re.I,
)
_HOW_TO = re.compile(r"\b(?:how (?:to|do|can|should)|verify|which documents|what documents)\b", re.I)


def _money_count(text: str) -> int:
    return len(_MONEY.findall(_FILTER_AMOUNT.sub(" ", text)))


def advice_kind(question: str | None) -> str | None:
    """"recall", "challenge", "calc", "regulated", "personal", or None for a normal search turn."""
    text = (question or "").strip()
    if not text:
        return None
    if _RECALL.search(text):
        return "recall"
    if _CHALLENGE.search(text):
        return "challenge"
    if _AGENT_ASK.search(text):
        return None
    money = _money_count(text)
    if (
        money >= 2
        and _CALC_METRIC.search(text)
        and (_CALC_CUE.search(text) or money >= 3)
        and not _LISTING_ASK.search(text)
    ):
        return "calc"
    if _REGULATED.search(text) and (_HOW_TO.search(text) or not _LISTING_ASK.search(text)):
        return "regulated"
    if _LISTING_ASK.search(text):
        return None
    if _PERSONAL.search(text):
        return "personal"
    return None


_PROMPT = """You are ACOT, a Dubai property assistant for buyers. Answer the buyer's latest message directly.

Rules:
- Answer the question asked in your first sentence. Do not list projects, listings or agents, and do not offer
  to find an agent, unless the buyer explicitly asked for one.
- Calculations: use only the numbers the buyer gave. Show each step as one short line, formula then result,
  and state any assumption. Gross yield = annual rent / price. NOI = rent actually collected (after vacancy)
  minus operating costs such as management, service charges and maintenance; financing is NOT part of NOI.
  Net yield = NOI / price (and NOI / (price + acquisition costs) when costs are given). Cash-on-cash =
  (NOI - annual debt service) / cash invested (deposit + closing costs). Mortgage payment: standard amortising
  formula with monthly rate = annual rate / 12. Round money to whole AED, percentages to 2 decimals.
- If the buyer challenges a figure, including one from earlier in this conversation, recompute it from the
  source numbers, say plainly whether it was wrong, give the corrected figure and what it changes.
- Laws, fees, visas, broker licensing, escrow, handover delays, refunds, tenancy: give general guidance, name
  the official body and service (Dubai Land Department, RERA, Ejari, Rental Dispute Settlement Centre, CBUAE),
  say rules change and to check the current official page, and that this is not legal advice. When asked,
  separate what is law, a regulator rule, a brokerage's policy and a website's own rule. Never invent article
  numbers, fees or deadlines you are not sure of.
- Privacy, trust and risk: respect the buyer's stated preference (for example researching privately). Explain
  what they can check themselves first (DLD transaction records, the Madmoun QR code on ads, title deed and
  Ejari checks). An agent introduction only happens if the buyer chooses it.
- Personal finance: keep separate budgets separate, never turn savings into a property budget, and raise the
  emergency reserve and the CBUAE debt-burden limit when relevant.
- Never promise or imply guaranteed returns. A recorded sale shows a past price, not current availability or a
  future yield.
- Recall and briefs: list only what the buyer stated in THIS conversation (shown below), the latest value
  winning when they revised it, and keep stated facts apart from assumptions. If they stated nothing, say so.
- Name official pages by title (for example "Dubai Land Department - Property Sale Registration"), no raw URLs.
- Under 220 words. Plain Markdown; short bullets are fine.

Conversation so far, oldest first (may be empty):
{history}

Buyer's latest message:
{question}
"""


def _history(conversation_memory, limit: int = 8) -> str:
    turns = list(getattr(conversation_memory, "turns", None) or [])[-limit:]
    lines = []
    for turn in turns:
        asked = str(turn.get("user_question") or "").strip()
        said = str(turn.get("answer") or "").strip()
        if asked:
            lines.append(f"Buyer: {asked}")
        if said:
            lines.append(f"ACOT: {said[:700]}")
    return "\n".join(lines) or "(none)"


async def advice_answer(question: str, conversation_memory, llm_generate) -> str | None:
    """The model's answer, or None (no model, or it failed) so the caller falls through to the normal paths."""
    if llm_generate is None:
        return None
    prompt = _PROMPT.format(history=_history(conversation_memory), question=question.strip())
    try:
        text = await llm_generate(prompt)
    except Exception as exc:
        print(f"advice answer failed: {exc}")
        return None
    text = (text or "").strip()
    return text or None


_AVAILABILITY = re.compile(
    r"\b(?:vacant|available (?:units?|now|to view)|availability|can i (?:actually )?view|viewings?|book a view\w*|"
    r"move[\s-]*in|still (?:for sale|available)|in stock)\b",
    re.I,
)

AVAILABILITY_NOTE = (
    "**Availability unverified.** ACOT shows recorded sales and project records, not a dated list of vacant "
    "units, so a sale or project card does not mean a unit is free to buy or view today. A viewing is confirmed "
    "only once the developer or a licensed agent books it."
)


def availability_note(question: str | None) -> str | None:
    """The honesty line for an availability or viewing ask, whatever path answered it."""
    return AVAILABILITY_NOTE if _AVAILABILITY.search(question or "") else None


_DOCUMENT = re.compile(r"\b(?:brochures?|floor[\s-]?plans?)\b", re.I)

DOCUMENT_NOTE = (
    "**Official brochure not on file.** I could not find the developer's brochure or floor plans for this, so I "
    "cannot cite a phase, tower or page. Any figures above come from recorded sales and project records, not "
    "the brochure. Ask the developer for the current official brochure."
)


def document_note(question: str | None, documents) -> str | None:
    """The unavailable state for a brochure or floor-plan ask that found no document (ACOT QA R06)."""
    return DOCUMENT_NOTE if _DOCUMENT.search(question or "") and not documents else None
