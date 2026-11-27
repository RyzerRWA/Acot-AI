"""Questions to answer, not searches to run: calculations on the buyer's own numbers, laws and rules,
privacy, trust and personal-finance worries, challenges to an earlier figure, and "what have I told you".

ACOT QA (Nov 2026) found these turned into project lists or agent cards: a 70k rent on 1m got a project
ranking instead of 7%, "check the math" opened a listing search, a privacy question got agents. This
route runs before every search path and answers with one model call over the conversation so far.
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

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
    r"complain\w*|disputes?|my rights|power of attorney|evict\w*|lender clearance|advertis\w*|permits?|"
    r"registered (?:launch|project|development|developer))\b",
    re.I,
)
_PERSONAL = re.compile(
    r"\b(?:privacy|privately|private research|my (?:phone )?number|personal (?:data|details)|contact details|"
    r"bait[\s-]*and[\s-]*switch|scam|fraud|distrust|guarantee[ds]?|guaranteeing|emergency fund|savings|"
    r"should i wait|worried|afraid|fear|crash|liquid|downside case|information is private|private (?:information|data)|"
    r"verify (?:it|this|that|the claim)|before any referral|before deciding to speak|no calls|"
    # AI-058 / AI-006 / AI-097: decision frameworks, conflicting goals and stress tests are reasoning, not searches.
    r"tax residen\w*|non-?resident|personal inputs|meets? both|satisf\w* both|"
    # AI-086: asked to present a future sale as closed; the answer is a refusal and how to verify, not a sales list.
    r"as if (?:it|they) (?:has|have|had) already|next calendar month|invent(?:ed|ing|s)?\b|fabricat\w*|"
    r"(?:losing|lose|without)\s+(?:my\s+|our\s+)?(?:car|job|income|tenant)\w*)\b",
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


# A number sample the buyer made up ("grew from 100 to 130 in a hypothetical sample"): arithmetic, not a market lookup.
_HYPOTHETICAL = re.compile(r"\b(?:hypothetical|suppose|assume|assuming|imagine|in a sample|sample of)\b", re.I)
_ANY_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
# A promised return: answering "is it guaranteed" comes before any listing (ACOT QA AI-081, R10).
_GUARANTEE = re.compile(r"\bguarantee(?:s|d|ing)?\b", re.I)


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
    if _HYPOTHETICAL.search(text) and len(_ANY_NUMBER.findall(text)) >= 2:
        return "calc"
    if _GUARANTEE.search(text) and "?" in text:
        return "personal"
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
- Today is {today} (Dubai). A month or year before today is in the past; never call it the future.
- Answer the question asked in your first sentence. Do not list projects, listings or agents, and do not offer
  to find an agent, unless the buyer explicitly asked for one.
- Calculations: use only the numbers the buyer gave. State any assumption first, then each step as one short
  line, formula then result, and the final figure last in bold. Never state a figure before you have worked it
  out, and never correct yourself mid-answer. When comparing options, net every money difference the buyer gave
  (price paid, fees, income) into one figure per option; at equal exit prices a higher purchase price is a cost.
  Include acquisition fees in the cash invested and in "cash" yields when they are given. IRR: solve it
  numerically to two decimals and say it is the annual IRR. Gross yield = annual rent / price. NOI = rent actually collected (after vacancy)
  minus operating costs such as management, service charges and maintenance; financing is NOT part of NOI.
  Net yield = NOI / price (and NOI / (price + acquisition costs) when costs are given). Cash-on-cash =
  (NOI - annual debt service) / cash invested (price - loan + fees, or deposit + closing costs); "excluding principal paydown" means
  exactly this formula, with no principal added back, so never ask for loan details to compute it. Mortgage payment: standard amortising
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
- Residence through property: the 10-year Golden Visa route needs a property value of at least AED 2m (DLD;
  for a mortgage, bank evidence of the amount paid; joint ownership is assessed by share). Below that, DLD's
  Taskeen service issues a two-year property residence permit and treats individual and joint ownership
  differently. Never attribute any other threshold or term to DLD; tell the buyer to confirm on the current page.
- Other cities or countries: ACOT holds Dubai data only; say so in one line, then still give the decision
  framework and the personal inputs that change it.
- A project or developer you cannot confirm: say it cannot be verified here and name where to check (DLD /
  Dubai REST project and developer registers); never describe a different project with a similar name.
- Never promise or imply guaranteed returns. A recorded sale shows a past price, not current availability or a
  future yield.
- Recall and briefs: list only what the buyer stated in THIS conversation (shown below), the latest value
  winning when they revised it, and keep stated facts apart from assumptions. If they stated nothing, say so.
- Name official pages by title (for example "Dubai Land Department - Property Sale Registration"), no raw URLs.
- Under 220 words. Plain Markdown; short bullets are fine. Write maths as plain text (130 / 220 = 59.09%), never LaTeX.

{calcs}Conversation so far, oldest first (may be empty):
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
            lines.append(f"ACOT: {said[:1500]}")
    return "\n".join(lines) or "(none)"


async def advice_answer(question: str, conversation_memory, llm_generate) -> str | None:
    """The model's answer, or None (no model, or it failed) so the caller falls through to the normal paths."""
    if llm_generate is None:
        return None
    today = datetime.now(ZoneInfo("Asia/Dubai")).strftime("%-d %B %Y")
    history = _history(conversation_memory)
    calcs = ""
    if advice_kind(question) in {"calc", "challenge"} or _money_count(question) >= 2:
        from app.answers.finance_calc import PLAN_PROMPT, results_block

        try:
            block = results_block(await llm_generate(PLAN_PROMPT.format(history=history, question=question.strip())))
        except Exception as exc:
            print(f"advice calc plan failed: {exc}")
            block = None
        if block:
            calcs = ("Exact results, computed by code from the buyer's numbers. Quote these figures (rounded: money to "
                     "whole AED, % to 2 decimals) and never recompute them yourself:\n" + block + "\n\n")
    prompt = _PROMPT.format(history=history, question=question.strip(), today=today, calcs=calcs)
    try:
        text = await llm_generate(prompt)
    except Exception as exc:
        print(f"advice answer failed: {exc}")
        return None
    # The chat does not render LaTeX: "\\((130 - 100) / 100 = 30.00\\%\\)" showed raw (ACOT QA AI-020 retest).
    text = re.sub(r"\\[()\[\]]", "", text or "").replace("\\%", "%").strip()
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


# An ask FOR a document; "the brochure says 900 sq ft" is not one (ACOT QA AI-099).
_DOCUMENT = re.compile(
    r"\b(?:find|show|send|get|share|open|download|need|want|where(?:'s| is| can i find))\b[\s\S]{0,40}?\b(?:brochures?|floor[\s-]?plans?)\b"
    r"|\b(?:brochures?|floor[\s-]?plans?)\s+(?:for|of)\b",
    re.I,
)

DOCUMENT_NOTE = (
    "**Official brochure not on file.** I could not find the developer's brochure or floor plans for this, so I "
    "cannot cite a phase, tower or page. Any figures above come from recorded sales and project records, not "
    "the brochure. Ask the developer for the current official brochure."
)


def document_note(question: str | None, documents) -> str | None:
    """The unavailable state for a brochure or floor-plan ask that found no document (ACOT QA R06)."""
    return DOCUMENT_NOTE if _DOCUMENT.search(question or "") and not documents else None


_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_PERIOD = re.compile(
    rf"\b(?:{_MONTHS}\s+20\d\d|q[1-4]\s*20\d\d|(?:in|during|for|of)\s+20\d\d|20\d\d\s+(?:versus|vs\.?|and|to)\s+20\d\d)\b",
    re.I,
)
_RENT_ASK = re.compile(r"\b(?:rents?|rental contracts?|leases?|ejari|renewals?|new[\s-]contract)\b", re.I)
_PSF_ASK = re.compile(r"\b(?:aed\s*/\s*sq\.?\s*ft|per\s+sq\.?\s*ft|psf)\b", re.I)


def scope_note(question: str | None, answer: str | None) -> str | None:
    """Say plainly when a generic 12-month sales figure answered a narrower ask: a named month or quarter, rents
    or renewals, or AED/sq ft (ACOT QA R05, AI-016 to AI-025: the figures were right but not what was asked)."""
    text, said = question or "", answer or ""
    if "last 12 months" not in said.lower():
        return None
    missing = []
    period = _PERIOD.search(text)
    if period and period.group(0).lower() not in said.lower():
        missing.append(f"the period you named ({period.group(0).strip()})")
    if _RENT_ASK.search(text) and "sale" in said.lower() and not re.search(r"\brent", said, re.I):
        missing.append("rent or lease contracts (this used sale records)")
    if _PSF_ASK.search(text) and not re.search(r"sq\.?\s*ft", said, re.I):
        missing.append("price per sq ft (this is a sale-price median)")
    if not missing:
        return None
    return (
        "**Data scope:** the figures above are recorded Dubai Land Department sales for the last 12 months, not "
        + " or ".join(missing)
        + ". ACOT cannot filter that cohort yet, so treat them as context rather than the figure you asked for."
    )


def documents_block(info: dict | None) -> str | None:
    """The official documents for a brochure / floor-plan ask, as links (ACOT QA R06, AI-013)."""
    docs = (info or {}).get("documents") or []
    if not docs:
        return None
    bits = [b for b in (info.get("community"), f"developer {info['developer'].title()}" if info.get("developer") else None,
                        info.get("status"), f"completion {info['completion']}" if info.get("completion") else None) if b]
    lines = [f"**Official documents for {info['project']}**" + (f" ({', '.join(bits)})" if bits else "") + ":", ""]
    for doc in docs:
        label = doc["kind"] + (f" {doc['n']} of {doc['of']}" if doc["of"] > 1 else "")
        lines.append(f"- {label}: [{doc['file']}]({doc['url']})")
    lines += ["", "These are the developer's published files as held by Prop-AI. Floor plans are filed per building; "
              "ACOT cannot read page numbers inside the files, so open the floor plans to find the layout you need. "
              "Links are valid for about an hour; ask again for a fresh link."]
    return "\n".join(lines)
