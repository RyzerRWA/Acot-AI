"""Answer contract: restated ask → motive → grounded facts → next step.

User-facing tone (every answer type):
1. Understood — mirror the ask in a full sentence (not a stub label)
2. Motive — why ACOT chose this evidence path (positive; never say "Prop-AI")
3. Evidence — latest recorded Dubai sales / rent / licence facts only
4. Next — one clear follow-up

Do not invent lifestyle claims (schools, parks, views). Attract with real numbers.
"""
from __future__ import annotations

import re
from typing import Any

# Never leak vendor names into user-facing copy.
_VENDOR_LEAK = re.compile(r"\bprop[\s\-]?ai\b", re.I)


def scrub_user_facing(text: str | None) -> str:
    """Strip vendor names from answers shown to the user."""
    return _VENDOR_LEAK.sub("our", (text or "")).strip()


def build_motive(intent: str = "browse", *, detail: str | None = None) -> str:
    """Why ACOT is answering this way — short paragraph, fact-led."""
    detail = (detail or "").strip()
    presets = {
        "recommend": (
            "**Acot Recommend:** I ranked the options on recorded Dubai sales activity, "
            "recent sale prices, and rental contracts where we have them — so the shortlist "
            "reflects what the market actually paid, not brochure starting prices alone."
        ),
        "split": (
            "**Acot Recommend:** Your ask mixes two product paths, so I built two "
            "transaction-backed shortlists from the latest Dubai sales data — one for each "
            "path — so you can compare with real numbers before going deeper."
        ),
        "sales": (
            "Below are the latest recorded Dubai Land Department-style sales that match "
            "your filters. Each line quotes a real transaction price and date when available — "
            "so you can see what buyers actually paid, not asking prices alone."
        ),
        "browse": (
            "I pulled matching projects from our Dubai catalog and attached recent sale "
            "prices where recorded, so you see market-backed figures next to each name — "
            "not marketing copy alone."
        ),
        "agents": (
            "These are licensed Dubai agents from our DLD-linked records for that area, "
            "ranked with card grade and licence details so you can contact people who are "
            "actually authorised to advise on the community."
        ),
        "family": (
            "**Acot Recommend:** For family buys I rank communities on ready villa share "
            "and 3+ bedroom sales in the last 12 months — measurable stock families usually "
            "need — rather than raw tower transaction volume."
        ),
        "afford": (
            "**Acot Recommend:** I checked recorded bedroom medians and recent sales against "
            "your budget, then listed only options that sit at or under that number when the "
            "sales data allows — so the shortlist is capped by what the market actually cleared."
        ),
        "clarify": (
            "To keep the next answer precise and fact-backed, I need one missing detail "
            "before I pull sales or project matches."
        ),
    }
    if detail:
        return detail
    return presets.get(intent, presets["browse"])


def build_family_motive(focus: dict | None = None) -> str:
    """Focus-aware family motive so kids/towers/villa asks do not share one generic line."""
    focus = focus or {}
    if focus.get("avoid_towers") or (focus.get("kids") and focus.get("want_villas")):
        return (
            "**Acot Recommend:** You asked for family communities suited to kids — low-rise "
            "villa stock, not the busiest apartment towers. I therefore rank on ready villa "
            "share and villa sales from the last 12 months of recorded Dubai sales, instead of "
            "raw tower transaction volume."
        )
    if focus.get("want_3bed") and focus.get("want_villas"):
        return (
            "**Acot Recommend:** You want areas with more 3-bedroom and villa inventory. "
            "I rank communities on recorded 3+ bedroom sales and villa sales in the last "
            "12 months — measurable family-size stock, not lifestyle marketing."
        )
    if focus.get("not_studios") or (focus.get("want_villas") and not focus.get("kids")):
        return (
            "**Acot Recommend:** You asked to avoid studio-heavy stock. I surface communities "
            "where recent recorded sales are led by villas and townhouses, with ready villa "
            "counts and typical 3+ bedroom sale prices when available."
        )
    if focus.get("kids"):
        return (
            "**Acot Recommend:** For a household with children, extra bedrooms and villas are "
            "the clearest signal in our sales data. I rank on ready villa share and 3+ bedroom "
            "activity — not tower transaction volume alone."
        )
    return build_motive("family")


def _bed_label(beds: int | None) -> str | None:
    if beds is None:
        return None
    if beds == 0:
        return "studio"
    return f"{int(beds)}-bedroom"


def build_understood(
    *,
    place: str | None = None,
    beds: int | None = None,
    budget_aed: int | None = None,
    ready_only: bool = False,
    goal: str | None = None,
    intent: str = "browse",
    under_budget: bool = True,
    question: str | None = None,
) -> str:
    """Full-sentence restatement of what we will answer (attractive, not a stub)."""
    bits: list[str] = []
    if ready_only:
        bits.append("ready")
    bed = _bed_label(beds)
    if bed:
        bits.append(bed)

    goal_key = str(goal or "").lower()
    if intent == "recommend":
        goal_bit = {
            "family": "a family-focused buy",
            "budget": "a budget-led buy",
            "afford": "an affordability check",
            "invest": "an investment shortlist",
            "open_buy": "where to buy with no area locked yet",
            "ready": "ready versus off-plan",
            "compare": "a shortlist compare",
        }.get(goal_key, "a recommendation")
        bits.append(goal_bit)
    elif intent == "sales":
        bits.append("recent recorded sales")
    elif intent == "agents":
        bits.append("licensed agents")
    elif intent == "clarify":
        bits.append("your request")
    else:
        bits.append("homes" if bed or ready_only else "projects")

    where = ""
    if place and str(place).strip().lower() not in {"", "dubai"}:
        where = f" in {str(place).strip()}"
    elif place and str(place).strip().lower() == "dubai":
        where = " in Dubai"

    budget_bit = ""
    if budget_aed is not None:
        budget_bit = (
            f" under AED {int(budget_aed):,}"
            if under_budget
            else f" around AED {int(budget_aed):,}"
        )

    core = " ".join(bits).strip() or "your request"
    lead = f"Understood: you are looking for {core}{where}{budget_bit}."

    # Second sentence when we can name the evidence angle without inventing lifestyle.
    if intent == "agents":
        lead += (
            " I will show licensed agents with card rank and licence facts for that area."
        )
    elif intent == "recommend" and goal_key == "family":
        lead += (
            " I will rank communities on ready villa and 3+ bedroom sales activity."
        )
    elif intent == "recommend" and goal_key in {"budget", "afford", "invest"}:
        lead += (
            " I will ground the picks in recorded sale prices and transaction activity."
        )
    elif intent == "browse" or intent == "sales":
        lead += (
            " I will answer with recorded Dubai project and sales facts for your filters."
        )
    elif question and len(str(question).split()) >= 6:
        lead += " Here is the clearest fact-backed answer from our latest Dubai data."
    return lead


def build_browse_narrative(
    *,
    place: str | None,
    total: int | None,
    shown: int,
    beds: int | None = None,
    ready_only: bool = False,
    more: bool = False,
) -> str:
    """Richer multi-sentence lead for project browse lists."""
    where = f" in {place}" if place else " across Dubai"
    bed_bit = ""
    if beds == 0:
        bed_bit = " that include studio units"
    elif beds is not None:
        bed_bit = f" that include {beds}-bedroom units"
    ready_bit = " ready" if ready_only else ""
    count = int(total) if isinstance(total, int) and total > 0 else shown
    more_bit = " more" if more else ""

    lines = [
        (
            f"I found {count}{ready_bit} projects{where}{bed_bit}. "
            f"Below are {shown}{more_bit} of them, with recent recorded sale prices "
            "when our sales data has a match — so you can scan market-backed figures, "
            "not brochure-only starting prices."
        ),
        (
            "Each line highlights community, recent sale level, bedroom range, and "
            "handover when that date is still ahead. Open any card for the full project view."
        ),
    ]
    return "\n\n".join(lines)


def build_agents_narrative(*, place: str | None, count: int) -> str:
    """Richer lead for licensed-agent answers (keeps the stable place line for routing tests)."""
    where = f" for {place}" if place else " in Dubai"
    place_line = (
        f"Here are licensed agents for {place}."
        if place
        else "Here are licensed agents in Dubai."
    )
    return (
        f"Understood: you asked for top licensed agents{where}.\n\n"
        f"**Acot Recommend:** I shortlisted {count} agents from our DLD-linked licence "
        "records for that community — using card rank, agency, BRN/licence numbers, and "
        "card expiry when present — so you contact authorised advisors, not an unverified list.\n\n"
        f"{place_line}"
    )


def slots_from_question(question: str | None) -> dict[str, Any]:
    """Parse common slots for the understood line (no DB)."""
    from app.answers.intent_route import extract_place, parse_bedrooms, parse_budget_aed

    text = question or ""
    beds = parse_bedrooms(text)
    budget = parse_budget_aed(text)
    under = bool(re.search(r"\b(?:under|below|up\s+to|max(?:imum)?)\b", text, re.I))
    above = bool(re.search(r"\b(?:above|over|starting)\b", text, re.I))
    return {
        "place": extract_place(text),
        "beds": beds[0] if beds else None,
        "budget_aed": budget,
        "ready_only": bool(
            re.search(r"\b(?:ready(?:\s+only)?|only\s+ready|completed)\b", text, re.I)
        ),
        "under_budget": under or (budget is not None and not above),
    }


def prepend_understood(body: str | None, understood: str | None) -> str:
    """Put Understood first unless the body already starts that way."""
    text = scrub_user_facing(body)
    lead = scrub_user_facing(understood)
    if not lead:
        return text
    if text.lower().startswith("understood:"):
        return text
    if not text:
        return lead
    # Avoid doubling when intro already mirrors the same filters closely.
    if lead[11:].strip(" .").lower() in text.lower():
        return f"{lead}\n\n{text}"
    return f"{lead}\n\n{text}"
