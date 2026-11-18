"""
Decide whether a question is about rental listings (our scraped data) or about buying / off-plan projects
(the existing pipeline). Keyword rules only: free, instant and predictable.
"""
from __future__ import annotations

import re

RENT = re.compile(
    r"\b(rent|rents|rental|rentals|renting|rented|lease|leasing|tenant|tenants|ejari|landlord|"
    r"per month|a month|monthly|/month|/mo|per year|a year|yearly|annual rent|/yr|"
    r"listing|listings|furnished|unfurnished)\b", re.I)
SALE = re.compile(
    r"\b(buy|buying|purchase|off[- ]?plan|offplan|developer|developers|handover|payment plan|launch|launching|"
    r"roi|invest|investment|investing|yield|brochure|project|projects|completion date|for sale|resale|"
    r"mortgage|down ?payment|starting price)\b", re.I)
FOLLOW_UP = re.compile(r"^(and|also|what about|how about|ok|okay|then|what if|any|show|only|cheaper|bigger|smaller)\b", re.I)
_BROWSE_VERB = re.compile(r"\b(?:show|list|find|give me|looking for)\b", re.I)
_RENT_BROWSE = re.compile(r"\b(?:for rent|to rent|rentals?|listings?)\b", re.I)
_AGENT_RENT = re.compile(r"\b(?:compar\w*|vs\.?|versus|yield|roi|fair|this listing|this one)\b", re.I)


def is_rent_question(question: str, last_turn_type: str | None = None) -> bool:
    """True when the question belongs to the rental-listings agent."""
    text = (question or "").strip()
    if not text:
        return False
    rent, sale = bool(RENT.search(text)), bool(SALE.search(text))
    if rent and not sale:
        return True
    if rent and sale:
        return False   # "buy or rent?", "rental yield for investors": the existing pipeline handles both sides
    # no clear signal: a short follow-up inside a rental conversation stays with the rental agent
    return last_turn_type == "listings" and (len(text.split()) <= 12 or bool(FOLLOW_UP.search(text))) and not sale


def is_rent_browse(question: str) -> bool:
    """True for 'show me 1 bedroom apartments for rent in X' — structured list, no LLM agent."""
    text = (question or "").strip()
    if not is_rent_question(text) or _AGENT_RENT.search(text):
        return False
    if not _BROWSE_VERB.search(text):
        return False
    return bool(_RENT_BROWSE.search(text) or re.search(r"\brent\b", text, re.I))
