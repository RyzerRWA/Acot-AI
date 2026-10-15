"""Rental search and follow-ups that stay on the listings already retrieved."""

import re
from typing import Any, Dict, Optional

from app.retrieval.hybrid.question_text import comparison_pair
from app.retrieval.supabase.listing_retriever import ListingRetriever


def retrieve_listings(supabase, question: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Search rental listings and keep project records out of the answer."""

    try:
        found = ListingRetriever(supabase).search(question, filters)
    except Exception as exc:
        print("Listing retrieval error:", exc)
        found = {
            "listings": [],
            "listing_total": 0,
            "subject": "that request",
        }

    rows = found.get("listings") or []
    print(f"Listings retrieved: {len(rows)}")

    result = {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": rows,
        "listing_query": True,
        "listing_total": found.get("listing_total"),
    }
    if not rows:
        result["understood_subject"] = found.get("subject") or "that request"
        result["understood_kind"] = "listing"
    return result


def retrieve_listing_comparison(
    supabase,
    filters: Dict[str, Any],
    pair: tuple,
) -> Optional[Dict[str, Any]]:
    """Load a rental sample for each of the two named places."""

    retriever = ListingRetriever(supabase)
    constraints = retriever._constraints(filters or {})
    sides = []
    rows = []

    for name in pair:
        community = retriever._match_community(name) or name
        matched, _total = retriever._sql_search(community, constraints, 8)
        if not matched:
            continue
        place = matched[0].get("community") or community
        sides.append({
            "entity_type": "community",
            "requested_name": name,
            "resolved_name": place,
            "data": {
                "listings": matched,
                "projects": [],
                "communities": [],
                "sub_communities": [],
            },
        })
        rows.extend(matched)

    if len(sides) < 2:
        return None

    print(f"Listing comparison: {[side['resolved_name'] for side in sides]}")
    return {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": rows,
        "listing_query": True,
        "listing_total": len(rows),
        "comparison": True,
        "comparison_entities": sides,
    }


def shown_listing_followup(
    question: str,
    conversation_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Filter the rows already on screen, such as "which of these are 4-bedroom"."""

    candidates = (conversation_context or {}).get("active_candidates") or []
    listings = []
    projects = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("entity_type") or "").lower()
        if kind == "listing":
            listings.append(item)
        elif kind == "project":
            projects.append(item)
    rows = listings or projects
    if not rows:
        return None

    texts = _question_texts(question, conversation_context)
    blob = "\n".join(texts).lower()
    if not re.search(
        r"\b(?:these|those|them|their|which of|which ones)\b",
        blob,
    ):
        return None

    wanted = wanted_bedrooms(blob)
    matched = list(rows)
    if wanted is not None:
        matched = [row for row in matched if row_has_bedrooms(row, wanted)]

    if re.search(r"\b(?:lowest|cheapest)\b", blob):
        prices = [row for row in matched if listing_price(row) is not None]
        if prices:
            lowest = min(listing_price(row) for row in prices)
            matched = [row for row in prices if listing_price(row) == lowest]

    if wanted is None and matched == rows and not re.search(
        r"\b(?:lowest|cheapest)\b",
        blob,
    ):
        return None

    print(
        "Shown list follow-up:",
        len(matched),
        "of",
        len(rows),
        "bedrooms",
        wanted,
    )
    if listings:
        return {
            "communities": [],
            "projects": [],
            "sub_communities": [],
            "listings": matched,
            "listing_query": True,
            "listing_total": len(matched),
            "comparison": False,
            "filter_no_match": not matched,
            "filter_evidence_listings": listings if not matched else [],
        }
    return {
        "communities": [],
        "projects": matched,
        "sub_communities": [],
        "listings": [],
        "listing_query": True,
        "comparison": False,
        "filter_no_match": not matched,
        "filter_evidence_projects": projects if not matched else [],
    }


def wanted_bedrooms(blob: str):
    """Read a bedroom count from messy wording, including "4-bedrooms bed"."""

    text = blob or ""
    number = re.search(r"\b(\d+)\s*[- ]?\s*bed", text)
    if number:
        return int(number.group(1))
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    word = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]?\s*bed",
        text,
    )
    if word:
        return words[word.group(1)]
    short = re.search(r"\b(\d+)\s*(?:br|bhk)\b", text)
    if short:
        return int(short.group(1))
    if re.search(r"\bstudios?\b", text):
        return 0
    return None


def row_has_bedrooms(record, wanted: int) -> bool:
    exact = listing_bed_count(record)
    if exact is not None and record.get("bedroom_min") is None:
        return exact == wanted

    low = record.get("bedroom_min")
    high = record.get("bedroom_max")
    bedrooms = record.get("bedrooms")
    if isinstance(bedrooms, dict):
        if low is None:
            low = bedrooms.get("min")
        if high is None:
            high = bedrooms.get("max")
    if low is None and high is None:
        return exact == wanted if exact is not None else False
    if low is None:
        low = high
    if high is None:
        high = low
    try:
        return float(low) <= wanted <= float(high)
    except (TypeError, ValueError):
        return False


def previous_listing_comparison(
    question: str,
    conversation_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compare two listings already shown, such as "compare 1 and 2"."""

    candidates = (conversation_context or {}).get("active_candidates") or []
    listings = [
        item
        for item in candidates
        if isinstance(item, dict)
        and str(item.get("entity_type") or "").lower() == "listing"
        and (item.get("name") or item.get("title"))
    ]
    if len(listings) < 2:
        return None

    texts = _question_texts(question, conversation_context)
    if not texts:
        return None
    if not any(wants_shown_comparison(text) for text in texts):
        return None
    if any(names_outside_places(text) for text in texts):
        return None

    picked = pick_listed_rows(listings, texts)
    if len(picked) < 2:
        return None

    sides = []
    for record in picked[:2]:
        name = str(record.get("name") or record.get("title")).strip()
        sides.append({
            "entity_type": "listing",
            "requested_name": name,
            "resolved_name": name,
            "data": {"listings": [record]},
        })

    print(
        "Listing follow-up comparison:",
        [side["resolved_name"] for side in sides],
    )
    return {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": picked[:2],
        "listing_query": True,
        "listing_total": 2,
        "comparison": True,
        "comparison_entities": sides,
    }


def _question_texts(question, conversation_context):
    texts = []
    original = str(
        (conversation_context or {}).get("current_user_question") or ""
    ).strip()
    if original:
        texts.append(original)
    if question and question not in texts:
        texts.append(question)
    return texts


def listing_bed_count(record):
    value = record.get("beds")
    if value is None and isinstance(record.get("bedrooms"), dict):
        value = record["bedrooms"].get("min")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def listing_price(record):
    value = record.get("price_aed")
    if value is None:
        value = record.get("price")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def wants_shown_comparison(text):
    """True when the aim is to compare or pick the best of what is on screen."""

    return bool(re.search(
        r"\b(?:compare|comparison|versus|vs\.?|better|best|recommend)\b",
        text or "",
        flags=re.IGNORECASE,
    ))


def names_outside_places(text):
    """True when the question names two places, not rows from the list."""

    pair = comparison_pair(text or "")
    if not pair:
        return False
    if any(is_list_slot(side) for side in pair):
        return False
    return True


def is_list_slot(value):
    text = re.sub(
        r"\b(?:the|a|an|rentals?|listings?|apartments?|options?|"
        r"ones?|properties|units?|of|them)\b",
        " ",
        str(value or "").lower(),
    )
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return True
    slot = (
        r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
        r"ninth|tenth|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    )
    if re.fullmatch(slot, text):
        return True
    if re.fullmatch(
        rf"{slot}\s+(?:and|&|,|with|vs\.?|versus)\s+{slot}",
        text,
    ):
        return True
    return bool(re.fullmatch(
        r"(?:first|top|these|those|last|next)\s+"
        r"(?:two|three|four|five|six|seven|eight|nine|ten|\d+)",
        text,
    ))


def pick_listed_rows(listings, texts):
    ordinals = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    words_to_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def in_range(positions):
        return (
            len(positions) >= 2
            and positions[0] != positions[1]
            and all(1 <= position <= len(listings) for position in positions[:2])
        )

    for text in texts:
        number_pair = re.search(
            r"\b(\d+)\s*(?:st|nd|rd|th)?\s*"
            r"(?:and|&|,|with|vs\.?|versus)\s*"
            r"(\d+)\s*(?:st|nd|rd|th)?\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if number_pair:
            positions = [int(number_pair.group(1)), int(number_pair.group(2))]
            if in_range(positions):
                return [listings[position - 1] for position in positions]

        words = re.findall(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|"
            r"eighth|ninth|tenth)\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if len(words) >= 2:
            positions = [ordinals[word.lower()] for word in words[:2]]
            if in_range(positions):
                return [listings[position - 1] for position in positions]

        span = re.search(
            r"\b(?:first|top|these|those)\s+"
            r"(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine|ten))\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if span:
            count = (
                int(span.group(1))
                if span.group(1)
                else words_to_numbers[span.group(2).lower()]
            )
            if count >= 2 and len(listings) >= 2:
                return listings[:2]

        last_two = re.search(
            r"\blast\s+(?:two|2)\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if last_two and len(listings) >= 2:
            return listings[-2:]

    mentioned = []
    blob = "\n".join(texts).lower()
    for record in listings:
        name = str(record.get("name") or record.get("title") or "").strip()
        if len(name) >= 8 and name.lower() in blob:
            mentioned.append(record)
        if len(mentioned) == 2:
            return mentioned

    if any(
        re.search(
            r"\b(?:these|those|them|both|above|shown|previous|rentals?|"
            r"listings?|options?|first|top|ones?)\b",
            text or "",
            flags=re.IGNORECASE,
        )
        for text in texts
    ):
        return listings[:2]
    return []
