"""Rental search and follow-ups that stay on the listings already retrieved."""

import re
from typing import Any, Dict, Optional

from app.retrieval.hybrid.question_text import comparison_pair
from app.retrieval.postgres.listing_retriever import ListingRetriever


def retrieve_listings(db, question: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Search rental listings and keep project records out of the answer."""

    try:
        found = ListingRetriever(db).search(question, filters)
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


def retrieve_bedroom_comparison(db, counts, place: str) -> Dict[str, Any]:
    """Load a rental sample for each bedroom count in one place."""

    retriever = ListingRetriever(db)
    community = retriever._match_community(place or "") or (place or "").strip()
    sides = []
    rows = []

    for count in counts:
        label = f"{count} bedroom rentals"
        if community:
            label = f"{label} in {community.title()}"
        constraints = ListingRetriever._constraints({"bedrooms": int(count)})
        try:
            matched, total = retriever._sql_search(community, constraints, 50)
        except Exception as exc:
            print("Bedroom listing comparison error:", exc)
            matched, total = [], 0
        if matched and matched[0].get("community"):
            label = f"{count} bedroom rentals in {matched[0]['community']}"
        sides.append({
            "entity_type": "listing",
            "requested_name": label,
            "resolved_name": label,
            "data": {
                "listings": matched,
                "listing_total": total,
                "projects": [],
                "communities": [],
                "sub_communities": [],
            },
        })
        rows.extend(matched)

    populated = [side for side in sides if side["data"]["listings"]]
    print(
        "Bedroom listing comparison:",
        community or place or "anywhere",
        [(side["resolved_name"], len(side["data"]["listings"])) for side in sides],
    )
    result = {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": rows,
        "listing_query": True,
        "listing_total": len(rows),
        "comparison": len(populated) >= 2,
        "comparison_entities": populated if len(populated) >= 2 else [],
    }
    if not rows:
        result["understood_subject"] = community.title() if community else (place or "that request")
        result["understood_kind"] = "listing"
    return result


def _normalize_listing_title(value: str) -> str:
    text = re.sub(r"[^a-z0-9\s]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def lookup_listing_by_title(title: str) -> Optional[Dict[str, Any]]:
    """Resolve a pasted listing title to one listings row."""

    from app.database.postgres import fetch

    needle = _normalize_listing_title(title)
    if len(needle) < 8:
        return None

    tokens = [token for token in needle.split() if len(token) > 2][:4]
    if not tokens:
        return None

    where = " and ".join(["title ilike %s"] * len(tokens))
    params = tuple(f"%{token}%" for token in tokens)
    try:
        rows = fetch(
            "select listing_id, title, purpose, property_type, category, beds, baths, "
            "size_sqft, price_aed, rent_frequency, furnishing, completion_status, "
            "community, sub_community, location, agency_name, cover_photo, "
            "image_urls, url "
            f"from listings where {where} "
            "order by price_aed asc nulls last "
            "limit 40",
            params,
        )
    except Exception as exc:
        print("Listing title lookup error:", exc)
        return None

    needle_compact = needle.replace(" ", "")
    best = None
    best_score = 0.0
    for row in rows or []:
        cand = _normalize_listing_title(row.get("title"))
        if not cand:
            continue
        cand_compact = cand.replace(" ", "")
        if cand == needle or needle in cand or cand in needle:
            score = 1.0 + min(len(cand), len(needle)) / 1000.0
        elif needle_compact and (
            needle_compact in cand_compact or cand_compact in needle_compact
        ):
            score = 0.92
        else:
            left = set(needle.split())
            right = set(cand.split())
            if not left or not right:
                continue
            score = len(left & right) / max(len(left), len(right))
            if score < 0.72:
                continue
        if score > best_score:
            best_score = score
            best = row
    return best


def payment_plan_subject(question: str) -> str:
    """Extract the named property from a payment-plan ask."""

    text = (question or "").strip()
    if not text:
        return ""
    match = re.search(
        r"(?:create|show|give|tell(?:\s+me)?|what(?:'s| is)?|find)\s+"
        r"(?:a\s+|the\s+|me\s+(?:a\s+|the\s+)?)?"
        r"payment\s+plans?\s+for\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"payment\s+plans?\s+(?:for|of|about)\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return ""
    name = match.group(1).strip(" .?!,")
    name = re.sub(
        r"\b(?:please|pls|project|projects|listing|listings|this|the)\b",
        " ",
        name,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", name).strip(" .?!,")


def asks_payment_plan(question: str) -> bool:
    return bool(
        re.search(
            r"\bpayment\s+plans?\b|\binstallments?\b|\bdown\s+payment\b",
            question or "",
            flags=re.IGNORECASE,
        )
    )


def retrieve_listing_by_title_result(title: str) -> Optional[Dict[str, Any]]:
    """Return a single listing as a structured retrieval result."""

    listing = lookup_listing_by_title(title)
    if not listing:
        return None
    name = str(listing.get("title") or title).strip()
    print("Listing title lookup:", name)
    return {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": [listing],
        "listing_query": True,
        "listing_total": 1,
        "understood_subject": name,
        "understood_kind": "listing",
    }


def retrieve_listing_title_comparison(
    pair: tuple,
) -> Optional[Dict[str, Any]]:
    """Compare two rental listings named by title (Spots-style paste)."""

    if not pair or len(pair) < 2:
        return None

    sides = []
    rows = []
    for name in pair[:2]:
        listing = lookup_listing_by_title(name)
        if not listing:
            continue
        title = str(listing.get("title") or name).strip()
        sides.append({
            "entity_type": "listing",
            "requested_name": name,
            "resolved_name": title,
            "data": {
                "listings": [listing],
                "projects": [],
                "communities": [],
                "sub_communities": [],
            },
        })
        rows.append(listing)

    if len(sides) < 2:
        return None

    print(
        "Listing title comparison:",
        [side["resolved_name"] for side in sides],
    )
    return {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": rows,
        "listing_query": True,
        "listing_total": 2,
        "comparison": True,
        "comparison_entities": sides,
    }


def retrieve_listing_comparison(
    db,
    filters: Dict[str, Any],
    pair: tuple,
) -> Optional[Dict[str, Any]]:
    """Load a rental sample for each of the two named places."""

    retriever = ListingRetriever(db)
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


def _mentioned_rental_place(texts) -> str:
    """Pull a place from 'rentals in Dubai South' style wording."""

    for text in texts or []:
        match = re.search(
            r"\b(?:rentals?|listings?|apartments?|studios?|villas?|units?|"
            r"options?|ones?)\s+in\s+(.+)$",
            text or "",
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\bin\s+(.+?)(?:\s+and\s+tell|\s+which|\s+tell me|$)",
                text or "",
                flags=re.IGNORECASE,
            )
        if not match:
            continue
        place = re.split(
            r"\b(?:and tell|tell me|which is|which one)\b",
            match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        place = re.sub(
            r"\b(?:rentals?|listings?|apartments?|studios?|villas?|"
            r"units?|options?|ones?|best|better)\b",
            " ",
            place,
            flags=re.IGNORECASE,
        )
        place = re.sub(r"\s+", " ", place).strip(" ,.")
        if place and not re.fullmatch(r"\d+", place):
            return place
    return ""


def _requested_listing_positions(texts):
    """Return explicit 1-based slots such as (1, 7) when the user named them."""

    for text in texts or []:
        number_pair = re.search(
            r"\b(\d+)\s*(?:st|nd|rd|th)?\s*"
            r"(?:and|&|,|with|vs\.?|versus)\s*"
            r"(\d+)\s*(?:st|nd|rd|th)?\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if number_pair:
            left = int(number_pair.group(1))
            right = int(number_pair.group(2))
            if left != right:
                return [left, right]

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
        words = re.findall(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|"
            r"eighth|ninth|tenth)\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if len(words) >= 2:
            left = ordinals[words[0].lower()]
            right = ordinals[words[1].lower()]
            if left != right:
                return [left, right]
    return None


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

    texts = _question_texts(question, conversation_context)
    if not texts:
        return None
    if not any(wants_shown_comparison(text) for text in texts):
        return None
    if any(names_outside_places(text) for text in texts):
        return None

    positions = _requested_listing_positions(texts)
    place = _mentioned_rental_place(texts)

    # If memory was shrunk by an earlier "first two" compare, reload the
    # named place so "1 and 7th" can still resolve.
    if (
        positions
        and place
        and (
            len(listings) < max(positions)
            or len(listings) < 2
        )
    ):
        try:
            found = ListingRetriever().search(
                f"listings in {place}",
                {},
            )
            refreshed = found.get("listings") or []
            if len(refreshed) >= max(positions):
                listings = refreshed
                print(
                    "Listing comparison reload:",
                    place,
                    len(listings),
                )
        except Exception as exc:
            print("Listing comparison reload error:", exc)

    if len(listings) < 2:
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

    # When the user names specific slots (1 and 7th), never silently fall
    # back to the first two rows if those slots are missing from memory.
    asked_specific_slots = False

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
            if positions[0] != positions[1]:
                asked_specific_slots = True
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
            if positions[0] != positions[1]:
                asked_specific_slots = True
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

    if asked_specific_slots:
        return []

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
