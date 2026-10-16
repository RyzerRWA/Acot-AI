"""Grounded answers and frontend cards built from database rows."""

import re
from datetime import datetime, timezone

from app.retrieval.hybrid.question_text import collapse_stretched_letters


def _json_safe(value):
    """
    Convert backend/database values into JSON-safe Python values.

    This keeps the existing retrieval data untouched while making the
    final response safe for a REST API or frontend client.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    # Handles Decimal, datetime, UUID, and other database values.
    return str(value)


def _format_bedrooms(record):
    """
    Return the bedroom range exactly from the available project data.
    """

    minimum = record.get("bedroom_min")
    maximum = record.get("bedroom_max")

    if minimum is None and maximum is None:
        return None

    if minimum is None:
        return str(maximum)

    if maximum is None:
        return str(minimum)

    if minimum == maximum:
        return str(minimum)

    return f"{minimum}-{maximum}"


def _bedroom_bounds(project):
    minimum = project.get("bedroom_min")
    maximum = project.get("bedroom_max")
    bedrooms = project.get("bedrooms")

    if isinstance(bedrooms, dict):
        if minimum is None:
            minimum = bedrooms.get("min")
        if maximum is None:
            maximum = bedrooms.get("max")

    return minimum, maximum


def _format_bedroom_phrase(project):
    minimum, maximum = _bedroom_bounds(project)

    if minimum is None and maximum is None:
        return None

    if minimum == 0 and maximum not in (None, 0):
        noun = "bedroom" if maximum == 1 else "bedrooms"
        return f"Studios to {maximum} {noun}"

    if minimum == 0:
        return "Studios"

    if minimum is None:
        return f"{maximum} bedrooms"

    if maximum is None or minimum == maximum:
        noun = "bedroom" if minimum == 1 else "bedrooms"
        return f"{minimum} {noun}"

    return f"{minimum} to {maximum} bedrooms"


def _format_aed(value):
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.replace("AED", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            value = float(cleaned)
        except ValueError:
            return value

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number <= 0:
        return None

    if number.is_integer():
        return f"AED {int(number):,}"

    return f"AED {number:,.0f}"


def _format_handover_month(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    return parsed.strftime("%B %Y")


def _listing_photo_urls(record):
    photos = []
    cover = str(record.get("cover_photo") or "").strip()
    if cover:
        photos.append(cover)

    raw = record.get("image_urls") or ""
    if isinstance(raw, str):
        for part in raw.split("|"):
            url = part.strip()
            if url and url not in photos:
                photos.append(url)
            if len(photos) >= 4:
                break
    return photos


def _format_listing_card(record):
    """Turn a rental row into the same card shape the page uses for projects."""

    beds = record.get("beds")
    if beds == 0:
        bedroom_label = "Studio"
    elif beds is None:
        bedroom_label = None
    elif beds == 1:
        bedroom_label = "1 bedroom"
    else:
        bedroom_label = f"{beds} bedrooms"

    price = record.get("price_aed")
    frequency = str(record.get("rent_frequency") or "").strip()
    price_text = _format_aed(price)
    if price_text and frequency:
        price_label = f"{price_text} {frequency}"
    else:
        price_label = price_text

    size = _json_safe(record.get("size_sqft"))
    property_type = record.get("property_type")

    return {
        "id": record.get("listing_id"),
        "name": record.get("title"),
        "developer": record.get("agency_name"),
        "community": record.get("community"),
        "sub_community": record.get("sub_community"),
        "price": _json_safe(price),
        "price_label": price_label,
        "rent_frequency": frequency or None,
        "bedrooms": {
            "min": _json_safe(beds),
            "max": _json_safe(beds),
            "label": bedroom_label,
        },
        "size_sqft": {
            "min": size,
            "max": size,
        },
        "property_types": [property_type] if property_type else [],
        "status": record.get("furnishing") or record.get("completion_status"),
        "photos": _listing_photo_urls(record),
        "property_finder_url": record.get("url"),
    }


def _format_project_card(record):
    """
    Convert a project database row into a frontend-friendly project card.

    Only fields that exist in the current database are exposed.
    No rental yield, fair value, deal score, ROI, or other missing
    investment metrics are invented here.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "developer": record.get("developer_name"),
        "developer_id": _json_safe(record.get("developer_id")),
        "city": record.get("city"),
        "community": record.get("community"),
        "sub_community": record.get("sub_community"),
        "price": _json_safe(record.get("price")),
        "price_label": (
            f"AED {record.get('price')}"
            if record.get("price") is not None
            else None
        ),
        "bedrooms": {
            "min": _json_safe(record.get("bedroom_min")),
            "max": _json_safe(record.get("bedroom_max")),
            "label": _format_bedrooms(record),
        },
        "size_sqft": {
            "min": _json_safe(record.get("size_min")),
            "max": _json_safe(record.get("size_max")),
        },
        "property_types": _json_safe(
            record.get("property_types") or []
        ),
        "status": record.get("status"),
        "project_status": record.get("project_status"),
        "pool_type": record.get("pool_type"),
        "handover_time": _json_safe(
            record.get("handover_time")
        ),
        "amenities": _json_safe(
            record.get("amenities") or []
        ),
        "photos": _json_safe(
            record.get("photos") or []
        ),
        "description": record.get("description"),
        "brochure_url": record.get("brochure_url"),
        "property_finder_url": record.get(
            "property_finder_url"
        ),
        "source": record.get("source"),
    }


def _format_community_card(record):
    """
    Convert a community database row into a frontend-friendly card.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "slug": record.get("slug"),
        "city": record.get("city"),
        "location": {
            "latitude": _json_safe(record.get("latitude")),
            "longitude": _json_safe(record.get("longitude")),
        },
        "inventory": {
            "sell_properties_count": _json_safe(
                record.get("sell_properties_count")
            ),
            "rent_properties_count": _json_safe(
                record.get("rent_properties_count")
            ),
            "projects_count": _json_safe(
                record.get("projects_count")
            ),
            "pool_projects_count": _json_safe(
                record.get("pool_projects_count")
            ),
            "total_count": _json_safe(
                record.get("total_count")
            ),
        },
        "assigned_agents": _json_safe(
            record.get("assigned_agents")
        ),
        "knowledge_text": record.get("knowledge_text"),
        "source": record.get("source"),
    }


def _format_sub_community_card(record):
    """
    Convert a sub-community database row into a frontend-friendly card.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "slug": record.get("slug"),
        "city": record.get("city"),
        "community": record.get("community"),
        "community_slug": record.get("community_slug"),
        "location": {
            "latitude": _json_safe(record.get("latitude")),
            "longitude": _json_safe(record.get("longitude")),
        },
        "photos": _json_safe(
            record.get("photos") or []
        ),
        "knowledge_text": record.get("knowledge_text"),
        "source": record.get("source"),
    }


def _format_document_card(record):
    """
    Convert a pgvector document result into a frontend-friendly source.
    """

    return {
        "id": _json_safe(record.get("id")),
        "document_id": _json_safe(
            record.get("document_id")
        ),
        "title": record.get("document_title"),
        "type": record.get("document_type"),
        "publisher": record.get("publisher"),
        "chunk_id": record.get("chunk_id"),
        "similarity": _json_safe(
            record.get("similarity")
        ),
        "url": record.get("document_url"),
        "content": record.get("content"),
    }


def _median(values):
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2


def _as_number(value):
    if isinstance(value, dict) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_number(record, *keys):
    for key in keys:
        number = _as_number(record.get(key))
        if number is not None:
            return number
        nested = record.get(key)
        if isinstance(nested, dict):
            number = _as_number(nested.get("min"))
            if number is None:
                number = _as_number(nested.get("max"))
            if number is not None:
                return number
    return None


def _handover_stamp(value):
    if isinstance(value, datetime):
        parsed = value
    elif value is None or isinstance(value, dict):
        return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _handover_label(stamp):
    if stamp is None:
        return None
    return datetime.fromtimestamp(stamp, timezone.utc).strftime("%B %Y")


def _record_photo(record):
    cover = str(record.get("cover_photo") or "").strip()
    if cover.startswith("http"):
        return cover
    photos = record.get("photos") or []
    if isinstance(photos, str):
        photos = [photos]
    for photo in photos:
        if isinstance(photo, str) and photo.startswith("http"):
            return photo
    raw = record.get("image_urls") or ""
    if isinstance(raw, str) and raw.strip():
        return raw.split("|")[0].strip()
    return None


def _side_snapshot(name, records):
    prices = []
    sizes = []
    beds = []
    handovers = []
    frequencies = []
    photo = None
    link = None
    for record in records:
        price = _record_number(record, "price_aed", "price")
        size = _record_number(record, "size_sqft")
        if size is None:
            size_min = _record_number(record, "size_min")
            size_max = _record_number(record, "size_max")
            if size_min is not None and size_max is not None:
                size = (size_min + size_max) / 2
            else:
                size = size_min if size_min is not None else size_max
        bed = _record_number(record, "beds")
        if bed is None:
            bed_min = _record_number(record, "bedroom_min")
            bed_max = _record_number(record, "bedroom_max")
            if bed_min is not None and bed_max is not None:
                bed = (bed_min + bed_max) / 2
            else:
                bed = bed_min if bed_min is not None else bed_max
        handover = _handover_stamp(record.get("handover_time"))
        if price is not None:
            prices.append(price)
            frequency = str(record.get("rent_frequency") or "").strip().lower()
            if frequency:
                frequencies.append(frequency)
        if size is not None:
            sizes.append(size)
        if bed is not None:
            beds.append(bed)
        if handover is not None:
            handovers.append(handover)
        if photo is None:
            photo = _record_photo(record)
        if link is None:
            link = record.get("url") or record.get("property_finder_url")

    shared_frequency = frequencies[0] if frequencies and len(set(frequencies)) == 1 else None
    return {
        "name": name,
        "count": len(records),
        "price": _median(prices),
        "size": _median(sizes),
        "beds": _median(beds),
        "handover": _median(handovers),
        "frequency": shared_frequency,
        "photo": photo,
        "link": link,
    }


def _score_snapshots(left, right):
    """Score two sides from shared fields. Higher means stronger on those fields."""

    checks = []

    def prefer_high(key, label, formatter):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None or left_value == right_value:
            return
        winner = left if left_value > right_value else right
        other = right if winner is left else left
        checks.append((
            winner["name"],
            (
                f"**{label.capitalize()}:** {winner['name']} is ahead "
                f"({formatter(winner[key])} versus {formatter(other[key])})."
            ),
        ))

    def prefer_low(key, label, formatter):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None or left_value == right_value:
            return
        winner = left if left_value < right_value else right
        other = right if winner is left else left
        checks.append((
            winner["name"],
            (
                f"**{label.capitalize()}:** {winner['name']} is ahead "
                f"({formatter(winner[key])} versus {formatter(other[key])})."
            ),
        ))

    prefer_low("price", "price", lambda value: _format_aed(value) or str(value))
    prefer_high("size", "size", lambda value: f"{int(value):,} sqft")
    def _bedroom_text(value):
        if value == 0:
            return "Studio"
        if value == 1:
            return "1 bedroom"
        return f"{value:g} bedrooms"

    prefer_high("beds", "bedrooms", _bedroom_text)
    left_handover = left.get("handover")
    right_handover = right.get("handover")
    if (
        left_handover is not None
        and right_handover is not None
        and left_handover != right_handover
    ):
        winner = left if left_handover < right_handover else right
        other = right if winner is left else left
        checks.append((
            winner["name"],
            (
                f"**Handover:** {winner['name']} is earlier "
                f"({_handover_label(winner['handover'])} versus "
                f"{_handover_label(other['handover'])})."
            ),
        ))

    if not checks:
        left["score"] = 50
        right["score"] = 50
        return (
            f"{left['name']} and {right['name']} both score 50%. "
            "The records do not show a difference on price, size, bedrooms, or handover."
        )

    left_wins = sum(1 for name, _text in checks if name == left["name"])
    right_wins = len(checks) - left_wins
    if left_wins == right_wins:
        left_score = right_score = 50
    else:
        left_score = round(40 + 60 * left_wins / len(checks))
        right_score = round(40 + 60 * right_wins / len(checks))
    left["score"] = left_score
    right["score"] = right_score
    winner = left if left_score >= right_score else right
    loser = right if winner is left else left
    bullets = "\n".join(f"- {text}" for _name, text in checks)
    if left_score == right_score:
        intro = f"{left['name']} and {right['name']} both score 50%."
    else:
        intro = (
            f"{winner['name']} scores {winner['score']}% and "
            f"{loser['name']} scores {loser['score']}%. "
            f"{winner['name']} is the stronger of the two on these records."
        )
    return f"{intro}\n{bullets}"


def _comparison_card(side):
    price_label = _format_aed(side.get("price"))
    if price_label and side.get("frequency"):
        price_label = f"{price_label} {side['frequency']}"
    beds = side.get("beds")
    if beds is None:
        bedroom_label = None
    elif beds == 0:
        bedroom_label = "Studio"
    elif beds == 1:
        bedroom_label = "1 bedroom"
    else:
        bedroom_label = f"{int(beds) if float(beds).is_integer() else beds} bedrooms"
    size = side.get("size")
    size_value = int(size) if isinstance(size, float) and size.is_integer() else size
    return {
        "name": side["name"],
        "score": side.get("score"),
        "photos": [side["photo"]] if side.get("photo") else [],
        "price": side.get("price"),
        "price_label": price_label,
        "community": side["name"],
        "bedrooms": {
            "min": beds,
            "max": beds,
            "label": bedroom_label,
        },
        "size_sqft": {
            "min": size_value,
            "max": size_value,
        },
        "property_finder_url": side.get("link"),
        "handover_time": _handover_label(side.get("handover")),
        "status": f"{side['score']}%" if side.get("score") is not None else None,
    }


def _build_comparison(structured_summary):
    entities = [
        entity
        for entity in (structured_summary.get("comparison_entities") or [])
        if isinstance(entity, dict)
    ]
    if len(entities) < 2:
        return None

    snapshots = []
    for entity in entities[:2]:
        data = entity.get("data") or {}
        records = data.get("listings") or data.get("projects") or []
        records = [record for record in records if isinstance(record, dict)]
        if not records:
            return None
        name = (
            entity.get("resolved_name")
            or entity.get("requested_name")
            or "Option"
        )
        snapshots.append(_side_snapshot(name, records))

    answer = _score_snapshots(snapshots[0], snapshots[1])
    return {
        "answer": answer,
        "sides": [_comparison_card(side) for side in snapshots],
    }


def _detect_response_type(
    question,
    question_type,
    structured_summary
):
    """
    Determine the frontend response component from the actual result.

    This is presentation metadata only. It does not affect retrieval.
    """

    question_lower = (question or "").lower()

    projects = structured_summary.get(
        "projects",
        []
    )
    communities = structured_summary.get(
        "community",
        []
    )
    sub_communities = structured_summary.get(
        "sub_communities",
        []
    )
    listings = structured_summary.get("listings") or []

    if structured_summary.get("comparison_view"):
        return "comparison"

    if question_type == "information" and len(listings) + len(projects or []) == 1:
        return "fact"

    if len(listings) > 1 and re.search(
        r"\b(?:which of these|which ones|which of those)\b",
        question_lower,
    ):
        return "listing_list"

    if listings and question_type == "search":
        return "listing_list"

    if question_type == "investment":
        return "investment_analysis"

    if any(
        phrase in question_lower
        for phrase in (
            "compare",
            "comparison",
            "difference between",
            "vs ",
            " versus "
        )
    ) and len(projects) >= 2:
        return "project_comparison"

    if question_type == "search" and projects:
        return "project_list"

    if sub_communities:
        return "sub_community_list"

    if communities:
        return "community_list"

    return "general"


def build_frontend_response(
    question,
    standalone_question,
    answer,
    question_type,
    structured_summary,
    documents,
    investment_analysis=None,
):
    """
    Build the single response object that a future frontend/API can consume.

    The AI answer remains available as `answer`.
    Structured database records are exposed as frontend-friendly `data`.
    """

    projects = structured_summary.get(
        "projects",
        []
    ) or []

    communities = structured_summary.get(
        "community",
        []
    ) or []

    sub_communities = structured_summary.get(
        "sub_communities",
        []
    ) or []

    formatted_projects = [
        _format_project_card(project)
        for project in projects
        if isinstance(project, dict)
    ]

    formatted_communities = [
        _format_community_card(community)
        for community in communities
        if isinstance(community, dict)
    ]

    formatted_sub_communities = [
        _format_sub_community_card(sub_community)
        for sub_community in sub_communities
        if isinstance(sub_community, dict)
    ]

    formatted_documents = [
        _format_document_card(document)
        for document in documents
        if isinstance(document, dict)
    ]

    formatted_listings = [
        _format_listing_card(listing)
        for listing in (structured_summary.get("listings") or [])
        if isinstance(listing, dict)
    ]

    response_type = _detect_response_type(
        question,
        question_type,
        structured_summary,
    )

    return {
        "status": "success",
        "response_type": response_type,
        "question": question,
        "standalone_question": standalone_question,
        "question_type": question_type,
        "answer": answer,
        "data": {
            "communities": formatted_communities,
            "projects": formatted_projects,
            "listings": formatted_listings,
            "sub_communities": formatted_sub_communities,
            "documents": formatted_documents,
            "investment_analysis": _json_safe(
                investment_analysis
            ),
            "comparison": structured_summary.get("comparison_view") or [],
        },
        "meta": {
            "counts": {
                "communities": len(formatted_communities),
                "projects": len(formatted_projects),
                "listings": len(formatted_listings),
                "sub_communities": len(
                    formatted_sub_communities
                ),
                "documents": len(formatted_documents),
            }
        },
    }


# =========================================================
# FAST STRUCTURED ANSWER
# =========================================================

def _is_fast_structured_question(
    question,
    question_type,
    structured_summary,
    documents,
):
    """
    Use the database template only for a browse/search that already has rows.

    Price, bedroom, handover, comparison, and other follow-ups stay on the
    model path so the reply is written instead of reprinted as a hit list.
    """

    del question

    if question_type != "search":
        return False

    if documents:
        return False

    if structured_summary.get("filter_no_match"):
        return True

    return bool(
        structured_summary.get("projects")
        or structured_summary.get("community")
        or structured_summary.get("sub_communities")
        or structured_summary.get("listings")
    )


def _listing_bedroom_phrase(listing):
    beds = listing.get("beds")
    if beds is None:
        return None
    if beds == 0:
        return "Studio"
    if beds == 1:
        return "1 bedroom"
    return f"{beds} bedrooms"


def _listing_rent_phrase(listing):
    price_text = _format_aed(listing.get("price_aed"))
    if not price_text:
        return None
    frequency = str(listing.get("rent_frequency") or "").strip()
    if frequency:
        return f"Rent {price_text} {frequency}"
    return f"Rent {price_text}"


def _listing_price_clause(listings):
    prices = []
    frequencies = set()
    for listing in listings:
        price_text = _format_aed(listing.get("price_aed"))
        if not price_text:
            continue
        try:
            prices.append(float(listing.get("price_aed")))
        except (TypeError, ValueError):
            continue
        frequency = str(listing.get("rent_frequency") or "").strip().lower()
        if frequency:
            frequencies.add(frequency)

    if not prices:
        return ""

    if len(frequencies) == 1:
        label = f"{next(iter(frequencies))} rents"
    else:
        label = "rents"

    low = _format_aed(min(prices))
    high = _format_aed(max(prices))
    if low == high:
        return f", with a {label[:-1] if label.endswith('s') else label} of {low}"
    return f", with {label} from {low} to {high}"


def _listing_answer(listings, structured_summary):
    """List rentals in the same shape as a project browse answer."""

    total = structured_summary.get("listing_total")
    count = len(listings)
    noun = "rental listing" if count == 1 else "rental listings"
    communities = _unique_field(listings, "community")
    named_place = str(structured_summary.get("catalog_place") or "").strip()
    if len(communities) == 1:
        place = f" in {communities[0]}"
    elif named_place:
        place = f" in {named_place}"
    else:
        place = ""
    price_clause = _listing_price_clause(listings)
    showing_sample = isinstance(total, int) and total > count

    if count == 1 and not showing_sample:
        title = listings[0].get("title") or "This listing"
        intro = (
            f"{title} is a rental listing{place}. "
            "Here is what is on record:"
        )
    elif showing_sample:
        intro = (
            f"I found {total} {noun}{place}. "
            f"Here are {count} of them{price_clause}:"
        )
    else:
        intro = (
            f"I found {count} {noun}{place}{price_clause}. "
            "Here is a short look at each one:"
        )

    lines = [intro, ""]
    repeat_community = len(communities) != 1
    for listing in listings:
        title = listing.get("title") or "Untitled listing"
        agency = str(listing.get("agency_name") or "").strip()
        sentence = f"- **{title}**"
        sentence += f" by {agency}." if agency else "."

        details = []
        if repeat_community and listing.get("community"):
            details.append(f"Community {listing.get('community')}.")
        rent_text = _listing_rent_phrase(listing)
        if rent_text:
            details.append(f"{rent_text}.")
        bedroom_text = _listing_bedroom_phrase(listing)
        if bedroom_text:
            details.append(f"{bedroom_text}.")
        size = listing.get("size_sqft")
        if size:
            details.append(f"{size} sqft.")
        if details:
            sentence = f"{sentence} {' '.join(details)}"
        lines.append(sentence)

    return "\n".join(lines)


def _build_fast_structured_answer(
    question,
    structured_summary,
):
    """
    Build a concise, grounded answer directly from PostgreSQL records.
    No model call is made for this path.
    """

    q = (question or "").lower()
    listings = structured_summary.get("listings", []) or []
    if listings:
        return _listing_answer(listings, structured_summary)

    evidence_listings = structured_summary.get("filter_evidence_listings") or []
    if structured_summary.get("filter_no_match") and evidence_listings:
        bedroom_match = re.search(r"\b(\d+)\s*[- ]?\s*bed", q)
        if bedroom_match:
            requested = bedroom_match.group(1)
            lines = [f"None of these listings have {requested} bedrooms."]
        else:
            lines = ["None of these listings matched that request."]
        for listing in evidence_listings:
            title = listing.get("title") or listing.get("name") or "Untitled listing"
            beds = listing.get("beds")
            if beds == 0:
                bedroom_text = "Studio"
            elif beds is None:
                bedroom_text = "bedroom count not recorded"
            elif beds == 1:
                bedroom_text = "1 bedroom"
            else:
                bedroom_text = f"{beds} bedrooms"
            lines.append(f"- **{title}**. {bedroom_text}.")
        return "\n".join(lines)

    projects = structured_summary.get("projects", []) or []
    communities = structured_summary.get("community", []) or []
    sub_communities = structured_summary.get("sub_communities", []) or []

    # ---------------------------------------------------------
    # CONVERSATIONAL FILTER: ZERO MATCHES
    # ---------------------------------------------------------
    # Example:
    #   Q1: Show me projects in Dubai Marina
    #   Q2: Which of these have 2 bedroom options?
    #
    # `projects` is intentionally empty because no candidate matched the
    # filter. `filter_evidence_projects` contains the previous candidates
    # so we can explain the result using the real Supabase data.
    if structured_summary.get("filter_no_match"):
        evidence_projects = (
            structured_summary.get(
                "filter_evidence_projects",
                []
            )
            or []
        )

        bedroom_match = re.search(
            r"\b(\d+)\s*(?:-?\s*)bed(?:room)?s?\b",
            q,
        )

        if bedroom_match and evidence_projects:
            requested_bedrooms = int(
                bedroom_match.group(1)
            )

            lines = [
                f"No retrieved projects have "
                f"{requested_bedrooms}-bedroom options."
            ]

            for project in evidence_projects:
                name = project.get("name") or "Unknown project"
                bedroom_label = _format_bedrooms(project)

                if bedroom_label is not None:
                    lines.append(
                        f"{name} offers {bedroom_label} bedrooms."
                    )
                else:
                    lines.append(
                        f"{name} does not have bedroom information "
                        f"available in the current data."
                    )

            return "\n".join(lines)

        # Generic fallback for other conversational filters.
        if evidence_projects:
            names = [
                project.get("name") or "Unknown project"
                for project in evidence_projects
            ]
            return (
                "None of the previously retrieved projects matched "
                "the requested filter. Previous candidates: "
                + ", ".join(names)
                + "."
            )

    if projects:
        lines = []

        wants_bedrooms = any(
            term in q
            for term in ("bedroom", "bedrooms")
        )
        wants_handover = "handover" in q
        wants_developer = any(
            term in q
            for term in ("developer", "developers")
        )
        wants_status = "status" in q
        wants_price = any(
            term in q
            for term in ("price", "prices", "cost")
        )

        # Field-specific follow-up answers.
        if len(projects) == 1:
            project = projects[0]
            name = project.get("name") or "Unknown project"

            if wants_bedrooms:
                bedroom_label = _format_bedrooms(project)
                value = (
                    bedroom_label
                    if bedroom_label is not None
                    else "Not available"
                )
                return f"{name} offers {value} bedrooms."

            if wants_handover:
                value = project.get("handover_time")
                if value is None:
                    return f"Handover date for {name} is not available in the current data."
                return f"{name} handover date: {value}"

            if wants_developer:
                value = project.get("developer_name")
                value = (
                    str(value).strip()
                    if value is not None and str(value).strip()
                    else "Not available"
                )
                return f"{name} developer: {value}"

            if wants_status:
                status = project.get("status")
                project_status = project.get("project_status")

                if status and project_status:
                    return (
                        f"{name} status: {status}; "
                        f"project status: {project_status}"
                    )

                value = status or project_status or "Not available"
                return f"{name} status: {value}"

            if wants_price:
                value = project.get("price")
                value = (
                    f"AED {value}"
                    if value is not None
                    else "Not available"
                )
                return f"{name} starting price: {value}"

        # Multi-project field-specific answers.
        if wants_bedrooms:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                bedroom_label = _format_bedrooms(project)
                bedroom_text = (
                    bedroom_label
                    if bedroom_label is not None
                    else "Not available"
                )
                lines.append(
                    f"{index}. {name} | Bedrooms: {bedroom_text}"
                )
            return "\n".join(lines)

        if wants_handover:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                value = project.get("handover_time")
                value = value if value is not None else "Not available"
                lines.append(
                    f"{index}. {name} | Handover: {value}"
                )
            return "\n".join(lines)

        if wants_developer:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                developer = project.get("developer_name")
                developer_text = (
                    str(developer).strip()
                    if developer is not None and str(developer).strip()
                    else "Not available"
                )
                lines.append(
                    f"{index}. {name} | Developer: {developer_text}"
                )
            return "\n".join(lines)

        if wants_price:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                price = project.get("price")
                price_text = (
                    f"AED {price}" if price is not None else "Not available"
                )
                lines.append(
                    f"{index}. {name} | Starting price: {price_text}"
                )
            return "\n".join(lines)

        # Default browse answer.
        intro = _browse_intro(
            projects,
            ambiguous_label=structured_summary.get("ambiguous_label"),
        )
        project_total = structured_summary.get("project_total")
        if isinstance(project_total, int) and project_total > len(projects):
            where = ""
            named_place = str(structured_summary.get("catalog_place") or "").strip()
            if named_place:
                where = f" in {named_place}"
            intro = (
                f"I found {project_total} projects{where}. "
                f"Here are {len(projects)} of them:"
            )
        lines = [intro, ""]

        for project in projects:
            name = project.get("name") or "Unknown project"
            developer = project.get("developer_name") or project.get("developer")
            developer_text = (
                str(developer).strip()
                if developer is not None and str(developer).strip()
                else ""
            )
            sentence = f"- **{name}**"
            sentence += f" by {developer_text}." if developer_text else "."

            details = []
            price_text = _format_aed(project.get("price"))
            if price_text:
                details.append(f"Starting price {price_text}.")

            bedroom_text = _format_bedroom_phrase(project)
            if bedroom_text:
                details.append(f"{bedroom_text}.")

            handover_text = _format_handover_month(project.get("handover_time"))
            if handover_text:
                details.append(f"Handover {handover_text}.")

            if details:
                sentence = f"{sentence} {' '.join(details)}"

            lines.append(sentence)

        if communities:
            community_total = structured_summary.get("community_total") or len(communities)
            named_place = str(structured_summary.get("catalog_place") or "").strip()
            where = f" in {named_place}" if named_place else ""
            lines.append("")
            lines.append(
                f"There are {community_total} communities{where}. "
                "Here are some of them:"
            )
            for record in communities:
                lines.append(f"- {record.get('name') or 'Unknown community'}")

        return "\n".join(lines)

    if sub_communities:
        lines = [
            f"Found {len(sub_communities)} sub-community(s) matching your request:"
        ]
        for index, record in enumerate(sub_communities, start=1):
            lines.append(
                f"{index}. {record.get('name') or 'Unknown sub-community'}"
            )
        return "\n".join(lines)

    if communities:
        community_total = structured_summary.get("community_total") or len(communities)
        named_place = str(structured_summary.get("catalog_place") or "").strip()
        where = f" in {named_place}" if named_place else ""
        if community_total > len(communities):
            opening = (
                f"I found {community_total} communities{where}. "
                f"Here are {len(communities)} of them:"
            )
        else:
            opening = f"I found {len(communities)} communities{where}:"
        lines = [opening]
        for index, record in enumerate(communities, start=1):
            name = record.get("name") or "Unknown community"
            city = record.get("city")
            if city:
                lines.append(f"{index}. {name} | City: {city}")
            else:
                lines.append(f"{index}. {name}")
        return "\n".join(lines)

    return _missing_data_answer(structured_summary)


# =========================================================
# SMALL TALK
# =========================================================

_SMALL_TALK_KINDS = {
    "hi": "greeting",
    "hii": "greeting",
    "hello": "greeting",
    "hey": "greeting",
    "good morning": "greeting",
    "good evening": "greeting",
    "good afternoon": "greeting",
    "thanks": "thanks",
    "thank you": "thanks",
    "bye": "bye",
    "goodbye": "bye",
    "who are you": "identity",
    "what are you": "identity",
    "what can you do": "capabilities",
    "what do you do": "capabilities",
    "how can you help": "capabilities",
    "help": "capabilities",
}

_STARTER_CHIPS = (
    "Show me projects in Jumeirah Village Circle.",
    "What can you do?",
    "Who are you?",
)


def _normalize_chat_text(question):
    text = collapse_stretched_letters(question or "").strip().lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[!?.]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _unique_field(projects, key):
    values = []
    for project in projects:
        raw = project.get(key)
        if raw is None and key == "developer_name":
            raw = project.get("developer")
        text = str(raw).strip() if raw is not None else ""
        if text and text not in values:
            values.append(text)
    return values


def _price_range_clause(projects):
    prices = []
    for project in projects:
        value = project.get("price")
        if value is None:
            continue
        try:
            if isinstance(value, str):
                value = value.replace("AED", "").replace(",", "").strip()
            number = float(value)
        except (TypeError, ValueError):
            continue
        prices.append(number)

    if not prices:
        return ""

    low = min(prices)
    high = max(prices)
    if low == high:
        return f", with a starting price of {_format_aed(low)}"
    return (
        f", with starting prices from {_format_aed(low)} "
        f"to {_format_aed(high)}"
    )


def _browse_intro(projects, ambiguous_label=None):
    """Open a project list in one sentence."""

    count = len(projects)
    noun = "project" if count == 1 else "projects"
    all_active = all(
        str(project.get("status") or "").strip().upper() == "ACTIVE"
        for project in projects
    )
    active = "active " if all_active else ""

    if ambiguous_label and count > 1:
        return f"{ambiguous_label} matches {count} {noun}. Here they are:"

    communities = _unique_field(projects, "community")
    developers = _unique_field(projects, "developer_name")
    price_clause = _price_range_clause(projects)

    if count == 1:
        name = projects[0].get("name") or "This project"
        place = f" in {communities[0]}" if communities else ""
        kind = f"{active}project".strip()
        article = "an" if kind.startswith("active") else "a"
        return (
            f"{name} is {article} {kind}{place}. "
            "Here is what is on record:"
        )

    if len(communities) == 1:
        return (
            f"I found {count} {active}{noun} in {communities[0]}"
            f"{price_clause}. Here is a short look at each one:"
        )

    if len(developers) == 1:
        return (
            f"I found {count} {active}{developers[0]} {noun}"
            f"{price_clause}. Here is a short look at each one:"
        )

    return (
        f"I found {count} {active}{noun}{price_clause}. "
        "Here is a short look at each one:"
    )


def _missing_data_answer(structured_summary):
    subject = str(
        structured_summary.get("understood_subject") or ""
    ).strip()
    closest = str(structured_summary.get("closest_match") or "").strip()
    kind = structured_summary.get("understood_kind") or "place"

    if not subject:
        return "I don't have a match for that in the current data."

    if kind == "listing":
        return (
            f"I don't have rental listings matching {subject} "
            "in the current data."
        )

    if kind == "project":
        return (
            f"I don't have a project called {subject} "
            "in the current data."
        )

    if kind == "developer":
        return (
            f"I don't have {subject} projects in the current data."
        )

    if kind == "developer_community":
        return f"I don't have {subject} in the current data."

    if kind == "pattern":
        return (
            f"I don't have projects matching {subject} "
            "in the current data."
        )

    if closest:
        return (
            f"I don't have projects in {subject}. "
            f"The closest community I do have is {closest}."
        )

    return (
        f"I don't have projects in {subject} in the current data."
    )
