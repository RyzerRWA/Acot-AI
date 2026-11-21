"""Grounded answers and frontend cards built from database rows."""

import re
from datetime import datetime, timezone

from app.answers.grounding import ground_project_row, ground_project_rows
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


def _recent_sale_row(record) -> bool:
    """True only when sale kind is set and a grounded price (or sale date) exists."""
    record = record or {}
    kind = str(record.get("price_kind") or "").lower()
    if kind not in {"recent_sale", "latest_sale"}:
        return False
    try:
        has_price = record.get("price") is not None and float(record["price"]) > 0
    except (TypeError, ValueError):
        has_price = False
    has_sale_date = bool(record.get("last_sold_on") or record.get("last_sale"))
    return has_price or has_sale_date


def _price_clause(record, *, sentence: bool = False) -> str | None:
    text = _format_aed((record or {}).get("price"))
    if not text:
        return None
    if _recent_sale_row(record):
        sold = str(
            (record or {}).get("last_sold_on") or (record or {}).get("last_sale") or ""
        ).strip()[:10]
        if sold and re.match(r"^\d{4}-\d{2}-\d{2}$", sold):
            bit = f"Recent sale {text} (recorded {sold})."
            return bit if sentence else bit.rstrip(".")
        return f"Recent sale {text}." if sentence else f"recent sale {text}"
    return f"Starting price {text}." if sentence else f"Starting price: {text}"


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


def _handover_is_upcoming(value) -> bool:
    """True only when handover is still ahead — past completion years are not 'current'."""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed.date() >= datetime.now().date()


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

    size = _record_number(record, "size_sqft", "size")
    if size is not None:
        size = int(round(size))
    baths = _record_number(record, "baths", "bathrooms")
    property_type = record.get("property_type")
    if not property_type:
        types = record.get("property_types") or []
        if isinstance(types, str):
            property_type = types
        elif isinstance(types, list):
            property_type = next((item for item in types if item), None)

    return {
        "id": record.get("listing_id") or record.get("id"),
        "listing_id": record.get("listing_id") or record.get("id"),
        "name": record.get("title") or record.get("name"),
        "title": record.get("title") or record.get("name"),
        "developer": record.get("agency_name"),
        "community": record.get("community"),
        "sub_community": record.get("sub_community"),
        "price": _json_safe(price),
        "price_label": price_label,
        "annual_rent_aed": _json_safe(price),
        "rent_frequency": frequency or None,
        "beds": _json_safe(beds),
        "baths": _json_safe(baths),
        "bedrooms": {
            "min": _json_safe(beds),
            "max": _json_safe(beds),
            "label": bedroom_label,
        },
        "size_sqft": _json_safe(size),
        "property_type": property_type,
        "property_types": [property_type] if property_type else [],
        "status": record.get("furnishing") or record.get("completion_status"),
        "photos": _listing_photo_urls(record),
        "cover_photo": record.get("cover_photo"),
        "property_finder_url": record.get("url") or record.get("property_finder_url"),
    }


def _format_project_card(record):
    """
    Convert a project database row into a frontend-friendly project card.

    Only fields that exist in the current database are exposed.
    No rental yield, fair value, deal score, ROI, or other missing
    investment metrics are invented here.
    """

    record = ground_project_row(record)
    price_kind = record.get("price_kind")
    if not price_kind and _recent_sale_row(record):
        price_kind = "recent_sale"

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "developer": record.get("developer_name"),
        "developer_id": _json_safe(record.get("developer_id")),
        "city": record.get("city"),
        "community": record.get("community"),
        "sub_community": record.get("sub_community"),
        "price": _json_safe(record.get("price")),
        "price_kind": price_kind,
        "price_label": (
            _price_clause(record)
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
        # Sale facts use last_sale; cards and chat text must show the same date.
        "last_sold_on": _json_safe(record.get("last_sold_on")),
        "last_sale": _json_safe(
            record.get("last_sale") or record.get("last_sold_on")
        ),
        "sales_count": _json_safe(
            record.get("sales_count") or record.get("sales")
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
        # Catalog projects: opens /project-intelligence/projects/<key> in the app.
        "key": record.get("key"),
    }


def _format_broker_card(record):
    """Public DLD broker fields only. Email and phone are never copied."""
    name = record.get("name") or record.get("name_en")
    photos = []
    photo = record.get("photo_url")
    if isinstance(photo, str) and photo.strip():
        photos = [photo.strip()]
    return {
        "id": _json_safe(record.get("id") or record.get("brn")),
        "brn": _json_safe(record.get("brn") or record.get("id")),
        "name": name,
        "agency_name": record.get("agency_name"),
        "card_rank": record.get("card_rank"),
        "orn": record.get("orn"),
        "license_number": record.get("license_number"),
        "awards_count": _json_safe(record.get("awards_count")),
        "card_expiry_date": _json_safe(record.get("card_expiry_date")),
        "office_rank": record.get("office_rank"),
        "photos": _json_safe(photos),
    }


def _format_community_card(record):
    """
    Convert a community database row into a frontend-friendly card.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        # The sales-data name the app's community pages use (Dubai Marina -> MARSA DUBAI).
        "official_area": record.get("official_area"),
        "focus": record.get("focus"),
        "signals": _json_safe(record.get("signals")),
        "ranking": _json_safe(record.get("ranking")),
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
    status_ranks = []
    sales_counts = []
    photo = None
    link = None
    status = None
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
        sale_count = _record_number(record, "sales_count", "recent_sales_count")
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
        if sale_count is not None:
            sales_counts.append(sale_count)
        record_status = record.get("project_status") or record.get("status")
        if record_status and status is None:
            status = record_status
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
        "status": status,
        "sales_count": _median(sales_counts),
        "photo": photo,
        "link": link,
        "area_facts": records[0].get("area_facts") if len(records) == 1 else None,
    }


def _clean_check_bullet(text: str) -> str:
    """Turn a scored-factor sentence into a Why bullet without markdown labels."""
    clean = re.sub(r"\*\*", "", str(text or "")).strip()
    clean = re.sub(
        r"^(?:Price|Yearly rent|Size|Bedrooms|Handover|Readiness|"
        r"More space|More bedrooms|More recent sales activity|"
        r"Recorded price|Recorded size|Recorded bedrooms|"
        r"Recorded sales activity|Lower price|Lower yearly rent)\s*:\s*",
        "",
        clean,
        flags=re.I,
    ).strip()
    return clean.rstrip(".")


def _pairwise_why_bullets(winner, loser, checks) -> list[str]:
    """Fact-rich head-to-head Why lines for the pairwise recommend block."""
    bullets = []
    seen = set()
    for name, _factor, text in checks:
        if name != winner["name"]:
            continue
        line = _clean_check_bullet(text)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {line}.")

    joined = " ".join(bullets).lower()
    # Extra contrast only when checks did not already cover that field.
    if (
        winner.get("price") is not None
        and loser.get("price") is None
        and "price" not in joined
    ):
        bullets.append(
            f"- {winner['name']} shows a recorded price of {_format_aed(winner['price'])}, "
            f"while {loser['name']} has no price on these records."
        )
    elif (
        winner.get("price") is not None
        and loser.get("price") is not None
        and winner["price"] != loser["price"]
        and "price" not in joined
    ):
        bullets.append(
            f"- Recorded price {_format_aed(winner['price'])} for "
            f"{winner['name']} vs {_format_aed(loser['price'])} for {loser['name']}."
        )

    joined = " ".join(bullets).lower()
    if (
        winner.get("sales_count")
        and not loser.get("sales_count")
        and "sales" not in joined
    ):
        bullets.append(
            f"- Stronger recent sales evidence on {winner['name']} "
            f"({int(winner['sales_count'])} recorded sales) than {loser['name']}."
        )
    elif (
        winner.get("sales_count")
        and loser.get("sales_count")
        and winner["sales_count"] != loser["sales_count"]
        and "sales" not in joined
    ):
        bullets.append(
            f"- More recent sales on {winner['name']} "
            f"({int(winner['sales_count'])} vs {int(loser['sales_count'])} for "
            f"{loser['name']})."
        )

    loser_edges = [
        _clean_check_bullet(text)
        for name, _factor, text in checks
        if name == loser["name"]
    ]
    if loser_edges:
        edge = loser_edges[0]
        bullets.append(
            f"- Also worth reviewing {loser['name']}: {edge}."
        )
    else:
        bullets.append(f"- Also worth reviewing: {loser['name']}.")
    return bullets[:6]


def _score_snapshots(left, right):
    """Score two sides from shared fields. Higher means stronger on those fields.

    A side with recorded price/beds/size/sales must not lose solely because the
    other side only has an earlier handover (thin catalog row).
    """

    checks = []
    _MARKET_KEYS = ("price", "beds", "size", "sales_count")

    def _table_cell(value):
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    def _coverage(side):
        return sum(1 for key in _MARKET_KEYS if side.get(key) is not None)

    rent_frequency = left.get("frequency")
    if rent_frequency == right.get("frequency") and rent_frequency:
        normalized_frequency = str(rent_frequency).strip().lower()
        frequency_label = (
            "yearly"
            if normalized_frequency in {"yearly", "annual", "annually"}
            else normalized_frequency
        )
        price_label = f"{frequency_label.capitalize()} rent"
    else:
        price_label = "Price"
    price_factor = f"lower {price_label.lower()}"

    def prefer_high(key, label, formatter):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None or left_value == right_value:
            return
        winner = left if left_value > right_value else right
        other = right if winner is left else left
        checks.append((
            winner["name"],
            label.lower(),
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
            label.lower(),
            (
                f"**{label.capitalize()}:** {winner['name']} is ahead "
                f"({formatter(winner[key])} versus {formatter(other[key])})."
            ),
        ))

    def prefer_present(key, factor, formatter, noun):
        """Credit the side that has a recorded value when the other does not."""
        left_value = left.get(key)
        right_value = right.get(key)
        if (left_value is None) == (right_value is None):
            return
        present = left if left_value is not None else right
        missing = right if present is left else left
        value = present.get(key)
        checks.append((
            present["name"],
            factor,
            (
                f"{present['name']} has a recorded {noun} ({formatter(value)}), "
                f"while {missing['name']} does not on these records."
            ),
        ))

    prefer_low("price", price_factor, lambda value: _format_aed(value) or str(value))
    prefer_present(
        "price",
        "recorded price",
        lambda value: _format_aed(value) or str(value),
        "price",
    )
    prefer_high("size", "more space", lambda value: f"{int(value):,} sqft")
    prefer_present(
        "size",
        "recorded size",
        lambda value: f"{int(value):,} sqft",
        "size",
    )

    def _bedroom_text(value):
        if value == 0:
            return "Studio"
        if value == 1:
            return "1 bedroom"
        return f"{value:g} bedrooms"

    prefer_high("beds", "more bedrooms", _bedroom_text)
    prefer_present("beds", "recorded bedrooms", _bedroom_text, "bedroom count")
    prefer_high(
        "sales_count",
        "more recent sales activity",
        lambda value: f"{int(value):,} sales",
    )
    prefer_present(
        "sales_count",
        "recorded sales activity",
        lambda value: f"{int(value):,} sales",
        "sales activity",
    )
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
            "earlier handover",
            (
                f"**Handover:** {winner['name']} is earlier "
                f"({_handover_label(winner['handover'])} versus "
                f"{_handover_label(other['handover'])})."
            ),
        ))
    from app.answers.recommend import _status_rank
    left["status_rank"] = _status_rank(left.get("status"))
    right["status_rank"] = _status_rank(right.get("status"))
    prefer_high("status_rank", "readiness", lambda value: str(int(value)))

    left_wins = sum(1 for name, _factor, _text in checks if name == left["name"])
    right_wins = len(checks) - left_wins
    if not checks or left_wins == right_wins:
        left_score = right_score = 50
    else:
        left_score = round(40 + 60 * left_wins / len(checks))
        right_score = round(40 + 60 * right_wins / len(checks))
    left["score"] = left_score
    right["score"] = right_score
    winner = left if left_score >= right_score else right
    loser = right if winner is left else left
    # Handover-only wins must not crown a thin catalog row over a fuller record.
    winner_factors = [factor for name, factor, _text in checks if name == winner["name"]]
    if (
        winner_factors
        and set(winner_factors) <= {"earlier handover"}
        and _coverage(loser) > _coverage(winner)
    ):
        checks[:] = [
            item for item in checks
            if item[1] != "earlier handover"
        ]
        left_wins = sum(1 for name, _factor, _text in checks if name == left["name"])
        right_wins = len(checks) - left_wins
        if not checks or left_wins == right_wins:
            left_score = right_score = 50
        else:
            left_score = round(40 + 60 * left_wins / len(checks))
            right_score = round(40 + 60 * right_wins / len(checks))
        left["score"] = left_score
        right["score"] = right_score
        winner = left if left_score >= right_score else right
        loser = right if winner is left else left
    table_rows = [
        f"| Factor | {_table_cell(left['name'])} | {_table_cell(right['name'])} |",
        "| --- | --- | --- |",
    ]
    fields = (
        (price_label, "price", lambda value: _format_aed(value) or str(value)),
        ("Size", "size", lambda value: f"{int(value):,} sqft"),
        ("Bedrooms", "beds", _bedroom_text),
        ("Handover", "handover", _handover_label),
    )
    comparable_fields = 0
    for label, key, formatter in fields:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None and right_value is None:
            continue
        comparable_fields += 1
        left_text = formatter(left_value) if left_value is not None else "Not recorded"
        right_text = formatter(right_value) if right_value is not None else "Not recorded"
        table_rows.append(f"| {label} | {left_text} | {right_text} |")
    table_rows.append(f"| Score | {left_score}% | {right_score}% |")
    table = "\n".join(table_rows)
    if not checks and not comparable_fields:
        intro = (
            f"{left['name']} and {right['name']} are both useful options to review; "
            "the recorded price, size, bedroom and handover details are still light, "
            "so the next step is to match them to your preferred layout."
        )
    elif left_score == right_score:
        intro = (
            f"{left['name']} and {right['name']} both score 50%. "
            "The available recorded factors do not favor either one."
        )
    else:
        intro = (
            f"{winner['name']} scores {winner['score']}% and "
            f"{loser['name']} scores {loser['score']}%. "
            f"{winner['name']} is the stronger of the two on these records."
        )

    factor_wins = {left["name"]: [], right["name"]: []}
    for name, factor, _text in checks:
        factor_wins[name].append(factor)

    if not checks:
        conclusion = (
            "Conclusion: Both options look useful to review side by side."
        )
        recommendation = (
            f"**Acot Recommend:** {left['name']} and {right['name']}\n"
            "**Why:**\n"
            "- Recorded price, size, bedroom and handover details are still light "
            "for a firm pick.\n"
            "- Match them next to the layout, community and budget you care about most."
        )
    elif left_score == right_score:
        left_bullets = _pairwise_why_bullets(left, right, checks)
        right_bullets = _pairwise_why_bullets(right, left, checks)
        # Drop the mutual "also worth reviewing" tails for a balanced block.
        left_core = [line for line in left_bullets if "also worth reviewing" not in line.lower()]
        right_core = [line for line in right_bullets if "also worth reviewing" not in line.lower()]
        left_advantages = ", ".join(factor_wins[left["name"]]) or "balanced factors"
        right_advantages = ", ".join(factor_wins[right["name"]]) or "balanced factors"
        conclusion = (
            f"Conclusion: The comparison is balanced. {left['name']} leads on "
            f"{left_advantages}, while {right['name']} leads on {right_advantages}."
        )
        why_parts = left_core[:2] + right_core[:2]
        if not why_parts:
            why_parts = [
                f"- {left['name']} and {right['name']} split the recorded factors evenly."
            ]
        why_parts.append(
            "- Pick the one that matches the lifestyle factor you value most."
        )
        recommendation = (
            f"**Acot Recommend:** {left['name']} and {right['name']}\n"
            f"**Why:**\n"
            + "\n".join(why_parts)
        )
    else:
        winning_factors = ", ".join(factor_wins[winner["name"]]) or "recorded factors"
        conclusion = (
            f"Conclusion: {winner['name']} leads on the recorded factors "
            f"({winning_factors})."
        )
        why_lines = _pairwise_why_bullets(winner, loser, checks)
        recommendation = (
            f"**Acot Recommend:** {winner['name']}\n"
            f"**Why this is the strongest match:**\n"
            + "\n".join(
                why_lines
                or [
                    f"- Stronger balance of recorded price, size, bedrooms and market "
                    f"activity than {loser['name']}.",
                    f"- Also worth reviewing: {loser['name']}.",
                ]
            )
        )

    return f"{intro}\n\n{table}\n\n{conclusion}\n{recommendation}"


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
        "area_facts": side.get("area_facts"),
    }


def _multi_comparison_answer(snapshots):
    from app.answers.recommend import format_footer, rank_projects

    def table_cell(value):
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    frequencies = {side.get("frequency") for side in snapshots}
    frequency = next(iter(frequencies)) if len(frequencies) == 1 else None
    frequency_label = (
        "yearly"
        if str(frequency or "").lower() in {"yearly", "annual", "annually"}
        else frequency
    )
    price_label = f"{str(frequency_label).capitalize()} rent" if frequency_label else "Price"

    projects = [
        {
            "name": side["name"],
            "price": side.get("price"),
            "beds": side.get("beds"),
            "size": side.get("size"),
            "handover_time": side.get("handover"),
            "status": side.get("status"),
            "sales_count": side.get("sales_count"),
        }
        for side in snapshots
    ]
    ranked = rank_projects(projects, attach_sales=False)
    score_by_name = {
        str(item["name"]).lower(): item.get("score", 50)
        for item in ranked.get("ranked") or []
    }
    for side in snapshots:
        side["score"] = score_by_name.get(str(side["name"]).lower(), 50)

    rows = [
        "| Factor | " + " | ".join(table_cell(side["name"]) for side in snapshots) + " |",
        "|" + " --- |" * (len(snapshots) + 1),
    ]
    fields = (
        (price_label, "price", lambda value: _format_aed(value) or str(value)),
        ("Size", "size", lambda value: f"{int(value):,} sqft"),
        (
            "Bedrooms",
            "beds",
            lambda value: (
                "Studio"
                if value == 0
                else f"{value:g} bedroom" if value == 1 else f"{value:g} bedrooms"
            ),
        ),
        ("Handover", "handover", _handover_label),
    )
    for label, key, formatter in fields:
        if not any(side.get(key) is not None for side in snapshots):
            continue
        values = [
            formatter(side[key]) if side.get(key) is not None else "Not recorded"
            for side in snapshots
        ]
        rows.append(
            f"| {label} | " + " | ".join(table_cell(value) for value in values) + " |"
        )
    rows.append(
        "| Score | "
        + " | ".join(table_cell(f"{side.get('score', 50)}%") for side in snapshots)
        + " |"
    )

    names = ", ".join(side["name"] for side in snapshots)
    return (
        f"Here is a side-by-side comparison of {len(snapshots)} options: "
        f"{names}.\n\n"
        + "\n".join(rows)
        + "\n\n"
        + format_footer(ranked)
    )


def _build_comparison(structured_summary):
    entities = [
        entity
        for entity in (structured_summary.get("comparison_entities") or [])
        if isinstance(entity, dict)
    ]
    if len(entities) < 2:
        return None

    snapshots = []
    for entity in entities:
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

    answer = (
        _score_snapshots(snapshots[0], snapshots[1])
        if len(snapshots) == 2
        else _multi_comparison_answer(snapshots)
    )
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
    brokers = structured_summary.get("brokers") or []

    if structured_summary.get("comparison"):
        return "comparison"

    if brokers and not listings and not (projects or []):
        return "broker_list"

    if question_type == "information" and len(listings) + len(projects or []) == 1:
        return "fact"

    if len(listings) > 1 and re.search(
        r"\b(?:which of these|which ones|which of those)\b",
        question_lower,
    ):
        return "listing_list"

    if listings and question_type == "search":
        return "listing_list"

    # The app only draws cards for project_list, listing_list and comparison, so an
    # investment answer that has projects must use one of them; the verdict still
    # shows from data.investment_analysis.
    if question_type == "investment":
        if listings and not projects:
            return "listing_list"
        if not projects:
            return "investment_analysis"

    # The agent marks a turn as a comparison when it used compare_projects.
    if question_type == "comparison" and len(listings) >= 2 and not (projects or []):
        return "listing_list"

    if question_type == "comparison" and len(projects) >= 2:
        return "project_comparison"

    if any(
        phrase in question_lower
        for phrase in (
            "compare",
            "comparison",
            "difference between",
            "vs ",
            " versus ",
            "which is better",
            "which one is better",
            "better option",
        )
    ) and len(projects) >= 2:
        return "project_comparison"

    if question_type in {"search", "investment", "fact", "prediction"} and projects:
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

    # Phase 6: ground before cards so price_kind / last_sold_on match answer text.
    projects = ground_project_rows(
        structured_summary.get("projects", []) or []
    )
    if isinstance(structured_summary, dict):
        structured_summary["projects"] = projects

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

    formatted_brokers = [
        _format_broker_card(broker)
        for broker in (structured_summary.get("brokers") or [])
        if isinstance(broker, dict)
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
            "brokers": formatted_brokers,
            "sub_communities": formatted_sub_communities,
            "documents": formatted_documents,
            "investment_analysis": _json_safe(
                investment_analysis
            ),
            "comparison": (_build_comparison(structured_summary) or {}).get("sides") or [],
            # Recorded sales (date, price, bedrooms, size, project) for "recent transactions" answers.
            "transactions": _json_safe(structured_summary.get("transactions") or []),
        },
        "meta": {
            "counts": {
                "communities": len(formatted_communities),
                "projects": len(formatted_projects),
                "listings": len(formatted_listings),
                "brokers": len(formatted_brokers),
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
    Use the database template when rows are already retrieved.

    Skips the answer LLM for browse/search, investment lists, and simple
    field asks (price/handover/bedrooms) that the template already covers.
    """

    if documents:
        return False

    if structured_summary.get("filter_no_match"):
        return True

    has_rows = bool(
        structured_summary.get("projects")
        or structured_summary.get("community")
        or structured_summary.get("sub_communities")
        or structured_summary.get("listings")
    )
    if not has_rows:
        return False

    if question_type in {"search", "investment", "ranking", "comparison"}:
        return True

    # Information field follow-ups: template already formats these.
    if question_type == "information" and re.search(
        r"\b(?:price|prices|cost|bedroom|bedrooms|hand[\s-]?over|"
        r"developer|status|payment\s+plans?|installments?|"
        r"down\s+payment)\b",
        question or "",
        flags=re.IGNORECASE,
    ):
        return True

    # Payment-plan ask with a retrieved listing (rentals have no plan fields).
    if structured_summary.get("listings") and re.search(
        r"\b(?:payment\s+plans?|installments?|down\s+payment)\b",
        question or "",
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _listing_bedroom_phrase(listing):
    beds = listing.get("beds")
    if beds is None:
        return None
    if beds == 0:
        return "Studio"
    if beds == 1:
        return "1 bedroom"
    return f"{beds} bedrooms"


def _wants_monthly_rent(question=None):
    return bool(
        re.search(
            r"\b(?:monthly|months?)\b",
            question or "",
            flags=re.IGNORECASE,
        )
    )


def _listing_display_rent(listing, monthly=False):
    """Return (amount, frequency label) for display."""

    try:
        amount = float(listing.get("price_aed"))
    except (TypeError, ValueError):
        return None, None
    frequency = str(listing.get("rent_frequency") or "").strip().lower()
    if monthly and frequency in {"yearly", "year", "annual", "annually", ""}:
        return amount / 12.0, "month"
    if monthly and frequency in {"monthly", "month"}:
        return amount, "month"
    return amount, frequency or None


def _listing_rent_phrase(listing, monthly=False):
    amount, frequency = _listing_display_rent(listing, monthly=monthly)
    price_text = _format_aed(amount)
    if not price_text:
        return None
    if frequency == "month":
        return f"Rent {price_text} / month"
    if frequency:
        return f"Rent {price_text} {frequency}"
    return f"Rent {price_text}"


def _listing_price_clause(listings, monthly=False):
    prices = []
    frequencies = set()
    for listing in listings:
        amount, frequency = _listing_display_rent(listing, monthly=monthly)
        if amount is None or not _format_aed(amount):
            continue
        prices.append(amount)
        if frequency:
            frequencies.add(frequency)

    if not prices:
        return ""

    if len(frequencies) == 1:
        freq = next(iter(frequencies))
        label = "monthly rents" if freq == "month" else f"{freq} rents"
    else:
        label = "rents"

    low = _format_aed(min(prices))
    high = _format_aed(max(prices))
    if low == high:
        return f", with a {label[:-1] if label.endswith('s') else label} of {low}"
    return f", with {label} from {low} to {high}"


def _listing_answer(listings, structured_summary, question=None):
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
    monthly = _wants_monthly_rent(question)
    price_clause = _listing_price_clause(listings, monthly=monthly)
    showing_sample = isinstance(total, int) and total > count
    wants_payment_plan = bool(
        re.search(
            r"\b(?:payment\s+plans?|installments?|down\s+payment)\b",
            question or "",
            flags=re.IGNORECASE,
        )
    )

    if wants_payment_plan and count == 1 and not showing_sample:
        title = listings[0].get("title") or "This listing"
        intro = (
            f"ACOT data does not include a payment plan for "
            f"{title}{place}. It is a rental listing. "
            "Here is what is on record:"
        )
    elif count == 1 and not showing_sample:
        title = listings[0].get("title") or "This listing"
        intro = (
            f"{title} is a rental listing{place}. "
            "Here is what is on record:"
        )
    elif wants_payment_plan:
        intro = (
            f"ACOT data does not include payment plans for these "
            f"rental listings{place}. Here is what is on record:"
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
        building = str(listing.get("sub_community") or "").strip()
        community = str(listing.get("community") or "").strip()
        if building and community and repeat_community:
            sentence = f"- **{title}**. {building}, {community}."
        elif building:
            sentence = f"- **{title}**. {building}."
        elif repeat_community and community:
            sentence = f"- **{title}**. {community}."
        else:
            sentence = f"- **{title}**."

        details = []
        rent_text = _listing_rent_phrase(listing, monthly=monthly)
        if rent_text:
            details.append(f"{rent_text}.")
        bedroom_text = _listing_bedroom_phrase(listing)
        if bedroom_text:
            details.append(f"{bedroom_text}.")
        size = listing.get("size_sqft")
        if size not in (None, ""):
            try:
                size = int(round(float(size)))
            except (OverflowError, TypeError, ValueError):
                size = None
            if size:
                details.append(f"{size} sqft.")
        if details:
            sentence = f"{sentence} {' '.join(details)}"
        lines.append(sentence)

    return "\n".join(lines)


_BROWSE_REQUEST = re.compile(
    r"\b(?:i want|i need|looking for|interested in|"
    r"show(?:\s+me|\s+only)?|give me|list|find|suggest|browse)\b",
    re.I,
)
_REQUESTED_COUNT = re.compile(
    r"\b(?:want|need|show me|give me|list|find|top|any|suggest)\s+(\d{1,2})\b"
    r"(?!\s*-?\s*(?:bed|bedroom|bedrooms|br|bath|sqft|million|m|k)\b)"
)
DEFAULT_BROWSE_COUNT = 10


def _is_project_browse_question(question: str | None) -> bool:
    """True when the user is searching/filtering projects, not asking one field about a shortlist."""
    text = question or ""
    if _BROWSE_REQUEST.search(text):
        return True
    return bool(
        re.search(r"\b(?:show|find|list)\b", text, re.I)
        and re.search(
            r"\b(?:apartments?|projects?|villas?|townhouses?|homes?|propert(?:y|ies)|flats?)\b",
            text,
            re.I,
        )
    )


def _filter_aware_browse_intro(question, projects, structured_summary) -> str | None:
    """Open with matching count (ready / beds / place / budget) — no Understood line."""
    from app.answers.intent_route import parse_bedrooms, parse_budget_aed, extract_place

    text = question or ""
    place = (
        str(structured_summary.get("catalog_place") or "").strip()
        or extract_place(text)
    )
    beds = parse_bedrooms(text)
    budget = parse_budget_aed(text)
    asked_max = structured_summary.get("asked_max_price")
    if asked_max is None:
        asked_max = budget
    ready = bool(re.search(r"\b(?:ready(?:\s+only)?|only\s+ready|completed)\b", text, re.I))
    under = bool(re.search(r"\b(?:under|below|up\s+to|max(?:imum)?)\b", text, re.I))
    above = bool(re.search(r"\b(?:above|over|starting)\b", text, re.I))
    # Place-only browse keeps the plain "I found N projects in X" opener.
    if not (beds or budget is not None or ready or structured_summary.get("budget_miss")):
        return None

    bits = []
    if ready:
        bits.append("ready")
    if beds:
        if beds[0] == 0:
            bits.append("studio")
        else:
            bits.append(f"{beds[0]}-bedroom")
    where = f" in {place}" if place and str(place).lower() != "dubai" else ""
    count = len(projects)

    # Budget miss: explain the miss, then show related projects near that range.
    if structured_summary.get("budget_miss") and asked_max is not None:
        lowest = structured_summary.get("lowest_price")
        low_bit = (
            f" Related recorded sales in that area start around AED {int(lowest):,}."
            if lowest is not None
            else " I do have related projects nearby in price."
        )
        bed_bit = ""
        if beds:
            bed_bit = "studio " if beds[0] == 0 else f"{beds[0]}-bedroom "
        ready_bit = "ready " if ready else ""
        return (
            f"I don't have recorded {ready_bit}{bed_bit}sales{where} "
            f"under AED {int(asked_max):,} in the last 12 months.{low_bit} "
            f"Here are {count} related options in that range:"
        )

    bits.append("options" if count != 1 else "option")
    budget_bit = ""
    if budget is not None:
        budget_bit = (
            f" under AED {int(budget):,}"
            if under or not above
            else f" from AED {int(budget):,}"
        )
    return (
        f"Here are {count} {' '.join(bits)}{where}{budget_bit} "
        "from recorded sales in the last 12 months when available:"
    )


def _limit_browse_projects(question, projects, structured_summary):
    """A browse like "I want 10 1-bedroom apartments" shows a short list, not every match.

    The order is kept as retrieved, so "the first one" in the next question still means the first one shown
    (the conversation memory keeps the full list in the same order). The summary is cut in place, so the
    cards the app draws match the text, and it gains project_total ("I found 165 projects. Here are 10").
    """
    if len(projects) <= 1 or not _is_project_browse_question(question):
        return projects, structured_summary, False

    match = _REQUESTED_COUNT.search(question)
    cap = int(match.group(1)) if match else DEFAULT_BROWSE_COUNT
    cap = max(1, min(cap, 20))

    if len(projects) > cap:
        structured_summary["project_total"] = len(projects)
        projects = projects[:cap]
        structured_summary["projects"] = projects
    return projects, structured_summary, True


def _limit_browse_listings(question, listings, structured_summary):
    """A rent browse shows a short unique list, matching the cards the app draws."""
    if not listings:
        return listings, structured_summary
    match = _REQUESTED_COUNT.search(question or "")
    cap = int(match.group(1)) if match else DEFAULT_BROWSE_COUNT
    cap = max(1, min(cap, 20))
    unique = []
    seen = set()
    for listing in listings:
        if not isinstance(listing, dict):
            continue
        title = re.sub(r"[^a-z0-9\s]+", " ", str(listing.get("title") or listing.get("name") or "").lower())
        title = re.sub(r"\s+", " ", title).strip()
        price = listing.get("price_aed")
        if price is None:
            price = listing.get("annual_rent_aed") or listing.get("price")
        try:
            price = int(round(float(price))) if price is not None else None
        except (OverflowError, TypeError, ValueError):
            price = None
        key = (title, price)
        if title and key in seen:
            continue
        if title:
            seen.add(key)
        unique.append(listing)
    listings = unique
    total = structured_summary.get("listing_total")
    if not isinstance(total, int) or total < len(listings):
        structured_summary["listing_total"] = len(listings)
    if len(listings) > cap:
        listings = listings[:cap]
    structured_summary["listings"] = listings
    return listings, structured_summary


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
        listings, structured_summary = _limit_browse_listings(
            question, listings, structured_summary
        )
        return _listing_answer(listings, structured_summary, question)

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

    projects = ground_project_rows(structured_summary.get("projects", []) or [])
    if isinstance(structured_summary, dict):
        structured_summary["projects"] = projects
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
    # so we can explain the result using the PostgreSQL data.
    if structured_summary.get("filter_no_match"):
        # Honest empty: never leave stale matched cards when the filter failed.
        if isinstance(structured_summary, dict):
            structured_summary["projects"] = []
        projects = []
        evidence_projects = ground_project_rows(
            structured_summary.get("filter_evidence_projects", []) or []
        )
        if isinstance(structured_summary, dict):
            structured_summary["filter_evidence_projects"] = evidence_projects

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
            return (
                "Those earlier projects don’t match this new request. "
                "Share the area, budget, or bedroom count again and I’ll search fresh "
                "instead of filtering the old list."
            )
        return (
            "None of the projects I have matched that filter. "
            "Share the area, budget, or bedroom count again and I’ll search fresh."
        )

    if projects:
        lines = []

        projects, structured_summary, browsing = _limit_browse_projects(
            q, projects, structured_summary
        )

        wants_bedrooms = any(
            term in q
            for term in ("bedroom", "bedrooms")
        )
        wants_handover = bool(
            re.search(r"hand[\s-]?over", q)
        )
        wants_developer = any(
            term in q
            for term in ("developer", "developers")
        )
        wants_status = "status" in q
        wants_price = any(
            term in q
            for term in ("price", "prices", "cost", "budget", "budgets")
        )

        # Browse/filter asks ("show only ready 1-bedroom…") must explain the match
        # with price/ready facts — never collapse to a bedrooms-only stub list.
        if browsing or _is_project_browse_question(question):
            wants_bedrooms = wants_handover = wants_developer = False
            wants_status = wants_price = False
        else:
            try:
                from app.retrieval.hybrid.question_text import is_group_attribute_question

                if not is_group_attribute_question(question):
                    wants_bedrooms = wants_handover = wants_developer = False
                    wants_status = wants_price = False
            except Exception:
                pass

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
                clause = _price_clause(project)
                value = clause or "Not available"
                return f"{name} {value}"

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
                # "2029-09-30 00:00:00" reads as "September 2029".
                try:
                    if isinstance(value, (int, float)):
                        value = _handover_label(value)
                    elif value:
                        value = datetime.fromisoformat(str(value)[:10]).strftime("%B %Y")
                except (ValueError, OverflowError, OSError):
                    pass
                lines.append(
                    f"{index}. **{name}** | Handover: {value or 'Not available'}"
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
                clause = _price_clause(project) or "Not available"
                lines.append(
                    f"{index}. {name} | {clause}"
                )
            return "\n".join(lines)

        # Default browse answer — explain the ask, then grounded project facts.
        intro = _browse_intro(
            projects,
            ambiguous_label=structured_summary.get("ambiguous_label"),
        )
        filter_intro = _filter_aware_browse_intro(
            question, projects, structured_summary
        )
        if filter_intro:
            intro = filter_intro
        project_total = structured_summary.get("project_total")
        # Budget-miss intros must not be replaced by a generic "I found N projects" block.
        if (
            not structured_summary.get("budget_miss")
            and isinstance(project_total, int)
            and project_total > len(projects)
        ):
            where = ""
            named_place = str(structured_summary.get("catalog_place") or "").strip()
            if named_place:
                where = f" in {named_place}"
            from app.answers.intent_route import parse_bedrooms, wants_more_results

            beds = parse_bedrooms(question)
            bed_bit = ""
            if beds:
                if beds[0] == 0:
                    bed_bit = " that include studio units"
                else:
                    bed_bit = f" that include {beds[0]}-bedroom units"
            more = " more" if wants_more_results(question) else ""
            ready_bit = (
                " ready"
                if re.search(r"\b(?:ready(?:\s+only)?|only\s+ready)\b", q, re.I)
                else ""
            )
            from app.answers.explain_contract import build_browse_narrative

            named_for_narrative = str(structured_summary.get("catalog_place") or "").strip() or None
            intro = build_browse_narrative(
                place=named_for_narrative,
                total=project_total,
                shown=len(projects),
                beds=beds[0] if beds else None,
                ready_only=bool(ready_bit),
                more=bool(more),
            )
            # Keep the classic count sentence for downstream asserts / scanners.
            intro = (
                f"I found {project_total}{ready_bit} projects{where}{bed_bit}. "
                f"Here are {len(projects)}{more} of them "
                "(prices from recorded sales in the last 12 months when available):\n\n"
                + "\n\n".join(intro.split("\n\n")[1:])
            )
            if filter_intro and project_total <= len(projects):
                intro = filter_intro
        elif structured_summary.get("budget_miss") and filter_intro:
            intro = filter_intro
        # Browse lists: facts only — no duplicated Understood / motive block.
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
            price_text = _price_clause(project, sentence=True)
            if price_text:
                details.append(price_text)

            bedroom_text = _format_bedroom_phrase(project)
            if bedroom_text:
                details.append(f"{bedroom_text}.")

            # Ready stock: never lead with decade-old handover years — quote the
            # recorded sale date instead. Upcoming handover stays for off-plan.
            handover_raw = project.get("handover_time")
            if _handover_is_upcoming(handover_raw):
                handover_text = _format_handover_month(handover_raw)
                if handover_text:
                    details.append(f"Handover {handover_text}.")
            else:
                sold = str(
                    project.get("last_sold_on") or project.get("last_sale") or ""
                ).strip()[:10]
                if sold and re.match(r"^\d{4}-\d{2}-\d{2}$", sold) and "recorded" not in (
                    price_text or ""
                ).lower():
                    details.append(f"Last recorded sale {sold}.")
                elif not price_text:
                    details.append("Ready (completed).")

            community = str(project.get("community") or "").strip()
            if community and len(projects) == 1:
                details.append(f"In {community}.")
            sales = project.get("sales_count") or project.get("recent_sales_count")
            if sales and len(projects) == 1:
                details.append(f"{int(sales)} recorded sales.")
            size = project.get("size_min") or project.get("size_sqft") or project.get("size")
            if isinstance(size, dict):
                size = size.get("min") or size.get("max")
            if size and len(projects) == 1:
                try:
                    details.append(f"{int(round(float(size))):,} sqft.")
                except (TypeError, ValueError):
                    pass

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

        if projects:
            lines.append("")
            lines.append(
                "Let me know if you need more details on any of these projects!"
            )

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
    "how are you": "greeting",
    "how are you doing": "greeting",
    "how are you today": "greeting",
    "how's it going": "greeting",
    "whats up": "greeting",
    "what's up": "greeting",
    "good night": "bye",
    "see you": "bye",
    "see you later": "bye",
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
        text = _format_aed(project.get("price"))
        if not text:
            continue
        prices.append(float(text.replace("AED", "").replace(",", "").strip()))

    if not prices:
        return ""

    low = _format_aed(min(prices))
    high = _format_aed(max(prices))
    if not low or not high:
        return ""
    recent = any(_recent_sale_row(project) for project in projects)
    if low == high:
        if recent:
            return f", with a recent sale of {low}"
        return f", with a starting price of {low}"
    if recent:
        return f", with recent sale prices from {low} to {high}"
    return f", with starting prices from {low} to {high}"


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
