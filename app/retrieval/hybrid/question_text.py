"""Turn a messy question into the request the search should run."""

import re


_SHORT_FORMS = (
    (r"\bjvc\b", "jumeirah village circle"),
    (r"\bjvt\b", "jumeirah village triangle"),
    (r"\bdip\b", "dubai investment park"),
    (r"\bdso\b", "dubai silicon oasis"),
    (r"\bbb\b", "business bay"),
    (r"\bhyd\b", "hyderabad"),
)

_PROJECT_WORDS = (
    (r"\bproperties\b", "projects"),
    (r"\bunits\b", "projects"),
    (r"\bproperty\b", "project"),
    (r"\bunit\b", "project"),
)

_LEAD = re.compile(
    r"^(?:please\s+)?"
    r"(?:show me|show|list|find|tell me about|tell me|give me|"
    r"i need|i want|can you|could you|what are|what is|what|which)\s+"
)

_BROWSE_LEAD = re.compile(
    r"^(?:please\s+)?(?:show me|show|list|find|tell me about|tell me|give me)\b"
)

_ATTRIBUTE_TOKENS = {
    "price","prices","cost","costs",
    "bedroom","bedrooms","handover","handovers",
    "amenity","amenities","status","size","sizes","date","dates","developer",
}

_GROUP_REFERENCE = re.compile(
    r"\b(?:their|them|these|those|which of|which ones|"
    r"the projects|the properties)\b"
)


def collapse_stretched_letters(text: str) -> str:
    """Collapse a letter repeated three or more times.

    ``hiiiii`` becomes ``hi``. A real double letter such as the one in
    ``Meydan`` stays, and digits are left alone.
    """

    return re.sub(r"([A-Za-z])\1{2,}", r"\1", text or "")


def _expand_short_forms(text: str) -> str:
    for pattern, replacement in _SHORT_FORMS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(
        r"\bjumeirah village\b(?!\s+(?:circle|triangle))",
        "jumeirah village circle",
        text,
    )
    return text


_DOMAIN_WORDS = (
    "property", "properties", "project", "projects",
    "community", "communities", "bedroom", "bedrooms",
    "apartment", "apartments", "price", "prices",
    "villa", "villas", "handover", "listing", "listings",
)

_GENERIC_WORDS = {
    "show", "me", "the", "a", "an", "all", "some", "any", "every",
    "of", "in", "at", "on", "and", "or", "for", "with", "to", "from",
    "their", "these", "those", "them", "please", "available", "options",
    "list", "find", "give", "tell", "about", "what", "which", "are", "is",
    "property", "properties", "project", "projects",
    "community", "communities", "sub",
    "bedroom", "bedrooms", "bed", "beds", "room", "rooms",
    "price", "prices", "cost", "costs", "aed", "rent", "rental", "rentals",
    "apartment", "apartments", "villa", "villas", "townhouse", "townhouses",
    "listing", "listings", "unit", "units", "home", "homes",
    "starting", "under", "below", "above", "near",
}

_CITIES = {
    "dubai", "abu dhabi", "sharjah", "ajman", "fujairah",
    "ras al khaimah", "umm al quwain",
}


def _within_one_or_two_edits(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 2:
        return False
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        if min(current) > 2:
            return False
        previous = current
    return previous[-1] <= 2


def _correct_domain_word(token: str) -> str:
    if token in _DOMAIN_WORDS or len(token) < 5:
        return token
    stems = ("prop", "proj", "commun", "bedr", "apart", "pric")
    if not token.startswith(stems):
        return token
    matches = [
        word for word in _DOMAIN_WORDS
        if _within_one_or_two_edits(token, word)
    ]
    if not matches:
        return token
    token_plural = token.endswith("s")
    matches.sort(key=lambda word: (
        abs(len(word) - len(token)),
        0 if word.endswith("s") == token_plural else 1,
    ))
    return matches[0]


def _correct_domain_typos(text: str) -> str:
    return " ".join(_correct_domain_word(token) for token in text.split())


def _normalize_bedrooms(text: str) -> str:
    text = re.sub(r"\b(\d+)\s*bed\s*rooms?\b", r"\1 bedroom", text)
    text = re.sub(r"\bbed\s+rooms?\b", "bedroom", text)
    return text


def _swap_project_words(text: str) -> str:
    for pattern, replacement in _PROJECT_WORDS:
        text = re.sub(pattern, replacement, text)
    return text


def _place_only(value: str) -> str:
    text = re.split(r"\b(?:and|with|plus)\b", value or "", maxsplit=1)[0]
    text = re.sub(
        r"\b(?:communities|community|projects?|prices?|bedrooms?|"
        r"aed|apartments?|villas?|listings?|rentals?)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip(" ,")


def _clean_compare_name(value: str) -> str:
    text = re.split(
        r"\b(?:and tell|tell me|which is|which one|for investment|"
        r"to invest|better to|and say)\b",
        value or "",
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = text.strip()
    prefix = re.compile(
        r"^(?:the|all|please|rentals?|apartments?|projects?|listings?|"
        r"villas?|for rent|to rent|in)\s+",
        flags=re.IGNORECASE,
    )
    previous = None
    while previous != text:
        previous = text
        text = prefix.sub("", text, count=1).strip()
    text = re.sub(
        r"\s+(?:for rent|to rent|rentals?|apartments?|projects?|"
        r"listings?|and)$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" ,")


def bedroom_rental_pair(question: str):
    """Two bedroom counts to compare as rentals, and the place when one is named.

    ``Compare the 2 and 3 rentals in Business Bay`` is a 2-bedroom versus
    3-bedroom rental comparison. ``Compare 1 and 2 rentals`` with no place
    stays a follow-up about rows already on screen.
    """

    text = prepare_user_question(question)
    if not text:
        return None

    bed = r"bed(?:room)?s?"
    offered = r"(?:rentals?|listings?|apartments?|flats?|properties)"
    match = None
    for pattern in (
        rf"\b(\d+)\s*{bed}\s+(?:and|vs\.?|versus)\s+(\d+)\s*{bed}\b",
        rf"\b(\d+)\s*(?:{bed}\s+)?(?:and)\s+(\d+)\s*(?:{bed}\s+)?{offered}\b",
    ):
        match = re.search(pattern, text)
        if match:
            break
    if not match:
        return None

    left = int(match.group(1))
    right = int(match.group(2))
    if left == right or left > 12 or right > 12:
        return None

    place = ""
    located = re.search(r"\b(?:in|at)\s+(.+)$", text)
    if located:
        place = _place_only(located.group(1))
    names_bedrooms = bool(re.search(rf"\b{bed}\b", text))
    if not place and not names_bedrooms:
        return None
    return left, right, place


def comparison_pair(question: str):
    """Return the two names in a comparison, without the trailing request."""

    text = (question or "").strip()
    if not text:
        return None

    match = None
    for pattern in (
        r"\bcompare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)$",
        r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+)$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            break
    if not match:
        return None

    left = _clean_compare_name(match.group(1))
    right = _clean_compare_name(match.group(2))
    if not left or not right or left.lower() == right.lower():
        return None
    return left, right


def asks_one_fact(question: str) -> bool:
    """True when the question asks for one field rather than a browse list."""

    text = (question or "").lower()
    if not text or re.search(r"\b(?:compare|versus|vs|list|show me all)\b", text):
        return False
    return bool(
        re.search(
            r"\b(?:price|rent|rental|bedroom|bedrooms|size|sqft|"
            r"handover|developer|furnished|furnishing)\b",
            text,
        )
        and re.search(r"\b(?:what|whats|what's|how much|when|who)\b", text)
    )


def asks_for_listings(question: str) -> bool:
    """True when the question is asking for rental listings, not projects."""

    text = (question or "").lower()
    if not text:
        return False

    # Investment and project questions stay on the project records.
    if re.search(r"\b(?:investment|yield|roi)\b", text):
        return False
    if re.search(r"\bprojects?\b", text) and not re.search(r"\blistings?\b", text):
        return False

    if re.search(r"\blistings?\b|\bfor rent\b|\bto rent\b|\brenting\b", text):
        return True
    # "rentals" is a listing request. "rental price" stays a project fact.
    if re.search(r"\brentals?\b", text) and not re.search(
        r"\brental\s+(?:price|yield|income)\b",
        text,
    ):
        return True

    return bool(
        re.search(r"\brent\b", text)
        and re.search(
            r"\b(?:apartments?|villas?|townhouses?|penthouses?|studios?|"
            r"bedrooms?|propert(?:y|ies)|units?|flats?)\b",
            text,
        )
    )


def normalize_for_match(text: str) -> str:
    """Lowercase, collapse stretched letters, and expand short forms.

    Used before a name is compared with a stored row. The stored row is
    still the source of the answer.
    """

    if not text:
        return ""

    text = collapse_stretched_letters(text).lower().strip()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _expand_short_forms(text)
    text = _swap_project_words(text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_user_question(text: str) -> str:
    """Read a messy or slang question as the plain request."""

    if not text:
        return ""

    text = collapse_stretched_letters(text).lower().strip()
    text = text.replace("\u2019", "'").replace("&", " and ")
    text = re.sub(r"[^a-z0-9,\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _normalize_bedrooms(text)
    text = _correct_domain_typos(text)
    text = re.sub(r"\b(?:yo|pls|lemme|wanna)\b", " ", text)
    text = re.sub(r"\bplease\b", " ", text)
    text = re.sub(r"\bgimme\b", "give me", text)
    text = re.sub(r"\bstuff\b", "projects", text)
    text = _expand_short_forms(text)
    text = _swap_project_words(text)
    return re.sub(r"\s+", " ", text).strip()


def display_phrase(value: str) -> str:
    words = str(value or "").split()
    return " ".join(
        word[:1].upper() + word[1:] if word else word
        for word in words
    )


def _strip_lead(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _LEAD.sub("", text).strip()
        text = re.sub(r"^(?:all|the)\s+", "", text).strip()
    text = re.sub(
        r"^(?:details for|information about|info about)\s+",
        "",
        text,
    ).strip()
    return text


def _clean_stem(value: str) -> str:
    stem = re.sub(r"\s+projects?$", "", (value or "").strip()).strip()
    return stem.strip(" ,")


def _name_parts(text: str) -> list:
    if "," not in (text or ""):
        return []

    parts = []
    for chunk in re.split(r",|\band\b", text):
        chunk = _strip_lead(chunk.strip())
        chunk = _clean_stem(chunk)
        if chunk and chunk not in {"and", "or"}:
            parts.append(chunk)
    return parts if len(parts) >= 3 else []


def is_attribute_phrase(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return bool(tokens & _ATTRIBUTE_TOKENS)


def is_group_attribute_question(question: str) -> bool:
    """A follow-up about price, handover, or bedrooms of the current projects.

    ``What are their handover dates?`` stays about those projects. It is
    not a search for a developer named ``handover dates``.
    """

    text = prepare_user_question(question)
    if not text or _BROWSE_LEAD.match(text):
        return False
    if not is_attribute_phrase(text):
        return False
    return bool(_GROUP_REFERENCE.search(text))


def interpret_search(question: str) -> dict:
    """Decide the search shape before any single entity is allowed to win."""

    raw = prepare_user_question(question)
    if not raw:
        return {"kind": "none"}

    compare = re.match(
        r"^compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
        raw,
    )
    if compare:
        return {
            "kind": "compare",
            "names": [
                compare.group(1).strip(),
                compare.group(2).strip(),
            ],
        }

    pattern = re.search(
        r"(?:starting with|containing|contains)\s+(.+)$",
        raw,
    )
    if pattern:
        rest = pattern.group(1).strip()
        developer = None
        by_developer = re.match(r"^(.+?)\s+by\s+(.+)$", rest)
        if by_developer:
            stem = _clean_stem(by_developer.group(1))
            developer = by_developer.group(2).strip()
        else:
            stem = _clean_stem(rest)
        if stem:
            return {
                "kind": "name_pattern",
                "stem": stem,
                "developer": developer,
                "starts_with": "starting with" in raw,
            }

    parts = _name_parts(raw)
    if parts:
        return {"kind": "multi_name", "names": parts}

    body = _strip_lead(raw)

    by_name = re.match(r"^(.+?)\s+projects?\s+by\s+(.+)$", body)
    if by_name:
        return {
            "kind": "name_pattern",
            "stem": _clean_stem(by_name.group(1)),
            "developer": by_name.group(2).strip(),
            "starts_with": False,
        }

    developed = re.match(
        r"^projects?\s+(?:developed by|by)\s+(.+)$",
        body,
    )
    if developed:
        return {"kind": "developer", "developer": developed.group(1).strip()}

    does_have = re.match(
        r"^projects?\s+does\s+(.+?)\s+have$",
        body,
    )
    if does_have:
        return {"kind": "developer", "developer": does_have.group(1).strip()}

    belong = re.match(
        r"^projects?\s+belong to\s+(.+?)\s+in\s+(.+)$",
        body,
    )
    if belong:
        return {
            "kind": "developer_community",
            "developer": belong.group(1).strip(),
            "community": belong.group(2).strip(),
        }

    developer_place = re.match(
        r"^(.+?)\s+projects?\s+in\s+(.+)$",
        body,
    )
    if developer_place:
        developer = developer_place.group(1).strip()
        community = developer_place.group(2).strip()
        if developer not in {"project", "projects"}:
            if is_attribute_phrase(developer):
                return {
                    "kind": "community",
                    "community": community,
                }
            return {
                "kind": "developer_community",
                "developer": developer,
                "community": community,
            }

    community = re.match(
        r"^projects?\s+(?:are\s+|available\s+|located\s+|that are\s+)*(?:in|at)\s+(.+)$",
        body,
    )
    if community:
        place = _place_only(community.group(1))
        if place:
            return {"kind": "community", "community": place}

    stem_only = re.match(r"^(.+?)\s+projects?$", body)
    if stem_only:
        stem = _clean_stem(stem_only.group(1))
        if stem and stem not in {"project", "projects"} and stem not in _GENERIC_WORDS:
            if len(stem.split()) == 1:
                return {"kind": "developer", "developer": stem}
            return {
                "kind": "name_pattern",
                "stem": stem,
                "developer": None,
                "starts_with": False,
            }

    mentioned = _mentioned_name(raw)
    if body and _BROWSE_LEAD.match(prepare_user_question(question) or raw):
        if (
            body not in {"project", "projects"}
            and "project" not in body.split()[:1]
            and _specific_tokens(body)
        ):
            return {"kind": "project", "name": body, "browse": True}

    if mentioned:
        return {"kind": "none", "mentioned_name": mentioned}

    return {"kind": "none"}


def _mentioned_name(text: str) -> str:
    match = re.search(
        r"(?:price|prices|cost|bedrooms?|handover|developer|status|"
        r"amenities|size|details|information)\s+(?:of|for|about)\s+(.+)$",
        text,
    )
    if not match:
        return ""
    name = _clean_stem(match.group(1))
    if not name or name in {"these", "those", "them", "their"}:
        return ""
    return name


def subject_for(shape: dict) -> tuple:
    """Return the phrase to name when a search finds nothing, and its kind."""

    kind = (shape or {}).get("kind")
    if kind == "community":
        return display_phrase(shape.get("community")), "place"
    if kind == "developer":
        return display_phrase(shape.get("developer")), "developer"
    if kind == "developer_community":
        developer = display_phrase(shape.get("developer"))
        community = display_phrase(shape.get("community"))
        return f"{developer} projects in {community}", "developer_community"
    if kind == "name_pattern":
        return display_phrase(shape.get("stem")), "pattern"
    if kind == "project":
        return display_phrase(shape.get("name")), "project"
    if kind == "multi_name":
        names = [
            display_phrase(name)
            for name in (shape.get("names") or [])
            if name
        ]
        return ", ".join(names), "project"
    mentioned = (shape or {}).get("mentioned_name")
    if mentioned:
        return display_phrase(mentioned), "project"
    return "", ""


def _specific_tokens(text: str, place: str = "") -> list:
    place_tokens = set((place or "").lower().split())
    tokens = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if token.isdigit() or token in _GENERIC_WORDS or token in place_tokens:
            continue
        if token in _CITIES or any(token in city.split() for city in _CITIES):
            continue
        tokens.append(token)
    return tokens


def _bedroom_count(text: str):
    match = re.search(r"\b(\d+)\s+bedrooms?\b", text or "")
    if match:
        return int(match.group(1))
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    }
    match = re.search(
        r"\b(one|two|three|four|five|six)\s+bedrooms?\b",
        text or "",
    )
    if match:
        return words[match.group(1)]
    if re.search(r"\bstudio\s+bed", text or ""):
        return 0
    return None


def _mentioned_place(text: str) -> str:
    match = re.search(r"\b(?:in|at)\s+(.+)$", text or "")
    if not match:
        return ""
    return _place_only(match.group(1))


def catalog_request(question: str):
    """A browse that names no project: a city, a community list, or prices.

    ``Show me proprties in dubai and communities`` and
    ``show me the 1bed room AED, bedrooms and prices`` land here.
    A named tower, developer, or comparison stays on the existing search.
    """

    text = prepare_user_question(question)
    if not text or comparison_pair(text):
        return None
    if re.search(r"\b(?:invest(?:ment|ing)?|compare|versus|vs|better|yield|roi)\b", text):
        return None
    if re.search(r"\b(?:first|second|third|these|those|them|their)\b", text):
        return None

    shape = interpret_search(text)
    kind = shape.get("kind")
    if kind in {"compare", "name_pattern", "developer_community"}:
        return None
    if kind == "multi_name" and any(
        _specific_tokens(name) for name in (shape.get("names") or [])
    ):
        return None
    if kind == "developer" and str(shape.get("developer") or "").lower() not in _GENERIC_WORDS:
        return None
    if shape.get("mentioned_name"):
        return None

    place = ""
    if kind == "community":
        place = str(shape.get("community") or "").strip()
        if place.lower() not in _CITIES:
            return None
    else:
        place = _mentioned_place(text)
        if place and place.lower() not in _CITIES and _specific_tokens(text, place):
            return None

    if kind == "project" and _specific_tokens(shape.get("name") or "", place):
        return None
    if _specific_tokens(text, place):
        return None

    asks_community = bool(re.search(r"\bcommunities\b", text))
    asks_project = bool(re.search(r"\bprojects?\b", text))
    asks_price = bool(re.search(r"\b(?:prices?|aed|cost|rent)\b", text))
    bedrooms = _bedroom_count(text)
    asks_bedroom = bedrooms is not None or bool(re.search(r"\bbedrooms?\b", text))

    if not (asks_community or asks_project or asks_price or asks_bedroom or place):
        return None

    listings = bool(bedrooms is not None or (asks_price and not asks_project))
    projects = bool(asks_project and bedrooms is None)
    communities = asks_community
    if place and not listings and not projects and not communities:
        projects = True
        communities = True
    if not listings and not projects and not communities:
        if asks_bedroom or asks_price:
            listings = True
        else:
            projects = True

    return {
        "place": place,
        "communities": communities,
        "projects": projects,
        "listings": listings,
        "bedrooms": bedrooms,
    }
