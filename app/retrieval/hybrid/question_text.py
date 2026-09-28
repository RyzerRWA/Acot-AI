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
    (r"\blistings\b", "projects"),
    (r"\bunits\b", "projects"),
    (r"\bproperty\b", "project"),
    (r"\blisting\b", "project"),
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
    "price",
    "prices",
    "cost",
    "costs",
    "bedroom",
    "bedrooms",
    "handover",
    "handovers",
    "amenity",
    "amenities",
    "status",
    "size",
    "sizes",
    "date",
    "dates",
    "developer",
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


def _swap_project_words(text: str) -> str:
    for pattern, replacement in _PROJECT_WORDS:
        text = re.sub(pattern, replacement, text)
    return text


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
        return {"kind": "community", "community": community.group(1).strip()}

    stem_only = re.match(r"^(.+?)\s+projects?$", body)
    if stem_only:
        stem = _clean_stem(stem_only.group(1))
        if stem and stem not in {"project", "projects"}:
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
        if body not in {"project", "projects"} and "project" not in body.split()[:1]:
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
