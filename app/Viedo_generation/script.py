"""
Build a narration script from a complete video pack (slideshow cues + TTS text).

No LLM. Every sentence is filled from pack facts only.
Includes: bedrooms, recent sale, recent rent, handover.
"""

from __future__ import annotations

from typing import Any

from app.Viedo_generation.narration import FPS

SLIDE_SECONDS = (5, 18, 18, 4)
SINGLE_SLIDE_SECONDS = (5, 24, 4)
INSIDE_SLIDE_SECONDS = (5, 18, 12, 4)
ROOM_WORDS_HI = {"bedroom": "बेडरूम", "living room": "लिविंग रूम", "kitchen": "किचन",
                 "bathroom": "बाथरूम", "dining": "डाइनिंग एरिया", "balcony": "बालकनी"}
MAX_SCRIPT_PROJECTS = 2


def build_script(
    pack: dict[str, Any], *, language: str = "en", cinematic: bool = False,
    broker_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Turn a complete pack into narration + slide cues.

    Returns:
      {
        "ok": bool,
        "reason": str | None,
        "full_text": str,          # one block for TTS
        "word_count": int,
        "estimated_seconds": float,
        "slides": [
          {
            "kind": "intro" | "project" | "outro",
            "title": str,
            "subtitle": str | None,
            "lines": list[str],    # on-screen captions
            "narration": str,      # spoken for this slide
            "project_index": int | None,
            "photo_urls": list[str],
            "cues": list[dict],    # paired narration/captions; measured by narration.py
          },
          ...
        ],
      }
    """
    if not pack or not pack.get("ok"):
        return {
            "ok": False,
            "reason": (pack or {}).get("reason") or "Pack is not ready.",
            "full_text": "",
            "word_count": 0,
            "estimated_seconds": 0.0,
            "slides": [],
        }

    projects = [p for p in (pack.get("projects") or []) if isinstance(p, dict)]
    projects = projects[:MAX_SCRIPT_PROJECTS]
    if not projects:
        return {
            "ok": False,
            "reason": "Need at least 1 complete project for the script.",
            "full_text": "",
            "word_count": 0,
            "estimated_seconds": 0.0,
            "slides": [],
        }

    bedrooms = pack.get("bedrooms")
    bed_labels = _collect_bedroom_labels(projects, bedrooms)
    communities = _unique_communities(projects)
    slides: list[dict[str, Any]] = []

    single = len(projects) == 1
    rooms = [x for x in (projects[0].get("interior_photos") or []) if x.get("url")] if single else []
    room_urls = [x["url"] for x in rooms]
    # Outside shots for the intro, facts and close; room shots get their own "step inside" part.
    photos = [u for u in (projects[0].get("photos") or []) if u not in room_urls] or room_urls[:1]
    say = PHRASES["hi" if language == "hi" else "en"]
    intro_narration = (
        say["intro_one"].format(name=projects[0].get("name") or "this property", community=projects[0].get("community") or "Dubai")
        if single else say["intro_two"]
    )
    amenity_line = " · ".join((projects[0].get("amenities") or [])[:4]) if single else ""
    intro_lines = [
        amenity_line or _join_names(communities),
        _bedroom_focus_line(bed_labels) or "Selected from recent search activity",
        "Recent sales and rents from market records",
    ]
    slides.append(
        {
            "kind": "intro",
            "title": (projects[0].get("name") or "Property overview") if single else "Property overview",
            "subtitle": _join_names(communities),
            "lines": [line for line in intro_lines if line],
            "narration": intro_narration,
            "project_index": None,
            "photo_urls": photos[:1] if single else _cover_photos(projects, 2),
        }
    )

    for i, project in enumerate(projects):
        cues = _project_cues(project, i + 1, len(projects), say, include_amenities=not cinematic)
        narration = " ".join(cue["narration"] for cue in cues)
        slides.append(
            {
                "kind": "project",
                "title": project.get("name") or "Project",
                "subtitle": project.get("community"),
                "uses_listing_photos": project.get("uses_listing_photos", False),
                "lines": _project_caption_lines(project),
                "stats": _project_stats(project),
                "narration": narration,
                "cues": cues,
                "project_index": i,
                # One property: the intro and close use their own photos, the middle gets the rest.
                "photo_urls": (photos[1:5] if len(photos) >= 3 else photos[:4]) if single else list(project.get("photos") or [])[:4],
            }
        )

    if rooms:
        hindi = language == "hi"
        names = list(dict.fromkeys(x["room"] for x in rooms if x.get("room")))[:3]
        names = [ROOM_WORDS_HI.get(n, n) if hindi else n for n in names]
        joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {say['and']} {names[-1]}" if names else ""
        inside = say["inside"] + (f" {say['inside_rooms'].format(rooms=joined)}" if joined else "")
        project = projects[0]
        slides.append({
            "kind": "inside",
            "uses_listing_photos": project.get("uses_listing_photos", False),
            "title": project.get("name") or "Project",
            "subtitle": "Inside the home" + (f" · {' · '.join(n.title() for n in dict.fromkeys(x['room'] for x in rooms if x.get('room')))}" if any(x.get("room") for x in rooms) else ""),
            "lines": _project_caption_lines(project),
            "stats": _project_stats(project),
            "narration": inside,
            "project_index": None,
            "photo_urls": room_urls[:4],
        })

    outro_narration = say["outro_one"] if single else say["outro_two"]
    slides.append(
        {
            "kind": "outro",
            "title": "Summary",
            "subtitle": projects[0].get("name") if single else f"{len(projects)} projects compared",
            "lines": [
                "Recent sale & rent" if single else f"{len(projects)} projects · recent sale & rent",
                _bedroom_focus_line(bed_labels) or "Bedroom details from latest deals",
                "Ask for a viewing or a deeper comparison",
            ],
            "narration": outro_narration,
            "project_index": None,
            "photo_urls": photos[-1:] if single else _cover_photos(projects, 2),
        }
    )

    seconds_per_slide = (INSIDE_SLIDE_SECONDS if rooms else SINGLE_SLIDE_SECONDS) if single else SLIDE_SECONDS
    if cinematic:
        slides = [slide for slide in slides if slide["kind"] in ("project", "inside")]
        if single:
            amenity_urls = set(projects[0].get("amenity_urls") or [])
            slides[0]["photo_urls"] = [url for url in photos if url not in amenity_urls] or photos
            seconds_per_slide = (17, 22) if rooms else (33,)
        else:
            seconds_per_slide = (22.48, 22.52)
        if not single:
            for slide, project in zip(slides, projects):
                interiors = [item["url"] for item in project.get("interior_photos", []) if item.get("url")]
                amenities = set(project.get("amenity_urls") or [])
                exterior_photos = [url for url in project.get("photos", []) if url not in interiors and url not in amenities]
                if interiors:
                    slide["photo_urls"] = exterior_photos[:1] + interiors
                elif exterior_photos:
                    slide["photo_urls"] = exterior_photos[:4]
        amenities = _amenity_slides(projects, slides, say)
        slides.extend(amenities)
        seconds_per_slide = (*seconds_per_slide, *(4 for _ in amenities))
        if broker_card:
            from app.Viedo_generation.render import AGENT_PANEL_SECONDS

            agent_name = str(broker_card.get("name") or "ACOT Agent").strip()
            slides.append({
                "kind": "agent",
                "narration": say["agent"].format(name=agent_name),
            })
            seconds_per_slide = (*seconds_per_slide, AGENT_PANEL_SECONDS)
    for slide, seconds in zip(slides, seconds_per_slide, strict=True):
        slide["target_frames"] = round(seconds * FPS)
        slide.setdefault("cues", [{"narration": slide["narration"], "lines": slide.get("lines", [])}])
    full_text = " ".join(s["narration"].strip() for s in slides if s.get("narration"))
    full_text = " ".join(full_text.split())
    words = len(full_text.split())

    return {
        "ok": True,
        "reason": None,
        "full_text": full_text,
        "word_count": words,
        "estimated_seconds": sum(seconds_per_slide),
        "slides": slides,
    }


def _amenity_slides(projects: list[dict[str, Any]], slides: list[dict], say: dict[str, str]) -> list[dict]:
    result = []
    def amenity_key(name: str) -> str:
        return name.strip().casefold().replace("shared ", "").replace("swimming pool", "pool")

    for i, project in enumerate(projects):
        name = project["name"]
        shown = (project.get("amenity_photos") or [])[:2]
        for shot in shown:
            result.append({
                "kind": "amenity", "title": shot["name"], "subtitle": name,
                "uses_listing_photos": project.get("uses_listing_photos", False),
                "narration": say["amenity_shot"].format(name=shot["name"]),
                "photo_urls": [shot["url"]],
                "amenity_photo": True,
                "focus_x": shot.get("focus_x"),
            })
        shown_names = {amenity_key(shot["name"]) for shot in shown}
        listed = [item for item in (project.get("amenities") or []) if item and amenity_key(item) not in shown_names][:2]
        if listed:
            result.append({
                "kind": "amenity", "title": "Amenities", "subtitle": " · ".join(listed),
                "uses_listing_photos": project.get("uses_listing_photos", False),
                "narration": say["amenities"].format(list=f" {say['and']} ".join(listed)),
                "photo_urls": (slides[i].get("photo_urls") or [])[:1],
            })
    return result[:3]


def _project_cues(
    project: dict[str, Any], index: int, total: int, say: dict[str, str] | None = None,
    *, include_amenities: bool = True,
) -> list[dict[str, Any]]:
    """Keep every spoken fact paired with its caption before calling TTS."""
    say = say or PHRASES["en"]
    hindi = say is PHRASES["hi"]
    name = project.get("name") or "This project"
    community = project.get("community") or "Dubai"
    sale = project.get("recent_sale") or {}
    rent = project.get("rent") or {}
    handover = project.get("handover")

    key = "open_one" if total == 1 else ("open_first" if index == 1 else "open_next")
    opener = say[key].format(name=name, community=community)
    beds = _bed_nums(project)
    if beds:
        span = _bed_word(beds[0], hindi) if len(beds) == 1 else f"{_bed_word(beds[0], hindi)} {say['to']} {_bed_word(beds[-1], hindi)}"
        opener = f"{opener} {say['beds'].format(span=span)}"

    def money(value: Any) -> str:
        return _money_hindi(value) if hindi else _money_spoken(value)

    def home(bedrooms: Any) -> str:
        n = _bed_int(bedrooms)
        if n is None:
            return say["a_home"]
        return say["a_bed_home"].format(bed=_bed_word(n, hindi))

    sale_bit = say["sale"].format(home=home(sale.get("bedrooms")), money=money(sale.get("price")))
    rent_bit = say["rent"].format(home=home(rent.get("bedrooms")), money=money(rent.get("annual")))
    if handover == "ready" and project.get("completion_date"):
        hand_bit = say["completed"].format(date=_month_year(project["completion_date"], hindi))
    elif handover == "ready":
        hand_bit = say["ready"]
    elif handover:
        hand_bit = say["handover"].format(date=_translate_months(str(handover)) if hindi else handover)
    else:
        hand_bit = ""
    captions = _project_caption_lines(project)
    amenities = [a for a in (project.get("amenities") or []) if a][:4]
    if amenities and include_amenities:
        joined = amenities[0] if len(amenities) == 1 else f"{', '.join(amenities[:-1])} {say['and']} {amenities[-1]}"
        opener = f"{opener} {say['amenities'].format(list=joined)}"
    cues = [
        {"narration": opener, "lines": [line for line in captions if line.startswith(("Developer:", "Bedrooms:"))]},
        {"narration": sale_bit, "lines": [line for line in captions if line.startswith("Sale ")]},
        {"narration": rent_bit, "lines": [line for line in captions if line.startswith("Rent ")]},
    ]
    if hand_bit:
        cues.append({"narration": hand_bit, "lines": [line for line in captions if line.startswith(("Ready", "Handover", "Catalog completion"))]})
    return cues


def _project_stats(project: dict[str, Any]) -> list[dict[str, str]]:
    """Three on-screen facts for the lower-third: sale, rent, handover."""
    sale = project.get("recent_sale") or {}
    rent = project.get("rent") or {}
    stats = []
    if sale.get("price") is not None:
        sub = " / ".join(x for x in (_date_spoken(sale.get("date")), _bed_label_short(sale.get("bedrooms"))) if x)
        stats.append({"label": "RECENT SALE", "value": f"AED {int(round(float(sale['price']))):,}", "sub": sub})
    if rent.get("annual") is not None:
        sub = " / ".join(x for x in (_date_spoken(rent.get("date")), _bed_label_short(rent.get("bedrooms"))) if x)
        stats.append({"label": "RECENT RENT", "value": f"AED {int(round(float(rent['annual']))):,} / year", "sub": sub})
    handover = project.get("handover")
    if handover == "ready":
        done = _date_spoken(project.get("completion_date"))
        stats.append({"label": "HANDOVER", "value": "Ready to move in", "sub": f"Completed {done}" if done else ""})
    elif handover:
        stats.append({"label": "HANDOVER", "value": str(handover), "sub": "Expected handover"})
    return stats


# Friendly, everyday wording. The Hindi lines use "we" for the agent so they fit any voice.
PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "intro_one": "Hi! Here's a quick look at a place I think you'll like.",
        "intro_two": "Hi! Let's take a quick look at two homes you might like.",
        "open_one": "This is {name}, in {community}.",
        "open_first": "First up, {name} in {community}.",
        "open_next": "Next, {name} in {community}.",
        "beds": "You'll find {span} homes here.",
        "to": "to",
        "and": "and",
        "amenities": "You also get {list}, right at home.",
        "amenity_shot": "Here's a look at the {name}.",
        "agent": "For a viewing, contact your agent, {name}.",
        "inside": "Now let's step inside.",
        "inside_rooms": "Here's a look at the {rooms}.",
        "a_home": "a home",
        "a_bed_home": "a {bed}",
        "sale": "Recently, {home} here sold for about {money}.",
        "rent": "And if you'd rather rent, {home} went for about {money} a year.",
        "ready": "The best part? It's ready to move in.",
        "completed": "It was finished in {date}, so you can move in right away.",
        "handover": "Handover is planned for {date}.",
        "outro_one": "Like what you see? Just message me here and I'll set up a viewing for you.",
        "outro_two": "Like either of them? Just message me here and I'll set up a viewing.",
    },
    "hi": {
        "intro_one": "नमस्ते! चलिए, एक ऐसा घर देखते हैं जो आपको ज़रूर पसंद आएगा।",
        "intro_two": "नमस्ते! चलिए, आपके लिए दो बढ़िया घर देखते हैं।",
        "open_one": "यह है {community} में {name}।",
        "open_first": "सबसे पहले, {community} में {name}।",
        "open_next": "अब देखते हैं, {community} में {name}।",
        "beds": "यहाँ {span} तक के घर मिलते हैं।",
        "to": "से",
        "and": "और",
        "amenities": "यहाँ {list} जैसी सुविधाएँ भी मिलती हैं।",
        "amenity_shot": "यहाँ देखिए {name}।",
        "agent": "घर देखने के लिए अपने एजेंट {name} से संपर्क करें।",
        "inside": "चलिए, अब घर के अंदर चलते हैं।",
        "inside_rooms": "देखिए {rooms}।",
        "a_home": "एक घर",
        "a_bed_home": "एक {bed} वाला घर",
        "sale": "हाल ही में यहाँ {home} लगभग {money} में बिका है।",
        "rent": "और अगर आप किराए पर लेना चाहें, तो {home} लगभग {money} सालाना में मिला है।",
        "ready": "सबसे अच्छी बात, यह घर रहने के लिए बिल्कुल तैयार है।",
        "completed": "यह {date} में बनकर तैयार हो चुका है, यानी आप तुरंत शिफ्ट हो सकते हैं।",
        "handover": "इसका हैंडओवर {date} तक होने की उम्मीद है।",
        "outro_one": "पसंद आया? बस यहीं मैसेज कीजिए, हम आपके लिए विज़िट प्लान कर देंगे।",
        "outro_two": "इनमें से कोई पसंद आया? बस यहीं मैसेज कीजिए, हम विज़िट प्लान कर देंगे।",
    },
}

_HINDI_MONTHS = {
    "Jan": "जनवरी", "Feb": "फ़रवरी", "Mar": "मार्च", "Apr": "अप्रैल", "May": "मई", "Jun": "जून",
    "Jul": "जुलाई", "Aug": "अगस्त", "Sep": "सितंबर", "Oct": "अक्टूबर", "Nov": "नवंबर", "Dec": "दिसंबर",
}


def _translate_months(text: str) -> str:
    for en, hi in _HINDI_MONTHS.items():
        text = text.replace(en, hi)
    return text


def _month_year(value: Any, hindi: bool) -> str:
    label = _date_spoken(value) or str(value)
    parts = label.split()
    label = " ".join(parts[1:]) if len(parts) == 3 else label  # "30 Apr 2025" -> "Apr 2025"
    return _translate_months(label) if hindi else label


def _bed_nums(project: dict[str, Any]) -> list[int]:
    values = (project.get("bedrooms_min"), project.get("bedrooms_max"),
              (project.get("recent_sale") or {}).get("bedrooms"), (project.get("rent") or {}).get("bedrooms"))
    return sorted({n for n in (_bed_int(v) for v in values) if n is not None})


def _bed_word(n: int, hindi: bool) -> str:
    if n <= 0:
        return "स्टूडियो" if hindi else "studio"
    return f"{n} बेडरूम" if hindi else f"{n}-bedroom"


def _money_hindi(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if n >= 1_000_000:
        return f"{f'{n / 1_000_000:.1f}'.rstrip('0').rstrip('.')} मिलियन दिरहम"
    return f"{round(n / 1000):,} हज़ार दिरहम"


def _project_caption_lines(project: dict[str, Any]) -> list[str]:
    sale = project.get("recent_sale") or {}
    rent = project.get("rent") or {}
    lines: list[str] = [
        project.get("name") or "Project",
        project.get("community") or "",
    ]

    bed_range = _project_bed_range_caption(project)
    if project.get("developer"):
        lines.append(f"Developer: {project['developer']}")
    if bed_range:
        lines.append(bed_range)

    if sale.get("price") is not None:
        sale_bed = _bed_label_short(sale.get("bedrooms"))
        bed_part = f" · {sale_bed}" if sale_bed else ""
        lines.append(
            f"Sale {_money(sale.get('price'))}{bed_part}{_on_date_short(sale.get('date'))}"
        )
    if rent.get("annual") is not None:
        rent_bed = _bed_label_short(rent.get("bedrooms"))
        bed_part = f" · {rent_bed}" if rent_bed else ""
        registration = rent.get("registration_date")
        date_part = f" · registered {_date_spoken(registration)}" if registration else _on_date_short(rent.get("date"))
        lines.append(f"Rent {_money(rent.get('annual'))}/yr{bed_part}{date_part}")
    if project.get("handover"):
        hand = project["handover"]
        lines.append("Ready to move in" if hand == "ready" else f"Handover {hand}")
        if project.get("completion_date"):
            lines.append(f"Catalog completion: {_date_spoken(project['completion_date'])}")
    return [line for line in lines if line]


def _collect_bedroom_labels(
    projects: list[dict[str, Any]],
    pack_bedrooms: Any,
) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        label = _bed_label(raw)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)

    if pack_bedrooms is not None:
        add(pack_bedrooms)
    for project in projects:
        add((project.get("recent_sale") or {}).get("bedrooms"))
        add((project.get("rent") or {}).get("bedrooms"))
        add(project.get("bedrooms_min"))
        add(project.get("bedrooms_max"))
    return labels


def _bedroom_focus_line(bed_labels: list[str]) -> str | None:
    if not bed_labels:
        return None
    return f"Bedrooms: {_join_names(bed_labels)}"


def _project_bed_range_caption(project: dict[str, Any]) -> str | None:
    lo = project.get("bedrooms_min")
    hi = project.get("bedrooms_max")
    sale_b = (project.get("recent_sale") or {}).get("bedrooms")
    rent_b = (project.get("rent") or {}).get("bedrooms")
    nums: list[int] = []
    for value in (lo, hi, sale_b, rent_b):
        n = _bed_int(value)
        if n is not None and n not in nums:
            nums.append(n)
    if not nums:
        return None
    nums.sort()
    if len(nums) == 1:
        return f"Bedrooms: {_bed_label_short(nums[0])}"
    return f"Bedrooms: {_bed_label_short(nums[0])}–{_bed_label_short(nums[-1])}"


def _project_bed_range_spoken(project: dict[str, Any]) -> str:
    caption = _project_bed_range_caption(project)
    if not caption:
        return ""
    # "Bedrooms: 2-bed–3-bed" → spoken
    detail = caption.replace("Bedrooms: ", "")
    return f"Bedrooms: {detail}."


def _handover_spoken(handover: Any) -> str:
    if not handover:
        return ""
    text = str(handover).strip()
    if text.lower() == "ready":
        return " It is ready to move in."
    return f" Handover is {text}."


def _bed_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n


def _bed_label(value: Any) -> str | None:
    n = _bed_int(value)
    if n is None:
        return None
    if n <= 0:
        return "studio"
    if n == 1:
        return "1-bedroom"
    return f"{n}-bedroom"


def _bed_label_short(value: Any) -> str | None:
    n = _bed_int(value)
    if n is None:
        return None
    if n <= 0:
        return "studio"
    return f"{n}-bed"


def _on_date_short(value: Any) -> str:
    label = _date_spoken(value)
    return f" · {label}" if label else ""


def _date_spoken(value: Any) -> str | None:
    if not value:
        return None
    try:
        from datetime import datetime

        d = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        return d.strftime("%d %b %Y")
    except ValueError:
        return str(value)[:10]


def _money(value: Any) -> str:
    if value is None:
        return "an undisclosed amount"
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if n >= 1_000_000:
        m = n / 1_000_000
        text = f"{m:.1f}".rstrip("0").rstrip(".")
        return f"AED {text} million"
    return f"AED {int(round(n)):,}"


def _money_spoken(value: Any) -> str:
    label = _money(value)
    return f"{label[4:]} dirhams" if label.startswith("AED ") else label


def _unique_communities(projects: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in projects:
        name = (p.get("community") or "").strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _join_names(names: list[Any] | None) -> str:
    clean = [str(n).strip() for n in (names or []) if n and str(n).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _cover_photos(projects: list[dict[str, Any]], count: int) -> list[str]:
    urls: list[str] = []
    for p in projects:
        photos = p.get("photos") or []
        if photos and isinstance(photos[0], str):
            urls.append(photos[0])
        if len(urls) >= count:
            break
    return urls


# Quick local check:
#   python -c "from app.Viedo_generation.history import searched_projects; from app.Viedo_generation.pack import build_pack; from app.Viedo_generation.script import build_script; s=build_script(build_pack(searched_projects('USER_ID'))); print(s['full_text']); print(s['word_count'], s['estimated_seconds'])"
