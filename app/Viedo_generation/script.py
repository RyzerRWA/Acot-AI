"""
Build a narration script from a complete video pack (slideshow cues + TTS text).

No LLM. Every sentence is filled from pack facts only.
Includes: bedrooms, recent sale, recent rent, handover.
"""

from __future__ import annotations

from typing import Any

from app.Viedo_generation.narration import FPS

SLIDE_SECONDS = (5, 18, 18, 4)
MAX_SCRIPT_PROJECTS = 2


def build_script(pack: dict[str, Any]) -> dict[str, Any]:
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
    if len(projects) < 2:
        return {
            "ok": False,
            "reason": "Need at least 2 complete projects for the script.",
            "full_text": "",
            "word_count": 0,
            "estimated_seconds": 0.0,
            "slides": [],
        }

    bedrooms = pack.get("bedrooms")
    bed_labels = _collect_bedroom_labels(projects, bedrooms)
    communities = _unique_communities(projects)
    slides: list[dict[str, Any]] = []

    intro_narration = "Come, let's take a quick look at these two properties."
    intro_lines = [
        _join_names(communities),
        _bedroom_focus_line(bed_labels) or "Selected from recent search activity",
        "Recent sales and rents from market records",
    ]
    slides.append(
        {
            "kind": "intro",
            "title": "Property overview",
            "subtitle": _join_names(communities),
            "lines": [line for line in intro_lines if line],
            "narration": intro_narration,
            "project_index": None,
            "photo_urls": _cover_photos(projects, 2),
        }
    )

    for i, project in enumerate(projects):
        cues = _project_cues(project, i + 1, len(projects))
        narration = " ".join(cue["narration"] for cue in cues)
        slides.append(
            {
                "kind": "project",
                "title": project.get("name") or "Project",
                "subtitle": project.get("community"),
                "lines": _project_caption_lines(project),
                "narration": narration,
                "cues": cues,
                "project_index": i,
                "photo_urls": list(project.get("photos") or [])[:4],
            }
        )

    outro_narration = "Like either property? Ask your ACOT agent about a viewing."
    slides.append(
        {
            "kind": "outro",
            "title": "Summary",
            "subtitle": f"{len(projects)} projects compared",
            "lines": [
                f"{len(projects)} projects · recent sale & rent",
                _bedroom_focus_line(bed_labels) or "Bedroom details from latest deals",
                "Ask for a viewing or a deeper comparison",
            ],
            "narration": outro_narration,
            "project_index": None,
            "photo_urls": _cover_photos(projects, 2),
        }
    )

    for slide, seconds in zip(slides, SLIDE_SECONDS, strict=True):
        slide["target_frames"] = seconds * FPS
        slide.setdefault("cues", [{"narration": slide["narration"], "lines": slide["lines"]}])
    full_text = " ".join(s["narration"].strip() for s in slides if s.get("narration"))
    full_text = " ".join(full_text.split())
    words = len(full_text.split())

    return {
        "ok": True,
        "reason": None,
        "full_text": full_text,
        "word_count": words,
        "estimated_seconds": sum(SLIDE_SECONDS),
        "slides": slides,
    }


def _project_cues(project: dict[str, Any], index: int, total: int) -> list[dict[str, Any]]:
    """Keep every spoken fact paired with its caption before calling TTS."""
    name = project.get("name") or "This project"
    community = project.get("community") or "its community"
    sale = project.get("recent_sale") or {}
    rent = project.get("rent") or {}
    handover = project.get("handover")
    range_bit = _project_bed_range_spoken(project)

    opener = f"Next, {name} in {community}."
    if index == 1:
        opener = f"First, {name} in {community}."
    elif index == total:
        opener = f"Next, {name} in {community}."
    if range_bit:
        opener = f"{opener} {range_bit}"

    sale_bed = _bed_label(sale.get("bedrooms"))
    rent_bed = _bed_label(rent.get("bedrooms"))
    sale_bit = (
        f" A recent sale was {_money_spoken(sale.get('price'))}"
        f"{f' for a {sale_bed}' if sale_bed else ''}"
        "."
    )
    rent_bit = (
        f" A recent rental was {_money_spoken(rent.get('annual'))} a year"
        f"{f' for a {rent_bed}' if rent_bed else ''}"
        "."
    )
    hand_bit = _handover_spoken(handover)
    if project.get("completion_date") and handover == "ready":
        hand_bit = f" Catalog completion was {_date_spoken(project['completion_date'])}."
    captions = _project_caption_lines(project)
    cues = [
        {"narration": opener, "lines": [line for line in captions if line.startswith(("Developer:", "Bedrooms:"))]},
        {"narration": sale_bit.strip(), "lines": [line for line in captions if line.startswith("Sale ")]},
        {"narration": rent_bit.strip(), "lines": [line for line in captions if line.startswith("Rent ")]},
    ]
    if hand_bit:
        cues.append({"narration": hand_bit.strip(), "lines": [line for line in captions if line.startswith(("Ready", "Handover", "Catalog completion"))]})
    return cues


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
