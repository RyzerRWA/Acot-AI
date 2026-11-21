"""
Build a ~60 second narration script from a complete video pack.

No LLM. Every sentence is filled from pack facts only.
Focus: sales, rents, handover, then a short wrap-up.
"""

from __future__ import annotations

from typing import Any

# Rough speaking pace for Google Neural2 TTS (words per second).
WORDS_PER_SECOND = 2.4
TARGET_SECONDS = 60
# 3 projects + intro/outro fits ~60s cleanly; 4 projects runs long.
MAX_SCRIPT_PROJECTS = 3


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
    communities = _unique_communities(projects)
    slides: list[dict[str, Any]] = []

    intro_narration = _intro_narration(communities, bedrooms, len(projects))
    slides.append(
        {
            "kind": "intro",
            "title": "Property overview",
            "subtitle": _join_names(communities),
            "lines": [
                _join_names(communities),
                _bedroom_line(bedrooms) or "Selected from recent search activity",
            ],
            "narration": intro_narration,
            "project_index": None,
            "photo_urls": _cover_photos(projects, 1),
        }
    )

    for i, project in enumerate(projects):
        narration = _project_narration(project, i + 1, len(projects))
        slides.append(
            {
                "kind": "project",
                "title": project.get("name") or "Project",
                "subtitle": project.get("community"),
                "lines": _project_caption_lines(project),
                "narration": narration,
                "project_index": i,
                "photo_urls": list(project.get("photos") or [])[:4],
            }
        )

    outro_narration = _outro_narration(projects, communities)
    slides.append(
        {
            "kind": "outro",
            "title": "Summary",
            "subtitle": f"{len(projects)} projects compared",
            "lines": [
                f"{len(projects)} projects with recent sales and rents",
                "Ask for a viewing or a deeper comparison",
            ],
            "narration": outro_narration,
            "project_index": None,
            "photo_urls": _cover_photos(projects, 2),
        }
    )

    full_text = " ".join(s["narration"].strip() for s in slides if s.get("narration"))
    full_text = " ".join(full_text.split())
    words = len(full_text.split())
    estimated = round(words / WORDS_PER_SECOND, 1)

    return {
        "ok": True,
        "reason": None,
        "full_text": full_text,
        "word_count": words,
        "estimated_seconds": estimated,
        "slides": slides,
    }


def _intro_narration(
    communities: list[str],
    bedrooms: int | None,
    project_count: int,
) -> str:
    place = _join_names(communities) or "Dubai"
    bed = _bedroom_spoken(bedrooms)
    bed_bit = f", focused on {bed}," if bed else ""
    return (
        f"Here are {project_count} projects in {place}{bed_bit} "
        f"with recent sales, rents, and handover facts from current market records."
    )


def _project_narration(project: dict[str, Any], index: int, total: int) -> str:
    name = project.get("name") or "This project"
    community = project.get("community") or "its community"
    sale = project.get("recent_sale") or {}
    rent = project.get("rent") or {}
    handover = project.get("handover")

    # Keep each project short so 3–4 projects fit in ~60 seconds.
    opener = f"Next, {name} in {community}."
    if index == 1:
        opener = f"First, {name} in {community}."
    elif index == total:
        opener = f"Finally, {name} in {community}."

    sale_bit = (
        f" Latest sale: {_money(sale.get('price'))}"
        f"{_bed_phrase(sale.get('bedrooms'))}"
        f"{_on_date(sale.get('date'))}."
    )
    rent_bit = (
        f" Latest rent: {_money(rent.get('annual'))} a year"
        f"{_bed_phrase(rent.get('bedrooms'))}"
        f"{_on_date(rent.get('date'))}."
    )
    hand_bit = _handover_spoken(handover)
    return f"{opener}{sale_bit}{rent_bit}{hand_bit}"


def _outro_narration(projects: list[dict[str, Any]], communities: list[str]) -> str:
    place = _join_names(communities) or "these areas"
    return (
        f"Altogether, these {len(projects)} projects in {place} "
        f"show recent sale and rent activity with a clear handover or ready status — "
        f"a solid base to compare and choose what to view next."
    )


def _project_caption_lines(project: dict[str, Any]) -> list[str]:
    sale = project.get("recent_sale") or {}
    rent = project.get("rent") or {}
    lines = [
        project.get("name") or "Project",
        project.get("community") or "",
    ]
    if sale.get("price") is not None:
        lines.append(f"Sale {_money(sale.get('price'))}{_on_date_short(sale.get('date'))}")
    if rent.get("annual") is not None:
        lines.append(f"Rent {_money(rent.get('annual'))}/yr{_on_date_short(rent.get('date'))}")
    if project.get("handover"):
        hand = project["handover"]
        lines.append("Ready to move in" if hand == "ready" else f"Handover {hand}")
    return [line for line in lines if line]


def _handover_spoken(handover: Any) -> str:
    if not handover:
        return ""
    text = str(handover).strip()
    if text.lower() == "ready":
        return " It is ready to move in."
    return f" Handover is {text}."


def _bedroom_line(bedrooms: int | None) -> str | None:
    spoken = _bedroom_spoken(bedrooms)
    return f"Focus: {spoken}" if spoken else None


def _bedroom_spoken(bedrooms: int | None) -> str | None:
    if bedrooms is None:
        return None
    try:
        n = int(bedrooms)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return "studios"
    if n == 1:
        return "1-bedroom homes"
    return f"{n}-bedroom homes"


def _bed_phrase(bedrooms: Any) -> str:
    if bedrooms is None:
        return ""
    try:
        n = float(bedrooms)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return " for a studio"
    if int(n) == 1:
        return " for a 1-bedroom"
    return f" for a {int(n)}-bedroom"


def _on_date(value: Any) -> str:
    label = _date_spoken(value)
    return f" ({label})" if label else ""


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
        # AED 3.1M style for large figures
        m = n / 1_000_000
        text = f"{m:.1f}".rstrip("0").rstrip(".")
        return f"AED {text} million"
    return f"AED {int(round(n)):,}"


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
