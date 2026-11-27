"""Fill catalog gaps from the web: official photos and amenities for one project.

Uses the OpenAI-compatible AI Credits endpoint (OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL)
with its built-in web_search tool to find the developer's official page, then reads that page
for images. Listing portals are never used. Results are cached per project, so each project
costs one search at most.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

log = logging.getLogger(__name__)

# Portals whose terms forbid scraping and whose photos are licensed.
BLOCKED = ("bayut.", "propertyfinder.", "dubizzle.", "houza.", "zillow.", "airbnb.", "booking.com")
MIN_SIZE = (900, 500)
UA = {"User-Agent": "Mozilla/5.0 (compatible; ACOT-video/1.0)"}

PROMPT = """Find the official developer page or official brochure for this Dubai real-estate project, using web search.
Project: "{name}". Community: "{community}". Developer: "{developer}".
Only use the developer's own website, its official project microsite or official brochure.
Never use Bayut, Property Finder, Dubizzle or any listing portal.
Reply with JSON only, no prose, in this shape:
{{"official_url": "https://...", "amenities": ["..."], "image_urls": ["https://..."]}}
amenities: up to 8 short items exactly as the official source lists them.
image_urls: up to 12 direct image file URLs of this exact project from the official source: the building outside,
and inside the homes (bedroom, living room, kitchen, bathroom) wherever the source shows them.
Use null or [] when you are not sure. Do not guess."""


def web_enabled() -> bool:
    off = (os.getenv("VIDEO_WEB_COLLECT") or "true").strip().lower() in {"0", "false", "no", "off"}
    return not off and bool((os.getenv("OPENAI_API_KEY") or "").strip())


def collect(key: str, name: str, community: str = "", developer: str = "", *, cache_dir: Path | None = None) -> dict:
    """Return {"official_url", "amenities", "photos"}; never raises, empty on failure."""
    empty = {"official_url": None, "amenities": [], "photos": []}
    if not web_enabled() or not name:
        return empty
    from app.Viedo_generation import storage

    cache = (cache_dir or storage.DEFAULT_OUT_DIR / "web") / f"{hashlib.sha256(f'v2|{key}'.encode()).hexdigest()[:24]}.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    try:
        found = _search(name, community, developer)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("Web lookup failed for %s: %s", name, exc)
        return empty
    official = found.get("official_url") if _allowed(found.get("official_url")) else None
    candidates = [u for u in (found.get("image_urls") or []) if _allowed(u)]
    if official:
        candidates += _page_images(official)
    result = {
        "official_url": official,
        # Amenities only count when they come with an official source.
        "amenities": [str(a).strip() for a in (found.get("amenities") or []) if str(a).strip()][:8] if official else [],
        "photos": _usable(list(dict.fromkeys(candidates)))[:12],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def _search(name: str, community: str, developer: str) -> dict:
    text = _ask(PROMPT.format(name=name, community=community or "Dubai", developer=developer or "unknown"), web=True)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON in the web lookup reply.")
    data = json.loads(match.group(0))
    return data if isinstance(data, dict) else {}


def _ask(content: str | list, *, web: bool = False) -> str:
    base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    body: dict = {
        "model": os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
        "input": content if isinstance(content, str) else [{"role": "user", "content": content}],
    }
    if web:
        body["tools"] = [{"type": "web_search"}]
    response = httpx.post(
        f"{base}/responses",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '').strip()}"},
        json=body,
        timeout=120,
    )
    response.raise_for_status()
    return "".join(
        part.get("text", "")
        for item in response.json().get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") in ("output_text", "text")
    )


ROOMS = ("bedroom", "living room", "kitchen", "bathroom", "dining", "balcony")
AMENITY_NAMES = (
    "Swimming pool", "Gym", "Children's play area", "Landscaped garden",
    "Walking path", "Sports court", "Lobby", "Barbecue area",
)
LABEL_PROMPT = (
    "Label each image in order. Reply with JSON only: "
    '[{"i": 0, "kind": "exterior|interior|amenity|other", "room": "bedroom|living room|kitchen|bathroom|dining|balcony|null", "amenity": null, "focus_x": 0.5}]. '
    "interior = inside a home. amenity = a clearly visible shared facility. "
    f"For amenity use exactly one of {json.dumps(AMENITY_NAMES)} only when that facility is clearly the subject. "
    "For a named amenity, report its visual center horizontally as focus_x from 0 (left) to 1 (right). "
    "Use null for uncertain labels or ordinary exterior views. "
    "other = floor plan, map, logo, text graphic or a close-up of people."
)


def label_photos(urls: list[str], *, cache_dir: Path | None = None) -> list[dict]:
    """[{"url", "kind", "room"}] in the same order, using one vision call (cached). [] if unavailable."""
    if not urls or not web_enabled():
        return []
    from app.Viedo_generation import storage

    urls = urls[:16]
    cache_key = hashlib.sha256("|".join([LABEL_PROMPT, *urls]).encode()).hexdigest()[:24]
    cache = (cache_dir or storage.DEFAULT_OUT_DIR / "web") / f"labels_{cache_key}.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    try:
        text = _ask([{"type": "input_text", "text": LABEL_PROMPT}]
                    + [{"type": "input_image", "image_url": u, "detail": "low"} for u in urls])
        match = re.search(r"\[.*\]", text, re.S)
        rows = json.loads(match.group(0)) if match else []
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Photo labelling failed: %s", exc)
        return []
    by_index = {r.get("i"): r for r in rows if isinstance(r, dict)}
    result = []
    for i, url in enumerate(urls):
        row = by_index.get(i) or {}
        kind = row.get("kind") if row.get("kind") in ("exterior", "interior", "amenity", "other") else "exterior"
        room = row.get("room") if row.get("room") in ROOMS else None
        amenity = row.get("amenity") if kind == "amenity" and row.get("amenity") in AMENITY_NAMES else None
        focus = row.get("focus_x")
        focus = focus if amenity and isinstance(focus, (int, float)) and 0 <= focus <= 1 else None
        result.append({"url": url, "kind": kind, "room": room, "amenity": amenity, "focus_x": focus})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def _allowed(url: object) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and bool(host) and not any(b in host for b in BLOCKED)


def _page_images(url: str) -> list[str]:
    """Large images the official page itself shows (og:image first)."""
    try:
        html = httpx.get(url, headers=UA, timeout=20, follow_redirects=True).text[:3_000_000]
    except httpx.HTTPError:
        return []
    found = re.findall(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', html, re.I)
    found += re.findall(r'(?:src|data-src|srcset)=["\']([^"\'\s,]+\.(?:jpe?g|png|webp))', html, re.I)
    urls = [urljoin(url, u) for u in found]
    return [u for u in urls if _allowed(u) and not re.search(r"logo|icon|sprite|favicon|avatar", u, re.I)][:24]


def _usable(urls: list[str]) -> list[str]:
    """Keep only real, large images (checks the bytes, not the model's word)."""
    from PIL import Image

    def check(url: str) -> str | None:
        try:
            r = httpx.get(url, headers=UA, timeout=20, follow_redirects=True)
            if not r.is_success or not r.headers.get("content-type", "").startswith("image/") or len(r.content) > 15_000_000:
                return None
            with Image.open(BytesIO(r.content)) as im:
                return url if im.width >= MIN_SIZE[0] and im.height >= MIN_SIZE[1] else None
        except (httpx.HTTPError, OSError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [u for u in pool.map(check, urls) if u]
