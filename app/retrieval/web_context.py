"""Web answers (OpenAI web search) for what our records do not hold: area profiles, local life, rules and fees.

Our database holds prices and rents, not what a community feels like. One web search (~12s, ~13k tokens plus
the per-search fee) is cached on disk for 7 days per area, so every later ask about that area is instant and
gives the same text. Prices are kept out of the web text: figures come from our DLD records only, so the two
never disagree. Any failure returns None and the caller answers from the database alone.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "web_profile_cache.json"
_TTL_SECONDS = 7 * 86400
_INLINE_CITE = re.compile(r"\s*\(\[[^\]]*\]\((https?://[^)\s]+)\)\)")

_PROMPT = (
    "Search the web and write a short profile of {place}, Dubai, for someone deciding whether to live or invest "
    "there. Cover: what it is like to live in, who it suits, transport and metro access, schools and amenities "
    "nearby, and the main downsides. 4-6 short bullet points starting with a bold label, under 130 words. "
    "Do not quote any prices, rents, yields, sales numbers or growth figures. If {place} is not a real Dubai "
    "community, reply only: UNKNOWN"
)


def _load() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


_CACHE: dict[str, list] = _load()


def _save() -> None:
    try:
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(_CACHE)))
        tmp.replace(_CACHE_FILE)
    except (OSError, RuntimeError) as exc:
        print(f"Web profile cache not saved: {exc}")


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    query = "&".join(p for p in parts.query.split("&") if p and not p.startswith("utm_"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _source_title(title: str | None, url: str) -> str:
    """A PDF's "title" is the first lines of its text ("DUBAI SUBSCRIBE 41°C ..."): show the site instead."""
    title = str(title or "").strip()
    if not title or "\n" in title or urlsplit(url).path.lower().endswith(".pdf"):
        return urlsplit(url).netloc.removeprefix("www.")
    return title[:90]


def _parse(response) -> dict | None:
    text = (response.output_text or "").strip()
    if not text or text.upper().startswith("UNKNOWN"):
        return None
    sources: dict[str, str] = {}
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            for note in getattr(part, "annotations", None) or []:
                if getattr(note, "type", "") == "url_citation":
                    sources.setdefault(_clean_url(note.url), _source_title(note.title, note.url))
    # Inline "([site](url))" links clutter each bullet; the sources are listed once under the answer.
    text = _INLINE_CITE.sub("", text).strip()
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)\s]+\)", r"\1", text)  # any other link: keep its words only
    return {"text": text, "sources": [[title, url] for url, title in sources.items()][:4]}


_ANSWER_PROMPT = (
    "You are ACOT, a Dubai real estate assistant. Search the web and answer this question{near}: {question}\n"
    "Answer first in one direct sentence, then up to 5 short bullets with real names and specifics. Under 140 "
    "words. For rules, fees, visas and laws prefer official sources (Dubai Land Department, u.ae, ICP, RTA, "
    "KHDA) and note that rules can change. Do not quote property prices, rents, yields or sales numbers. "
    "If the question is not about Dubai or living or owning property there, reply only: UNKNOWN"
)


def _search(key: str, prompt: str, ttl: float) -> dict | None:
    """One cached web-search answer; the last good one is reused if the search fails."""
    if not OPENAI_API_KEY or "openai.com" not in str(OPENAI_BASE_URL):
        return None  # web search is an OpenAI Responses tool; other gateways do not offer it
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=45, max_retries=1)
        response = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
            max_output_tokens=900,
            input=prompt,
        )
        result = _parse(response)
    except Exception as exc:
        print(f"Web search failed for {key}: {exc}")
        return hit[1] if hit else None
    _CACHE[key] = [time.time(), result]
    _save()
    return result


def area_profile(place: str, key: str | None = None) -> dict | None:
    """{"text": markdown bullets, "sources": [[title, url], ...]} for one Dubai area, or None."""
    key = str(key or place or "").strip().upper()
    return _search(key, _PROMPT.format(place=place), _TTL_SECONDS) if key else None


def web_answer(question: str, place: str | None = None) -> dict | None:
    """A question our records cannot answer ("best restaurant near JVC", golden visa rules), from the web.
    Cached a day per wording, so asking again gives the same answer."""
    text = re.sub(r"[^a-z0-9 ]+", "", str(question or "").lower()).strip()
    if not text:
        return None
    near = f" (the user is asking about {place}, Dubai)" if place and str(place).lower() != "dubai" else ""
    key = f"Q:{' '.join(text.split())}|{str(place or '').upper()}"
    return _search(key, _ANSWER_PROMPT.format(question=question, near=near), 86400)
