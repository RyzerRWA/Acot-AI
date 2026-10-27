"""
The listings agent: the model chooses read-only tools, the tools query PostgreSQL, the model explains the result.

Numbers always come from the database. Listing cards are drawn by the app from [[id]] tags, so the model never
writes a URL or a photo link (an early test showed it invents them), and any link it does write is stripped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import AICREDITS_API_KEY, AICREDITS_BASE_URL
from app.listings.prompt import SYSTEM_PROMPT, TOOLS
from app.listings.tools import ToolInputError, run_tool

log = logging.getLogger("acot.listings")

MAX_ROUNDS = 5
MAX_TOOL_JSON = 9_000  # characters of tool output handed back to the model
_ID_TAG = re.compile(r"\[\[(\d{1,12})\]\]")


class ListingsUnavailable(RuntimeError):
    """The language model could not be reached."""


@dataclass
class AgentResult:
    answer: str
    cards: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    model: str = ""
    tokens: int = 0
    ms: int = 0


def strip_links(text: str) -> str:
    """The model must never show URLs or images (it would invent them); the app draws cards from [[id]] tags."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)               # markdown images
    text = re.sub(r"\[([^\]]+)\]\((?:https?:)?//[^)]*\)", r"\1", text)  # [label](http://...) keeps the label
    text = re.sub(r"\[([^\]]+)\]\(/[^)]*\)", r"\1", text)            # [label](/path)
    text = re.sub(r"https?://\S+", "", text)                          # bare URLs
    return re.sub(r"\n{3,}", "\n\n", text)


class ListingAgent:
    def __init__(self, client: OpenAI | None = None, pgvector=None):
        # pgvector: the shared brochure retriever, so the agent does not load a second embedding model
        self.pgvector = pgvector
        if client is None:
            if not AICREDITS_API_KEY:
                raise ValueError("AICREDITS_API_KEY is not configured.")
            client = OpenAI(base_url=AICREDITS_BASE_URL, api_key=AICREDITS_API_KEY, timeout=45.0)
        self.client = client
        primary = os.getenv("AICREDITS_MODEL", "openai/gpt-4o-mini")
        fallbacks = [m.strip() for m in os.getenv("AICREDITS_FALLBACK_MODELS", "").split(",") if m.strip()]
        self.models = [primary, *fallbacks]

    # one model call, trying the fallback models if the first one fails
    def _chat(self, messages):
        last = "no model answered"
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, tools=TOOLS, tool_choice="auto",
                    temperature=0.2, max_tokens=900)
                return model, response
            except Exception as exc:  # network, rate limit, provider error
                last = f"{model}: {exc}"
                if any(code in str(exc) for code in ("401", "403")):
                    break
        raise ListingsUnavailable(last)

    def answer(self, question: str, history: list[dict] | None = None, listing_id: str | None = None,
               profile: dict | None = None) -> AgentResult:
        """profile: what is known about the user. The update_profile tool writes into this same dict."""
        started = time.perf_counter()
        profile = {} if profile is None else profile
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if profile:
            messages.append({"role": "system", "content":
                             "What is known about this user (from their own earlier messages or the app). Use it to "
                             "personalise, but never present it as something they just said, and if their question "
                             "conflicts with it, the question wins: " + json.dumps(profile, default=str)})
        if listing_id:
            messages.append({"role": "system", "content":
                             f'The user is looking at listing {listing_id}. "this one" or "is this fair" means that '
                             "listing: call get_listing with this id."})
        for turn in (history or [])[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:2000]})
        messages.append({"role": "user", "content": question})

        seen: dict[str, dict] = {}   # every listing a tool actually returned this turn
        used: list[dict] = []
        tokens, model = 0, ""

        for _ in range(MAX_ROUNDS):
            model, response = self._chat(messages)
            tokens += getattr(response.usage, "total_tokens", 0) or 0
            message = response.choices[0].message

            if not message.tool_calls:
                return self._finish(message.content or "", seen, used, model, tokens, started)

            messages.append({
                "role": "assistant", "content": message.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in message.tool_calls],
            })
            for call in message.tool_calls:
                try:
                    result = run_tool(call.function.name, json.loads(call.function.arguments or "{}"),
                                      pgvector=self.pgvector, profile=profile)
                    output = result.output
                    used.append({"name": call.function.name, "ok": True})
                    for card in result.cards:
                        seen[card["listing_id"]] = card
                except (ToolInputError, json.JSONDecodeError) as exc:
                    # bad arguments go back to the model so it can correct itself
                    used.append({"name": call.function.name, "ok": False})
                    output = {"error": str(exc)}
                except Exception:
                    log.exception("tool %s failed", call.function.name)
                    used.append({"name": call.function.name, "ok": False})
                    output = {"error": "That lookup failed."}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(output, default=str)[:MAX_TOOL_JSON]})

        return self._finish("I couldn't pin that down. Could you rephrase it or narrow it to an area or a budget?",
                            seen, used, model, tokens, started)

    @staticmethod
    def _finish(text, seen, used, model, tokens, started) -> AgentResult:
        cleaned = strip_links(text)
        cited = list(dict.fromkeys(_ID_TAG.findall(cleaned)))
        # A card is shown for every id the model tagged. If it described a project or listing by name
        # without tagging it, the card is still shown (only items a tool returned, only if named).
        lowered = cleaned.lower()
        for key, card in seen.items():
            row = card.get("row") or {}
            name = str(row.get("name") or card.get("title") or "").strip().lower()
            if key not in cited and len(name) >= 6 and name in lowered:
                cited.append(key)
        cards = [seen[i] for i in cited if i in seen][:10]   # only ids a tool really returned
        answer = _ID_TAG.sub(lambda m: m.group(0) if m.group(1) in seen else "", cleaned).strip()
        ms = int(round((time.perf_counter() - started) * 1000))
        log.info("listings: %s | %s tokens | %sms | %s",
                 ",".join(u["name"] + ("" if u["ok"] else "!") for u in used) or "no tools", tokens, ms, model)
        return AgentResult(answer=answer, cards=cards, tools=used, model=model, tokens=tokens, ms=ms)
