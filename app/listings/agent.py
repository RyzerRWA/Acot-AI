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
from app.graph.graph import build_agent_graph
from app.listings.prompt import SYSTEM_PROMPT, TOOLS, tools_for
from app.listings.tools import ToolInputError, mentions_known_name, run_tool

log = logging.getLogger("acot.listings")

MAX_ROUNDS = 5
MAX_TOOL_JSON = 9_000  # characters of tool output handed back to the model
HISTORY_TURNS = 6      # earlier messages sent along with a question
HISTORY_CHARS = 1_200  # longest earlier message sent in full
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
            if os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("1", "true", "yes", "on") \
                    and os.getenv("LANGSMITH_API_KEY"):
                # Optional: every model call, tool call and graph step shows up in LangSmith with timing and tokens.
                try:
                    from langsmith.wrappers import wrap_openai
                    client = wrap_openai(client)
                except Exception as exc:
                    log.warning("LangSmith tracing is not available: %s", exc)
        self.client = client
        primary = os.getenv("AICREDITS_MODEL", "openai/gpt-4o-mini")
        fallbacks = [m.strip() for m in os.getenv("AICREDITS_FALLBACK_MODELS", "").split(",") if m.strip()]
        self.models = [primary, *fallbacks]

    # one model call, trying the fallback models if the first one fails
    def _chat(self, messages, tools=None):
        last = "no model answered"
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, tools=tools or TOOLS, tool_choice="auto",
                    temperature=0.2, max_tokens=600)
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
        for turn in (history or [])[-HISTORY_TURNS:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                text = str(turn["content"])
                if len(text) > HISTORY_CHARS:
                    # keep the start and the end: the end carries the "[internal note" with the ids on screen
                    text = text[:HISTORY_CHARS * 2 // 3] + " ... " + text[-HISTORY_CHARS // 3:]
                messages.append({"role": turn["role"], "content": text})
        messages.append({"role": "user", "content": question})

        tools = tools_for(question)
        graph = build_agent_graph(
            lambda msgs: self._chat(msgs, tools),
            lambda name, args: run_tool(name, args, pgvector=self.pgvector, profile=profile),
            ToolInputError, max_rounds=MAX_ROUNDS, max_tool_json=MAX_TOOL_JSON, log=log,
            needs_grounding=mentions_known_name)
        state = graph.invoke({"messages": messages, "seen": {}, "used": [], "tokens": 0, "rounds": 0})
        return self._finish(state.get("text", ""), state.get("seen", {}), state.get("used", []),
                            state.get("model", ""), state.get("tokens", 0), started)

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
