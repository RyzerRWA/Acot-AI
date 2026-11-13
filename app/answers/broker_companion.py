"""Broker Ask ACOT — friendly companion.

Nest only sends structured data (`profile.companion`: CRM + claim list + page).
This module reads the question, thinks, and answers in a warm companion voice.
Same idea as `spoken.py`: phrasing lives here, not in Nest.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

BROKER_STARTER_CHIPS = (
    "Who should I follow up with today?",
    "Summarise my pipeline this week",
    "What are the best projects to claim?",
)

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _clip(value: Any, n: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text[:n] if text else None


def _parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _suggestions(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        text = _clip(item, 90)
        if text and text not in out:
            out.append(text)
        if len(out) >= 3:
            break
    return out or list(BROKER_STARTER_CHIPS)


async def answer_companion(
    *,
    question: str,
    companion: dict,
    llm,
    agent_name: str | None = None,
    today: str | None = None,
) -> dict:
    """One path for every broker question: validate against data, answer as a friend."""

    who = agent_name or companion.get("agent_name") or "there"
    day = today or companion.get("today") or "today"
    crm = companion.get("crm") or {}
    claims = companion.get("claimProjects") or companion.get("projects") or []
    index = {
        str(row.get("leadId")): row
        for row in (companion.get("index") or [])
        if isinstance(row, dict) and row.get("leadId")
    }
    history = companion.get("history") or ""
    page_note = companion.get("page_note") or ""

    prompt = (
        f"You are ACOT, a friendly CRM companion for a Dubai real-estate broker"
        f"{f' named {who}' if who and who != 'there' else ''}. "
        f"Speak warmly and clearly, like a helpful colleague — not a robot. Today is {day} (Dubai).\n\n"
        "You have the broker's CRM snapshot and ranked unclaimed projects (claim fits) below. "
        "Read the QUESTION carefully. Decide what they actually want "
        "(follow-ups; pipeline; intros accepted/waiting; "
        "best vs worst projects to claim; draft a message; or Dubai market data).\n\n"
        "Rules:\n"
        "- Use ONLY facts in the data. Never invent leads, scores, demand, prices or names.\n"
        "- If they ask for the WORST / weakest / avoid claims, pick the LOWEST scores and say why they are weak — "
        "do not praise the top fits.\n"
        "- If they ask for the BEST claims, pick the HIGHEST scores.\n"
        "- If it is about Dubai market (areas, prices, projects, rents) and not their CRM, set market true.\n"
        "- Be concise: 2–5 short sentences. No markdown tables.\n"
        "- Nothing is sent to a buyer unless they press Send on a draft.\n\n"
        "Reply with ONLY JSON:\n"
        "{"
        '"market": false, '
        '"source": "crm" or "claim", '
        '"answer": "friendly plain text", '
        '"leads": [{"leadId": "...", "why": "..."}] (max 5, only ids from the data), '
        '"draft": null or {"leadId": "...", "text": "message the broker could send"}, '
        '"claimKeys": ["exact key field from claimProjects", ...] (0–5 when answering about claims; '
        "use the key value, not the project name; order worst-first or best-first), "
        '"suggestions": ["short follow-up", "short follow-up"]'
        "}\n"
        'Set "market": true and leave leads/draft/claimKeys empty when they need Dubai market data instead.\n\n'
        + (f"EARLIER IN THIS CHAT:\n{history}\n\n" if history else "")
        + (f"PAGE CONTEXT: {page_note}\n\n" if page_note else "")
        + f"CRM:\n{json.dumps(crm, default=str)}\n\n"
        + f"CLAIM PROJECTS (scored, usually best-first):\n{json.dumps(claims, default=str)}\n\n"
        + f"QUESTION: {question}"
    )

    raw = await llm.generate(prompt, max_output_tokens=800)
    parsed = _parse_json(raw) or {}

    if parsed.get("market") is True:
        return {
            "answer": "",
            "leads": [],
            "draft": None,
            "suggestions": _suggestions(parsed.get("suggestions")),
            "source": "market",
            "market": True,
            "claimCandidates": [],
        }

    leads_out = []
    for item in parsed.get("leads") or []:
        if not isinstance(item, dict):
            continue
        lead = index.get(str(item.get("leadId")))
        if not lead:
            continue
        leads_out.append(
            {
                "leadId": lead["leadId"],
                "leadType": lead.get("leadType") or "engine",
                "name": lead.get("name") or "A buyer",
                "why": _clip(item.get("why"), 200) or "",
            }
        )
        if len(leads_out) >= 5:
            break

    draft = None
    draft_raw = parsed.get("draft")
    if isinstance(draft_raw, dict) and draft_raw.get("leadId"):
        lead = index.get(str(draft_raw.get("leadId")))
        text = _clip(draft_raw.get("text"), 800)
        if lead and text:
            draft = {
                "leadId": lead["leadId"],
                "leadType": lead.get("leadType") or "engine",
                "name": lead.get("name") or "A buyer",
                "text": text,
            }

    by_key = {
        str(p.get("key")): p
        for p in claims
        if isinstance(p, dict) and p.get("key")
    }
    by_name = {
        str(p.get("name") or "").strip().lower(): p
        for p in claims
        if isinstance(p, dict) and p.get("key") and p.get("name")
    }

    def _resolve_claim(token: Any) -> dict | None:
        raw = str(token or "").strip()
        if not raw:
            return None
        if raw in by_key:
            return by_key[raw]
        hit = by_name.get(raw.lower())
        if hit:
            return hit
        # Partial name match when the model shortens the title.
        needle = raw.lower()
        for name, row in by_name.items():
            if needle in name or name in needle:
                return row
        return None

    claim_candidates: list[dict] = []
    seen_keys: set[str] = set()
    for token in parsed.get("claimKeys") or []:
        row = _resolve_claim(token)
        if not row:
            continue
        key = str(row.get("key"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        claim_candidates.append(row)
        if len(claim_candidates) >= 5:
            break

    source = str(parsed.get("source") or "crm").lower()
    if source not in ("crm", "claim"):
        source = "claim" if claim_candidates else "crm"
    # Answer named projects but forgot claimKeys — still attach matching cards.
    if not claim_candidates and source == "claim":
        answer_l = str(parsed.get("answer") or "").lower()
        for name, row in by_name.items():
            if name and name in answer_l:
                key = str(row.get("key"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                claim_candidates.append(row)
                if len(claim_candidates) >= 5:
                    break
    if claim_candidates and source == "crm":
        source = "claim"

    answer = _clip(parsed.get("answer"), 2000) or (
        f"I'm here, {who}. Ask about follow-ups, your pipeline, what to claim, or the Dubai market."
    )

    return {
        "answer": answer,
        "leads": leads_out,
        "draft": draft,
        "suggestions": _suggestions(parsed.get("suggestions")),
        "source": source,
        "market": False,
        "claimCandidates": claim_candidates,
    }


async def prepare_companion_turn(question: str, components: dict, companion: dict) -> dict:
    """Prepared turn for Nest-supplied companion data."""
    ask_started = time.perf_counter()
    llm = components["rag_chain"].llm
    result = await answer_companion(
        question=question,
        companion=companion if isinstance(companion, dict) else {},
        llm=llm,
        agent_name=(companion or {}).get("agent_name") if isinstance(companion, dict) else None,
        today=(companion or {}).get("today") if isinstance(companion, dict) else None,
    )

    empty = {
        "community": [],
        "projects": [],
        "sub_communities": [],
        "listings": [],
        "brokers": [],
        "filter_no_match": False,
        "filter_evidence_projects": [],
    }

    return {
        "question": question,
        "standalone_question": question,
        "question_type": "general",
        "structured_summary": empty,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "general",
            "structured_data": {
                "community": [],
                "projects": [],
                "sub_communities": [],
                "listings": [],
            },
            "documents": [],
        },
        "rag_chain": components["rag_chain"],
        "conversation_memory": components["conversation_memory"],
        "use_fast": True,
        "fast_answer": result.get("answer") or "",
        "context": None,
        "timing_ms": {},
        "ask_started": ask_started,
        "small_talk": None,
        "broker_companion": result,
        "audience": components.get("audience") or "broker",
    }
