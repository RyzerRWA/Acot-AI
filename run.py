import asyncio
from collections import Counter
import contextvars
import os
import re
import io
import time
import json
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from dotenv import load_dotenv

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

from app.listings.tools import _find_projects, top_areas_by_sales
from app.listings.agent import ListingAgent, ListingsUnavailable, _ID_TAG as _AGENT_TAG
from app.listings.router import is_rent_question
from app.llm.client import LLMQueueTimeoutError
from app.answers.spoken import MAX_FACT_CHARS, start_spoken_reply
from app.subscription.token_usage import add_tokens
from app.listings import session_store
from app.retrieval.hybrid.listing_turns import should_use_structured_listing_comparison


# =========================================================
# POSTGRES STRUCTURED RETRIEVER
# =========================================================

from app.retrieval.postgres.structured_retriever import (
    StructuredRetriever
)


# =========================================================
# POSTGRES PGVECTOR RETRIEVER
# =========================================================

from app.retrieval.postgres.pgvector_retriever import (
    PGVectorRetriever
)


# =========================================================
# QUERY PLANNER
# =========================================================

from app.retrieval.hybrid.query_planner import (
    QueryPlanner
)


# =========================================================
# ENTITY RESOLVER
# =========================================================

from app.retrieval.hybrid.entity_resolver import (
    EntityResolver
)
from app.retrieval.hybrid.question_text import (
    asks_for_listings,
    collapse_stretched_letters,
    is_group_attribute_question,
    prepare_user_question,
)


# =========================================================
# HYBRID RETRIEVER
# =========================================================

from app.retrieval.hybrid.hybrid_retriever import (
    HybridRetriever
)


# =========================================================
# CONTEXT BUILDER
# =========================================================

from app.retrieval.hybrid.context_builder import (
    HybridContextBuilder
)


# =========================================================
# INVESTMENT ANALYZER
# =========================================================

from app.intelligence.investment_analyzer import (
    InvestmentAnalyzer
)


# =========================================================
# RAG CHAIN
# =========================================================

from app.rag.chains.hybrid_rag_chain import (
    HybridRAGChain
)


# =========================================================
# CONVERSATIONAL MEMORY
# =========================================================

from app.memory.conversation_memory import (
    ConversationMemory
)


# =========================================================
# INITIALIZE ACOT
# =========================================================

def initialize_acot():

    structured_retriever = StructuredRetriever()

    query_planner = QueryPlanner()

    entity_resolver = EntityResolver(
        structured_retriever
    )

    pgvector_retriever = PGVectorRetriever(
        match_threshold=0.5,
        match_count=5
    )

    hybrid_retriever = HybridRetriever(
        structured_retriever=structured_retriever,
        document_retriever=None,
        pgvector_retriever=pgvector_retriever,
        query_planner=query_planner,
        entity_resolver=entity_resolver
    )

    context_builder = HybridContextBuilder()

    investment_analyzer = InvestmentAnalyzer()

    rag_chain = HybridRAGChain()

    conversation_memory = ConversationMemory(
        llm_client=query_planner._llm_client
    )

    # Rental listings (scraped data in PostgreSQL) have their own agent; if it cannot start,
    # the rest of ACOT still works and rental questions fall back to the normal pipeline.
    try:
        # The agent answers project and rental questions with tools. It shares the brochure retriever.
        listing_agent = ListingAgent(pgvector=pgvector_retriever)
    except Exception as exc:
        print(f"Listings agent unavailable: {exc}")
        listing_agent = None

    return {
        "listing_agent": listing_agent,
        "structured_retriever": structured_retriever,
        "query_planner": query_planner,
        "entity_resolver": entity_resolver,
        "pgvector_retriever": pgvector_retriever,
        "hybrid_retriever": hybrid_retriever,
        "context_builder": context_builder,
        "investment_analyzer": investment_analyzer,
        "rag_chain": rag_chain,
        "conversation_memory": conversation_memory
    }


# =========================================================
# PRINT STRUCTURED POSTGRES DATA
# =========================================================

def print_structured_data(structured_data):

    print("\n================================")
    print("POSTGRES DATA")
    print("================================")


    if not structured_data:

        print("No structured PostgreSQL data.")

        return


    communities = structured_data.get(
        "community",
        []
    )

    projects = structured_data.get(
        "projects",
        []
    )

    sub_communities = structured_data.get(
        "sub_communities",
        []
    )


    print(
        f"\nCommunities found: "
        f"{len(communities)}"
    )

    print(
        f"Projects found: "
        f"{len(projects)}"
    )

    print(
        f"Sub-communities found: "
        f"{len(sub_communities)}"
    )


    # -----------------------------------------------------
    # COMMUNITIES
    # -----------------------------------------------------

    if communities:

        print("\nCOMMUNITIES")

        for index, record in enumerate(
            communities,
            start=1
        ):

            print("\n--------------------------------")
            print(f"Community {index}")
            print("--------------------------------")

            print(
                f"ID: "
                f"{record.get('id')}"
            )

            print(
                f"Name: "
                f"{record.get('name')}"
            )

            print(
                f"City: "
                f"{record.get('city')}"
            )

            print(
                f"Projects: "
                f"{record.get('projects_count')}"
            )

            print(
                f"Pool Projects: "
                f"{record.get('pool_projects_count')}"
            )

            print(
                f"Source: "
                f"{record.get('source')}"
            )


    # -----------------------------------------------------
    # PROJECTS
    # -----------------------------------------------------

    if projects:

        print("\nPROJECTS")

        for index, record in enumerate(
            projects,
            start=1
        ):

            print("\n--------------------------------")
            print(f"Project {index}")
            print("--------------------------------")

            print(
                f"ID: "
                f"{record.get('id')}"
            )

            print(
                f"Name: "
                f"{record.get('name')}"
            )

            print(
                f"Developer: "
                f"{record.get('developer_name')}"
            )

            print(
                f"City: "
                f"{record.get('city')}"
            )

            print(
                f"Community: "
                f"{record.get('community')}"
            )

            print(
                f"Sub-community: "
                f"{record.get('sub_community')}"
            )

            print(
                f"Price: "
                f"{record.get('price')}"
            )

            print(
                f"Bedrooms: "
                f"{record.get('bedroom_min')} - "
                f"{record.get('bedroom_max')}"
            )

            print(
                f"Property Types: "
                f"{record.get('property_types')}"
            )

            print(
                f"Handover: "
                f"{record.get('handover_time')}"
            )

            print(
                f"Brochure URL: "
                f"{record.get('brochure_url')}"
            )

            print(
                f"Property Finder URL: "
                f"{record.get('property_finder_url')}"
            )

            print(
                f"Source: "
                f"{record.get('source')}"
            )

            print(
                f"Fetched At: "
                f"{record.get('fetched_at')}"
            )

            print(
                f"Updated At: "
                f"{record.get('updated_at')}"
            )


    # -----------------------------------------------------
    # SUB-COMMUNITIES
    # -----------------------------------------------------

    if sub_communities:

        print("\nSUB-COMMUNITIES")

        for index, record in enumerate(
            sub_communities,
            start=1
        ):

            print("\n--------------------------------")
            print(f"Sub-community {index}")
            print("--------------------------------")

            print(
                f"ID: "
                f"{record.get('id')}"
            )

            print(
                f"Name: "
                f"{record.get('name')}"
            )

            print(
                f"Community: "
                f"{record.get('community')}"
            )

            print(
                f"City: "
                f"{record.get('city')}"
            )

            print(
                f"Source: "
                f"{record.get('source')}"
            )


# =========================================================
# PRINT PGVECTOR DATA
# =========================================================

def print_pgvector_data(documents):

    print("\n================================")
    print("POSTGRES PGVECTOR DATA")
    print("================================")


    if not documents:

        print("No PostgreSQL pgvector document results.")

        return


    print(
        f"Documents found: "
        f"{len(documents)}"
    )


    for index, document in enumerate(
        documents,
        start=1
    ):

        print("\n--------------------------------")
        print(f"Document Chunk {index}")
        print("--------------------------------")

        print(
            f"Database Row ID: "
            f"{document.get('id')}"
        )

        print(
            f"Document ID: "
            f"{document.get('document_id')}"
        )

        print(
            f"Document Title: "
            f"{document.get('document_title')}"
        )

        print(
            f"Document Type: "
            f"{document.get('document_type')}"
        )

        print(
            f"Publisher: "
            f"{document.get('publisher')}"
        )

        print(
            f"Chunk ID: "
            f"{document.get('chunk_id')}"
        )

        print(
            f"Similarity: "
            f"{document.get('similarity')}"
        )

        print(
            f"Source URL: "
            f"{document.get('document_url')}"
        )

        print("\nCONTENT:")

        print(
            document.get(
                "content",
                ""
            )
        )


# =========================================================
# PRINT EXACT LLM CONTEXT
# =========================================================

def print_llm_context(context):

    print("\n================================")
    print("EXACT CONTEXT SENT TO LLM")
    print("================================")

    if not context:

        print(
            "No context was sent to the LLM."
        )

        return


    print(context)


# =========================================================
# PRINT QUERY PLAN
# =========================================================

def print_query_plan(plan):

    print("\n================================")
    print("QUERY PLAN")
    print("================================")


    if not plan:

        print("No query plan.")

        return


    if hasattr(
        plan,
        "to_dict"
    ):

        plan_data = plan.to_dict()

    elif isinstance(
        plan,
        dict
    ):

        plan_data = plan

    else:

        plan_data = str(plan)


    print(plan_data)


# =========================================================
# PRINT ENTITY RESOLUTION
# =========================================================

def print_entity_resolution(entity):

    print("\n================================")
    print("ENTITY RESOLUTION")
    print("================================")


    if not entity:

        print("No entity resolved.")

        return


    if hasattr(
        entity,
        "to_dict"
    ):

        entity_data = entity.to_dict()

    elif isinstance(
        entity,
        dict
    ):

        entity_data = entity

    else:

        entity_data = str(entity)


    print(entity_data)


# =========================================================
# FRONTEND-READY RESPONSE FORMATTER
# =========================================================

from app.answers.area_comparison import build_area_comparison, looks_like_comparison, market_signals, mentioned_area, official_area, parse_area_comparison
from app.answers.structured_answer import (
    _SMALL_TALK_KINDS,
    _STARTER_CHIPS,
    _build_comparison,
    _build_fast_structured_answer,
    _detect_response_type,
    _is_fast_structured_question,
    _normalize_chat_text,
    build_frontend_response,
)




# A greeting, thanks, or goodbye followed by nothing but filler is still
# small talk: "hey, how are you today?", "thanks a lot", "hello acot".
# Any other word ("hi, show me projects") means a real request, so the
# message goes through the normal pipeline.
_SMALL_TALK_LEADS = (
    (
        re.compile(
            r"^(?:hi+|hey+|hel+o+|hiya|hai|yo|greetings|howdy|sup|namaste|gm|"
            r"good|morning|mornin|afternoon|evening|"
            r"salam|salaam|assalam|assalamu|assalamualaikum|salamualaikum)$"
        ),
        "greeting",
    ),
    (re.compile(r"^(?:thanks?|thx|ty|thank|cheers)$"), "thanks"),
    (re.compile(r"^(?:bye+|goodbye)$"), "bye"),
)

_SMALL_TALK_FILLER = {
    "there", "how", "are", "you", "u", "doing", "today", "is", "it",
    "going", "what's", "whats", "up", "acot", "assistant", "bot",
    "buddy", "friend", "bro", "sir", "madam", "dear", "again", "so",
    "much", "a", "lot", "very", "for", "your", "the", "help", "all",
    "everyone", "team", "good", "morning", "afternoon", "evening",
    "night", "hi", "hello", "hey", "thanks", "thank", "bye", "nice",
    "to", "meet", "see", "later", "take", "care", "and", "i", "am",
    "fine", "well", "great", "mornin", "alaikum", "alaykum", "aleikum",
    "wa", "everybody", "folks", "guys",
}


_TRANSACTION = re.compile(
    r"\b(?:transactions?|transcations?|trancations?|transctions?|sale history|sales history)\b",
    re.I,
)
_MORE_TRANSACTIONS = re.compile(
    r"\b(?:more|another|next|additional|further)\b",
    re.I,
)
_TRANSACTION_PAGE_SIZE = 5
_MORE_TRANSACTION_FOLLOWUP = "Can you show me more transactions"


def _transaction_subject(question, conversation_memory):
    """Projects a transaction follow-up refers to, or None so the agent does not run."""
    if not _TRANSACTION.search(question or ""):
        return None
    items = getattr(conversation_memory, "active_candidates", None) or []
    projects = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    asked = (question or "").lower()
    named = [project for project in projects if str(project["name"]).lower() in asked]
    chosen = named or projects
    return chosen[:8] or None


def _sale_line(sale) -> str:
    parts = []
    if sale.get("price") is not None:
        parts.append(f"AED {int(sale['price']):,}")
    beds = sale.get("bedrooms")
    if beds == 0:
        parts.append("studio")
    elif beds is not None:
        parts.append("1 bedroom" if int(beds) == 1 else f"{int(beds)} bedrooms")
    if sale.get("size_sqft"):
        parts.append(f"{int(sale['size_sqft']):,} sqft")
    detail = ", ".join(parts)
    date = str(sale.get("date") or "")[:10]
    if date:
        return f"• **{date}**: {detail}" if detail else f"• **{date}**"
    return f"• {detail}"


def _format_transactions(name, payload, label=None, more=False) -> str:
    sales = payload.get("sales") or []
    names = [name] if isinstance(name, str) else list(name or [])
    intro = label or (names[0] if len(names) == 1 else "these projects")
    if not sales:
        if more:
            return f"There are no more recorded sales for {intro} in the transaction data."
        return f"There are no recorded sales for {intro} in the transaction data."
    grouped = {}
    for sale in sales:
        key = str(sale.get("project") or "").strip()
        if not key:
            continue
        grouped.setdefault(key.upper(), {"name": key, "sales": []})["sales"].append(sale)
    order = []
    for item in names:
        key = str(item or "").strip().upper()
        if key and key in grouped and key not in order:
            order.append(key)
    for key in grouped:
        if key not in order:
            order.append(key)
    lead = (
        f"Here are more recent transaction details for {intro}:"
        if more else
        f"Here are the recent transaction details for {intro}:"
    )
    lines = [lead, ""]
    for key in order:
        block = grouped[key]
        lines.append(f"**{block['name']}**")
        lines.append("")
        lines.append("**• Recent Transactions:**")
        lines.append("")
        for sale in block["sales"]:
            lines.append(_sale_line(sale))
        lines.append("")
    return "\n".join(lines).rstrip()


def _transaction_turn(question, conversation_memory):
    projects = _transaction_subject(question, conversation_memory)
    if not projects:
        return None
    from app.retrieval.propai_catalog import recent_sales
    names = [project.get("name") for project in projects]
    communities = {
        str(project.get("community") or "").strip()
        for project in projects
        if project.get("community")
    }
    developers = {
        str(project.get("developer_name") or project.get("developer") or "").strip()
        for project in projects
        if project.get("developer_name") or project.get("developer")
    }
    if len(names) == 1:
        intro = names[0]
    elif len(developers) == 1 and len(communities) == 1:
        intro = f"the {next(iter(developers))} projects in {next(iter(communities))}"
    elif len(communities) == 1:
        intro = f"the projects in {next(iter(communities))}"
    else:
        intro = "these projects"
    fingerprint = "|".join(str(name or "").strip().upper() for name in names if name)
    more = bool(_MORE_TRANSACTIONS.search(question or ""))
    prev_fp = getattr(conversation_memory, "transaction_names", None)
    prev_offset = int(getattr(conversation_memory, "transaction_offset", 0) or 0)
    offset = prev_offset if more and prev_fp == fingerprint else 0
    payload = recent_sales(names, per_project=_TRANSACTION_PAGE_SIZE, offset=offset)
    if conversation_memory is not None:
        conversation_memory.transaction_names = fingerprint
        conversation_memory.transaction_offset = (
            offset + _TRANSACTION_PAGE_SIZE if payload.get("sales") else offset
        )
    # Text only: do not attach project rows, or the app draws property cards under the sales list.
    summary = _empty_structured_summary()
    summary["transaction_followup"] = True
    # The sales themselves, so the app can show them by unit type and over time (data.transactions).
    summary["transactions"] = payload.get("sales") or []
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "general",
        "structured_summary": summary,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "general",
            "structured_data": {"community": [], "projects": [], "sub_communities": []},
            "documents": [],
        },
        "use_fast": True,
        # With sales, the app draws them from data.transactions, so the text is one lead line
        # (spoken in voice mode); without sales, the full message explains why.
        "fast_answer": (
            f"Here are the {'next' if more and offset > 0 else 'recent'} sales for {intro}. "
            f"{len(payload['sales'])} sales are listed below."
            if payload.get("sales")
            else _format_transactions(names, payload, label=intro, more=more and offset > 0)
        ),
        "context": None,
        "small_talk": None,
    }


_SHORTLIST_RECOMMEND = re.compile(
    r"\b(?:"
    r"which (?:one |of these |of those )?(?:is|are) (?:the )?best|"
    r"which of these (?:is|are) better|"
    r"which (?:of these |ones? |of those )?(?:should|would) i|"
    r"what should i (?:pick|choose|buy)|"
    r"which property is best|"
    r"best (?:property|project) to (?:invest|buy)|"
    r"recommend (?:one|one of these|the best|among these)|"
    r"best (?:one|option|pick|project)(?:\s+of these)?|"
    r"pick (?:one|for me)|"
    r"which of these (?:do you )?recommend"
    r")\b",
    re.I,
)
_COMPARE_WORD = re.compile(r"\bcompar(?:e|ison|ing)\b", re.I)
_COMPARE_ALL = re.compile(r"\b(?:all|every|each)\b", re.I)
_COMPARE_FIRST_N = re.compile(
    r"\b(?:first|top)\s+(two|three|four|both|2|3|4)\b|\b(?:both|the two)\b",
    re.I,
)
_COMPARE_N = {"two": 2, "three": 3, "four": 4, "both": 2, "2": 2, "3": 3, "4": 4}
_COMPARE_ALL_LIMIT = 12


def _active_projects(conversation_memory) -> list[dict]:
    items = getattr(conversation_memory, "active_candidates", None) or []
    return [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]


def _shown_compare_limit(question: str) -> int | None:
    """How many on-screen projects a compare follow-up asked for, or None."""
    text = question or ""
    if not _COMPARE_WORD.search(text):
        return None
    if _COMPARE_ALL.search(text):
        return _COMPARE_ALL_LIMIT
    match = _COMPARE_FIRST_N.search(text)
    if not match:
        return None
    token = (match.group(1) or "both").lower()
    return _COMPARE_N.get(token, 2)


def _compare_shown_turn(question, conversation_memory):
    """Compare N (or all) on-screen projects in a table + Acot Recommend."""
    limit = _shown_compare_limit(question)
    if not limit:
        return None
    if _TRANSACTION.search(question or ""):
        return None
    available = _active_projects(conversation_memory)
    projects = available[:limit]
    if len(projects) < 2:
        return None
    from app.answers.recommend import format_compare_table, rank_projects, wanted_bedrooms

    ranked = rank_projects(projects, wanted_beds=wanted_bedrooms(question))
    answer = format_compare_table(projects, ranked)
    summary = _empty_structured_summary()
    # Cards show the compared set; memory keeps the fuller shortlist via preserve logic.
    summary["projects"] = projects
    summary["project_total"] = len(projects)
    summary["recommendation"] = ranked.get("recommendation")
    summary["comparison"] = True
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "comparison",
        "structured_summary": summary,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "comparison",
            "structured_data": {
                "community": [],
                "projects": projects,
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _shortlist_recommend_turn(question, conversation_memory):
    """Rank projects already on screen and return Acot Recommend."""
    if _shown_compare_limit(question):
        return None
    if not _SHORTLIST_RECOMMEND.search(question or ""):
        return None
    if _TRANSACTION.search(question or ""):
        return None
    projects = _active_projects(conversation_memory)
    if len(projects) < 2:
        return None
    from app.answers.recommend import format_shortlist, rank_projects, wanted_bedrooms

    selected = projects[:_COMPARE_ALL_LIMIT]
    ranked = rank_projects(selected, wanted_beds=wanted_bedrooms(question))
    # Compare-style "which is best" with many options still gets a table when asked to compare.
    if _COMPARE_WORD.search(question or ""):
        from app.answers.recommend import format_compare_table
        answer = format_compare_table(selected, ranked)
    else:
        answer = format_shortlist(ranked, question=question)
    summary = _empty_structured_summary()
    by_name = {str(item.get("name") or "").lower(): item for item in selected}
    ordered = []
    for item in ranked.get("ranked") or []:
        row = by_name.get(str(item.get("name") or "").lower())
        if row and row not in ordered:
            ordered.append(row)
    summary["projects"] = ordered
    summary["project_total"] = len(ordered)
    summary["recommendation"] = ranked.get("recommendation")
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "comparison",
        "structured_summary": summary,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "comparison",
            "structured_data": {
                "community": [],
                "projects": summary["projects"],
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _listing_question(question, conversation_memory) -> bool:
    """Rental questions stay on the listing path. A place name must not turn them into projects."""
    if asks_for_listings(question):
        return True
    context = {}
    if conversation_memory is not None and hasattr(conversation_memory, "context"):
        try:
            context = conversation_memory.context() or {}
        except Exception:
            context = {}
    return bool(should_use_structured_listing_comparison(question, context))


def _shown_project_followup(question, conversation_memory):
    """Projects already on screen for "their starting prices" and the other chip follow-ups.

    Returns (projects, no_match) or None when this is not that kind of question.
    """
    if not is_group_attribute_question(question):
        return None
    items = getattr(conversation_memory, "active_candidates", None) or []
    projects = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    if not projects:
        return None
    bedrooms = re.search(r"\b(\d+)\s*[- ]?\s*bed", question or "", re.I)
    if not bedrooms:
        return projects, False
    wanted = int(bedrooms.group(1))
    narrowed = []
    for project in projects:
        low = project.get("bedroom_min")
        high = project.get("bedroom_max")
        if low is None:
            continue
        high = low if high is None else high
        if float(low) <= wanted <= float(high):
            narrowed.append(project)
    if narrowed:
        return narrowed, False
    return projects, True


def _classify_small_talk(question):
    """Return a small-talk kind when the whole message is a greeting or identity question."""

    text = _normalize_chat_text(question)
    kind = _SMALL_TALK_KINDS.get(text)
    if kind:
        return kind

    words = re.findall(r"[a-z']+", text)
    if not words or len(words) > 8:
        return None
    for pattern, lead_kind in _SMALL_TALK_LEADS:
        if pattern.match(words[0]):
            if all(word in _SMALL_TALK_FILLER for word in words[1:]):
                return lead_kind
            return None
    return None


def _small_talk_answer(kind, name=None):
    """name: the signed-in user's first name, when the app sent it."""
    who = f" {name}" if name else ""

    if kind == "identity":
        return (
            "I'm ACOT, an AI assistant for Dubai real estate. I answer from "
            "project records and documents, including prices, developers, "
            "bedroom ranges, handover dates, and amenities."
        )

    if kind == "capabilities":
        return (
            "I can help you with Dubai project data:\n\n"
            "- Find projects in a community\n"
            "- List starting prices, bedroom ranges, and handover dates\n"
            "- Compare projects already on screen\n"
            "- Pull amenities from their documents"
        )

    if kind in ("GoodMorning", "Good Morning", "Morning"):
        return (
            f"Good morning{who}, How can I help you with Dubai real estate today?"
        )

    if kind in ("GoodAfternoon", "Good Afternoon", "Afternoon"):
        return (
            f"Good afternoon{who}, How can I help you with Dubai real estate today?"
        )

    if kind in ("GoodEvening", "Good Evening", "Evening"):
        return (
            f"Good evening{who}, How can I help you with Dubai real estate today?"
        )

    if kind == "thanks":
        return (
            f"You're welcome{who}. Ask me about a Dubai community or project "
            "whenever you're ready."
        )

    if kind == "bye":
        return (
            f"Goodbye{who}. Have a great day! I'll be here when you want to look at Dubai projects again."
        )

    return f"Hello{who}, I'm ACOT, a Dubai real estate assistant. How can I help you?"


def _empty_structured_summary():
    return {
        "community": [],
        "projects": [],
        "sub_communities": [],
        "listings": [],
        "filter_no_match": False,
        "filter_evidence_projects": [],
    }


# =========================================================
# PROCESS QUESTION
# =========================================================

# =========================================================
# TOOL-CALLING AGENT TURN
# =========================================================
# The model chooses read-only tools (projects, brochures, rentals) and explains
# the result. It is switched on with ACOT_USE_AGENT=1. If it is off, or it
# fails for any reason, the standard pipeline below answers as before.

def _agent_enabled(components):
    on = os.getenv("ACOT_USE_AGENT", "0").strip().lower() in ("1", "true", "yes", "on")
    return on and components.get("listing_agent") is not None


def _agent_listing_record(card):
    """Rental card from the agent, in the row shape the frontend card builder reads."""
    return {
        "listing_id": card["listing_id"],
        "title": card.get("title"),
        "property_type": card.get("property_type"),
        "price_aed": card.get("annual_rent_aed"),
        "rent_frequency": "yearly",
        "beds": card.get("beds"),
        "baths": card.get("baths"),
        "size_sqft": card.get("size_sqft"),
        "furnishing": card.get("furnishing"),
        "community": card.get("community"),
        "sub_community": card.get("sub_community"),
        "cover_photo": card.get("cover_photo"),
        "url": card.get("path"),
    }


def _collapse_doubled_names(text, names):
    """'Samana Waves "Samana Waves" from AED...' -> 'Samana Waves from AED...' (the model wrote the name after its tag)."""
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        text = re.sub(
            rf'({re.escape(name)})[\s"“”\'*_]+{re.escape(name)}["“”\']?',
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _drop_repeated_name_lines(text, names):
    """Remove lines that only repeat a project name (for example "- **ID:** Samana Waves") after it was already shown."""
    wanted = {name.lower() for name in names if name}
    kept, seen_names = [], set()
    for line in text.split("\n"):
        bare = re.sub(r"^[\s>*\-•\d.)]*(?:\*{0,2}id\*{0,2}\s*:?\s*\*{0,2})?", "", line, flags=re.I).strip(" *_:")
        if bare.lower() in wanted:
            if bare.lower() in seen_names:
                continue  # a stray line that only repeats a name already written
            seen_names.add(bare.lower())
        else:
            seen_names.update(name for name in wanted if name in line.lower())
        kept.append(line)
    return "\n".join(kept)


# Everything in a document request except the project name.
_DOC_WORDS = re.compile(
    r"\b(?:uh+|um+|okay|ok|please|can|could|would|you|i|we|get|send|share|give|show|open|me|us|the|a|an|of|for|"
    r"to|about|its|it|project|brochures?|floor ?plans?|pdfs?|documents?|catalog(?:ue)?s?)\b|['’]s\b|[?.!,]",
    re.I,
)
_INVEST_ASK = re.compile(
    r"\b(?:invest\w*|should i (?:buy|get)|worth (?:it|buying|investing)|good (?:buy|deal)|roi|yields?|returns?|"
    r"appreciat\w*|capital growth|price growth)\b",
    re.I,
)
_DOC_ASK = re.compile(r"\b(?:brochures?|floor ?plans?|pdfs?|documents?|catalog(?:ue)?s?)\b", re.I)


async def _try_agent_turn(question, components, timing_ms, ask_started):
    """Answer with the agent, or return None so the standard pipeline answers."""

    if not _agent_enabled(components):
        return None

    conversation_memory = components.get("conversation_memory")
    if conversation_memory and should_use_structured_listing_comparison(
        question,
        conversation_memory.context(),
    ):
        return None

    agent = components["listing_agent"]
    history = components.setdefault("agent_history", [])
    profile = components.setdefault("profile", {})
    session_id = components.get("session_id")
    if session_id and not components.get("memory_loaded"):
        # first turn of this session in this process: pick up what was saved before a restart or idle cleanup
        components["memory_loaded"] = True
        saved_history, saved_profile = await asyncio.to_thread(session_store.load, session_id)
        history[:0] = saved_history
        for key, value in saved_profile.items():
            profile.setdefault(key, value)
    llm = components["rag_chain"].llm

    started = time.perf_counter()
    # Same concurrency cap as every other model call, so the provider is not flooded.
    await llm._acquire_slot()
    try:
        result = await asyncio.to_thread(
            agent.answer, question, list(history), None, profile, _agent_results_hook.get()
        )
    except LLMQueueTimeoutError:
        raise
    except Exception as exc:
        # Model unreachable, out of credit, or anything unexpected.
        print(f"ACOT agent unavailable, using the standard pipeline: {exc}")
        return None
    finally:
        llm._release_slot()

    add_tokens(result.tokens)
    if not (result.answer or "").strip():
        return None

    projects, listings, labels = [], [], {}
    for card in result.cards:
        if card.get("kind") == "project":
            row = card["row"]
            projects.append(row)
            labels[card["listing_id"]] = row.get("name") or "this project"
        else:
            listings.append(_agent_listing_record(card))
            labels[card["listing_id"]] = card.get("title") or "this rental"

    # [[id]] tags become the project or rental name; the card is in the response data.
    answer = _AGENT_TAG.sub(lambda m: labels.get(m.group(1), ""), result.answer)
    answer = _collapse_doubled_names(answer, set(labels.values()))
    answer = _drop_repeated_name_lines(answer, set(labels.values()))
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()

    used = {tool["name"] for tool in result.tools if tool.get("ok")}
    # Plain browse/search should not invent an Acot Recommend section.
    wants_recommend = bool(re.search(
        r"\b(?:invest|buy|recommend|best|compar(?:e|ison)|which should|suggest)\b",
        question or "",
        re.I,
    ))
    if (
        not wants_recommend
        and "compare_projects" not in used
        and re.search(r"(?i)\b(?:acot recommend|recommendation)\b", answer)
    ):
        answer = re.split(
            r"(?im)^\s*(?:\*\*)?(?:acot recommend|recommendation)\b.*",
            answer,
            maxsplit=1,
        )[0].rstrip()
        answer = re.sub(
            r"(?is)\n+\s*(?:i suggest|i recommend|other strong options include).*$",
            "",
            answer,
        ).rstrip()

    if "compare_projects" in used and len(projects) >= 2:
        question_type = "comparison"
    elif len(projects) + len(listings) > 1:
        question_type = "search"
    elif len(projects) + len(listings) == 1:
        question_type = "information"
    else:
        question_type = "general"

    shown = "; ".join(f"{name} (id {key})" for key, name in labels.items())
    history.append({"role": "user", "content": question})
    history.append({
        "role": "assistant",
        "content": answer + (f"\n[internal note, not part of the reply. Shown: {shown}]" if shown else ""),
    })
    del history[:-12]
    if session_id and session_store.enabled():
        await asyncio.to_thread(session_store.save, session_id, list(history), dict(profile))

    structured_summary = _empty_structured_summary()
    structured_summary["projects"] = projects
    structured_summary["listings"] = listings
    timing_ms["agent"] = _elapsed_ms(started)
    # "Tell me about JVC": no cards, so show the community the answer is about.
    # "Should I buy in JVC?": the same community, with its market signals from sales and rent records.
    investing = bool(_INVEST_ASK.search(question or ""))
    # A question weighing places ("JVC or Downtown?") never gets a single community card.
    if (investing or (not projects and not listings)) and not looks_like_comparison(question):
        area = mentioned_area(question)
        if not area and investing and projects:
            area = Counter(str(row.get("community") or "") for row in projects).most_common(1)[0][0] or None
        official = await asyncio.to_thread(official_area, area) if area else None
        if official:
            community = {"name": area, "official_area": official}
            if investing:
                community["focus"] = "investment"
                community["signals"] = await asyncio.to_thread(market_signals, official)
            structured_summary["community"] = [community]
    # "Which area has the most sales?": the ranked areas go to the screen as a list.
    if "area_sales" in used:
        ranked = await asyncio.to_thread(top_areas_by_sales, 8)
        structured_summary["community"] = [
            {"name": row["area"], "official_area": row["official_area"], "focus": "ranking", "ranking": row}
            for row in ranked
        ]
    # The agent sometimes only searches brochure text and returns no project: look the project up by name.
    if _DOC_ASK.search(question or "") and not any(row.get("brochure_url") for row in projects):
        name = re.sub(r"\s+", " ", _DOC_WORDS.sub(" ", question or "")).strip(" ?.!,")
        rows = await asyncio.to_thread(_find_projects, name) if name else []
        if len(rows) == 1:
            projects.append(rows[0])
    # "Send me the brochure": the projects' brochure links go to the screen as documents.
    documents = [
        {"id": row.get("id"), "document_title": f"{row.get('name') or 'Project'} brochure",
         "document_type": "Brochure", "document_url": row["brochure_url"]}
        for row in projects
        if row.get("brochure_url")
    ] if _DOC_ASK.search(question or "") else []

    return {
        "question": question,
        "standalone_question": question,
        "question_type": question_type,
        "structured_summary": structured_summary,
        "documents": documents,
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": question_type,
            "structured_data": {
                "community": [],
                "projects": projects,
                "sub_communities": [],
                "listings": listings,
            },
            "documents": documents,
        },
        "rag_chain": components["rag_chain"],
        "conversation_memory": components["conversation_memory"],
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "timing_ms": timing_ms,
        "ask_started": ask_started,
        "small_talk": None,
        "agent": {
            "model": result.model,
            "tools": [tool["name"] for tool in result.tools],
            "tokens": result.tokens,
        },
    }


async def _prepare_acot_turn(
    question,
    components
):
    """Retrieve evidence and choose the fast or model answer path.

    Conversation memory is not updated here. The caller updates it only
    after the full answer exists.
    """

    hybrid_retriever = components["hybrid_retriever"]
    context_builder = components["context_builder"]
    investment_analyzer = components["investment_analyzer"]
    rag_chain = components["rag_chain"]
    conversation_memory = components["conversation_memory"]

    timing_ms = {}
    ask_started = time.perf_counter()

    small_talk = _classify_small_talk(question)
    if small_talk:
        timing_ms["rewrite"] = 0
        return {
            "question": question,
            "standalone_question": question,
            "question_type": "general",
            "structured_summary": _empty_structured_summary(),
            "documents": [],
            "investment_analysis": None,
            "retrieval_result": {
                "question_type": "general",
                "structured_data": {
                    "community": [],
                    "projects": [],
                    "sub_communities": [],
                },
                "documents": [],
            },
            "rag_chain": rag_chain,
            "conversation_memory": conversation_memory,
            "use_fast": True,
            "fast_answer": _small_talk_answer(small_talk, components.get("user_name")),
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "small_talk": small_talk,
        }

    # "Compare Dubai Marina and Business Bay": a table built from the database, no model call.
    areas = None
    try:
        areas = await asyncio.to_thread(parse_area_comparison, question)
        compared = await asyncio.to_thread(build_area_comparison, *areas) if areas else None
    except Exception as exc:
        print(f"Area comparison unavailable, using the normal path: {exc}")
        compared = None
    if compared:
        timing_ms["rewrite"] = 0
        history = components.setdefault("agent_history", [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": compared})
        del history[:-12]
        # Build comparison_entities from area facts so the frontend renders cards.
        from app.answers.area_comparison import _beds_text, _facts as _area_facts
        _area_comparison_entities = []
        if areas:
            for _area_name in areas:
                try:
                    _f = _area_facts(_area_name)
                    # Build a minimal project-like record so _build_comparison can score it.
                    _rec = {
                        "name": _area_name,
                        "community": _area_name,
                        "price": _f.get("min_price"),
                        "bedroom_min": None,
                        "bedroom_max": None,
                        "size_min": None,
                        "size_max": None,
                        "rent_median": _f.get("rent_median"),
                        "projects_count": _f.get("projects"),
                        "listings_count": _f.get("listings"),
                        # Shown as extra rows in the app's comparison table.
                        "area_facts": {
                            "projects": _f.get("projects"),
                            "median_price": _f.get("median_price"),
                            "handover_year": (_f.get("top_year") or {}).get("year"),
                            "bedrooms": _beds_text(_f["beds"]) if _f.get("beds") else None,
                            "rentals": _f.get("listings"),
                            "rent_median": _f.get("rent_median"),
                        },
                    }
                    _area_comparison_entities.append({
                        "entity_type": "community",
                        "requested_name": _area_name,
                        "resolved_name": _area_name,
                        "data": {"projects": [_rec], "listings": []},
                    })
                except Exception:
                    pass
        _area_summary = _empty_structured_summary()
        _area_summary["comparison"] = len(_area_comparison_entities) >= 2
        _area_summary["comparison_entities"] = _area_comparison_entities
        return {
            "question": question,
            "standalone_question": question,
            "question_type": "general",
            "structured_summary": _area_summary,
            "documents": [],
            "investment_analysis": None,
            "retrieval_result": {
                "question_type": "general",
                "structured_data": {"community": [], "projects": [], "sub_communities": []},
                "documents": [],
            },
            "rag_chain": rag_chain,
            "conversation_memory": conversation_memory,
            "use_fast": True,
            "fast_answer": compared,
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "small_talk": None,
        }

    transaction_turn = await asyncio.to_thread(
        _transaction_turn, question, conversation_memory
    )
    if transaction_turn is not None:
        transaction_turn["timing_ms"] = timing_ms
        transaction_turn["ask_started"] = ask_started
        transaction_turn["rag_chain"] = rag_chain
        transaction_turn["conversation_memory"] = conversation_memory
        return transaction_turn

    compare_shown = await asyncio.to_thread(
        _compare_shown_turn, question, conversation_memory
    )
    if compare_shown is not None:
        compare_shown["timing_ms"] = timing_ms
        compare_shown["ask_started"] = ask_started
        compare_shown["rag_chain"] = rag_chain
        compare_shown["conversation_memory"] = conversation_memory
        return compare_shown

    shortlist_turn = await asyncio.to_thread(
        _shortlist_recommend_turn, question, conversation_memory
    )
    if shortlist_turn is not None:
        shortlist_turn["timing_ms"] = timing_ms
        shortlist_turn["ask_started"] = ask_started
        shortlist_turn["rag_chain"] = rag_chain
        shortlist_turn["conversation_memory"] = conversation_memory
        return shortlist_turn

    shown_followup = _shown_project_followup(question, conversation_memory)
    if shown_followup is not None:
        projects, no_match = shown_followup
        summary = _empty_structured_summary()
        if no_match:
            summary["filter_no_match"] = True
            summary["filter_evidence_projects"] = projects
        else:
            summary["projects"] = projects
            summary["project_total"] = len(projects)
        return {
            "question": question,
            "standalone_question": question,
            "question_type": "search",
            "structured_summary": summary,
            "documents": [],
            "investment_analysis": None,
            "retrieval_result": {
                "question_type": "search",
                "structured_data": {
                    "community": [],
                    "projects": [] if no_match else projects,
                    "sub_communities": [],
                },
                "documents": [],
            },
            "rag_chain": rag_chain,
            "conversation_memory": conversation_memory,
            "use_fast": True,
            "fast_answer": _build_fast_structured_answer(question, summary),
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "small_talk": None,
        }

    agent_turn = await _try_agent_turn(
        question, components, timing_ms, ask_started
    )
    if agent_turn is not None:
        return agent_turn

    rewrite_started = time.perf_counter()
    question_for_search = prepare_user_question(question) or question
    standalone_question = await conversation_memory.rewrite_question(
        question_for_search
    )
    timing_ms["rewrite"] = _elapsed_ms(rewrite_started)

    context = conversation_memory.context()
    context["current_user_question"] = question_for_search
    result = await hybrid_retriever.retrieve(
        question=standalone_question,
        conversation_context=context,
    )
    timing_ms.update(result.get("timing_ms") or {})

    question_type = result.get(
        "question_type",
        "general"
    )

    structured_data = result.get(
        "structured_data",
        {}
    )

    documents = result.get(
        "documents",
        []
    )

    structured_summary = {
        "community": structured_data.get(
            "community",
            []
        ),
        "projects": structured_data.get(
            "projects",
            []
        ),
        "sub_communities": structured_data.get(
            "sub_communities",
            []
        ),
        "listings": structured_data.get(
            "listings",
            []
        ) or result.get("listings", []),
        "listing_total": structured_data.get("listing_total"),
        "catalog": structured_data.get("catalog", False),
        "catalog_place": structured_data.get("catalog_place"),
        "project_total": structured_data.get("project_total"),
        "community_total": structured_data.get("community_total"),
        "comparison": structured_data.get("comparison", False),
        "comparison_entities": structured_data.get("comparison_entities") or [],
        "filter_no_match": result.get(
            "filter_no_match",
            structured_data.get(
                "filter_no_match",
                False
            )
        ),
        "filter_evidence_projects": result.get(
            "filter_evidence_projects",
            structured_data.get(
                "filter_evidence_projects",
                []
            )
        ) or [],
        "filter_evidence_listings": result.get(
            "filter_evidence_listings",
            structured_data.get(
                "filter_evidence_listings",
                []
            )
        ) or [],
        "understood_subject": structured_data.get("understood_subject"),
        "understood_kind": structured_data.get("understood_kind"),
        "closest_match": structured_data.get("closest_match"),
        "ambiguous_label": structured_data.get("ambiguous_label"),
    }

    investment_analysis = None
    investment_started = time.perf_counter()

    if question_type == "investment":
        try:
            investment_analysis = (
                investment_analyzer.analyze(
                    structured_summary
                )
            )
        except Exception:
            investment_analysis = None

    timing_ms["investment"] = _elapsed_ms(investment_started)

    comparison_built = _build_comparison(structured_summary)
    if comparison_built:
        structured_summary["comparison_view"] = comparison_built["sides"]

    open_recommend = None
    projects_for_rank = [
        project for project in (structured_summary.get("projects") or [])
        if isinstance(project, dict) and project.get("name")
    ]
    if question_type == "investment" and len(projects_for_rank) >= 2:
        from app.answers.recommend import format_shortlist, rank_projects, wanted_bedrooms
        open_recommend = rank_projects(
            projects_for_rank,
            wanted_beds=wanted_bedrooms(standalone_question),
        )
        structured_summary["recommendation"] = open_recommend.get("recommendation")
        if investment_analysis is not None:
            investment_analysis = dict(investment_analysis)
            investment_analysis["project_recommendation"] = open_recommend.get("recommendation")
            investment_analysis["recommendation_text"] = format_shortlist(
                open_recommend,
                question=standalone_question,
                assumptions=(
                    "Ranked on price, size, sales/transactions and rental activity."
                ),
            )

    use_fast = _is_fast_structured_question(
        question=standalone_question,
        question_type=question_type,
        structured_summary=structured_summary,
        documents=documents,
    )
    if comparison_built:
        use_fast = True
    elif open_recommend is not None:
        use_fast = True
    elif re.search(
        r"\b(?:which of these|which ones|which of those|which of the)\b",
        f"{question} {standalone_question}",
        flags=re.IGNORECASE,
    ) and (
        structured_summary.get("listings")
        or structured_summary.get("filter_evidence_listings")
        or structured_summary.get("projects")
        or structured_summary.get("filter_evidence_projects")
    ):
        use_fast = True
    if (
        not documents
        and not structured_summary.get("projects")
        and structured_summary.get("understood_subject")
        and not structured_summary.get("filter_no_match")
    ):
        use_fast = True

    if comparison_built:
        fast_answer = comparison_built["answer"]
        context = None
    elif open_recommend is not None:
        from app.answers.recommend import format_shortlist
        fast_answer = format_shortlist(
            open_recommend,
            question=standalone_question,
            assumptions=(
                "Ranked on price, size, sales/transactions and rental activity."
            ),
        )
        context = None
    elif use_fast:
        fast_answer = _build_fast_structured_answer(
            question=standalone_question,
            structured_summary=structured_summary,
        )
        context = None
    else:
        fast_answer = None
        context = context_builder.build_context(
            structured_summary=structured_summary,
            investment_analysis=investment_analysis,
            document_results=documents,
            community_results=structured_data.get(
                "community",
                []
            )
        )

    return {
        "question": question,
        "standalone_question": standalone_question,
        "question_type": question_type,
        "structured_summary": structured_summary,
        "documents": documents,
        "investment_analysis": investment_analysis,
        "retrieval_result": result,
        "rag_chain": rag_chain,
        "conversation_memory": conversation_memory,
        "use_fast": use_fast,
        "fast_answer": fast_answer,
        "context": context,
        "timing_ms": timing_ms,
        "ask_started": ask_started,
    }


def _finalize_acot_turn(prepared, answer, answer_started):
    timing_ms = dict(prepared["timing_ms"])
    timing_ms["answer"] = _elapsed_ms(answer_started)
    timing_ms["ask"] = _elapsed_ms(prepared["ask_started"])

    prepared["conversation_memory"].update(
        user_question=prepared["question"],
        standalone_question=prepared["standalone_question"],
        answer=answer,
        retrieval_result=prepared["retrieval_result"],
    )

    response = build_frontend_response(
        question=prepared["question"],
        standalone_question=prepared["standalone_question"],
        answer=answer,
        question_type=prepared["question_type"],
        structured_summary=prepared["structured_summary"],
        documents=prepared["documents"],
        investment_analysis=prepared["investment_analysis"],
    )
    response.setdefault("meta", {})["timing_ms"] = timing_ms
    if prepared.get("agent"):
        response["meta"]["agent"] = prepared["agent"]
    return response


async def ask_acot(
    question,
    components
):
    prepared = await _prepare_acot_turn(
        question,
        components
    )

    answer_started = time.perf_counter()
    if prepared["use_fast"]:
        answer = prepared["fast_answer"]
    else:
        answer = await prepared["rag_chain"].generate_answer(
            question=prepared["standalone_question"],
            context=prepared["context"],
            question_type=prepared["question_type"]
        )

    return _finalize_acot_turn(
        prepared,
        answer,
        answer_started,
    )


def _iter_typewriter_pieces(text):
    """Split a finished database answer into small typewriter pieces."""

    tokens = re.findall(r"\S+\s*|\n+", text or "")
    if not tokens:
        if text:
            yield text
        return

    group_size = 1 if len(tokens) <= 80 else max(1, len(tokens) // 40)
    buffer = []
    for token in tokens:
        buffer.append(token)
        if len(buffer) >= group_size:
            yield "".join(buffer)
            buffer = []
    if buffer:
        yield "".join(buffer)


def _stream_sources(prepared):
    sources = []
    projects = prepared["structured_summary"].get("projects") or []
    for project in projects[:6]:
        if not isinstance(project, dict):
            continue
        title = project.get("name") or "Project"
        bits = [
            project.get("community"),
            project.get("developer_name") or project.get("developer"),
        ]
        excerpt = " · ".join(
            str(bit).strip()
            for bit in bits
            if bit is not None and str(bit).strip()
        )
        sources.append({
            "title": title,
            "excerpt": excerpt[:240],
        })
    return sources


_TRANSACTION_FOLLOWUP = "Show me the recent transaction details"

_PROJECT_FOLLOWUPS = (
    ("transactions", _TRANSACTION_FOLLOWUP),
    ("price", "What are their starting prices?"),
    ("bedroom", "Which of these have 2-bedroom options?"),
    ("handover", "What are their handover dates?"),
    ("developer", "Who is the developer of each of these projects?"),
    (
        "compare",
        "Compare the first and second ones based on price, bedroom range, and handover date.",
    ),
    (
        "amenities",
        "What amenities do those projects offer according to their documents?",
    ),
    ("size", "What is the size range of each project?"),
)


def _question_topics(text):
    question = (text or "").lower()
    topics = set()
    if any(term in question for term in ("price", "prices", "cost")):
        topics.add("price")
    if "bedroom" in question:
        topics.add("bedroom")
    if "handover" in question:
        topics.add("handover")
    if "developer" in question:
        topics.add("developer")
    if any(term in question for term in ("compare", "comparison", "versus", " vs ")):
        topics.add("compare")
    if "amenit" in question or "facilities" in question:
        topics.add("amenities")
    if any(term in question for term in ("size", "sqft", "square foot")):
        topics.add("size")
    if any(term in question for term in ("transaction", "transactions", "transcation", "sale history")):
        topics.add("transactions")
    return topics


def _questions_already_asked(prepared):
    asked = [prepared.get("question") or ""]
    memory = prepared.get("conversation_memory")
    turns = getattr(memory, "turns", None) or []
    for turn in turns:
        if isinstance(turn, dict):
            asked.append(turn.get("user_question") or "")
            asked.append(turn.get("standalone_question") or "")
    return asked


def _unused_followups(prepared, chips, pinned=None):
    asked = _questions_already_asked(prepared)
    asked_text = {_normalize_chat_text(item) for item in asked if item}
    covered = set()
    for item in asked:
        covered.update(_question_topics(item))

    unused = []
    for chip in chips:
        if _normalize_chat_text(chip) in asked_text:
            continue
        if _question_topics(chip) & covered:
            continue
        unused.append(chip)

    pinned_chip = pinned or None
    if pinned_chip and _normalize_chat_text(pinned_chip) not in asked_text:
        unused = [chip for chip in unused if _normalize_chat_text(chip) != _normalize_chat_text(pinned_chip)]
        unused = [pinned_chip, *unused]
    return unused[:3]


def _followup_place(prepared):
    summary = prepared.get("structured_summary") or {}
    for key in ("listings", "projects"):
        places = []
        for row in summary.get(key) or []:
            if not isinstance(row, dict):
                continue
            place = str(row.get("community") or "").strip()
            if place and place not in places:
                places.append(place)
        if len(places) == 1:
            return places[0]

    subject = str(summary.get("understood_subject") or "").strip()
    if subject and subject.lower() not in {"that request", "your request"}:
        return subject

    match = re.search(r"\bin\s+(.+)$", prepared.get("question") or "", flags=re.I)
    if not match:
        return None
    place = re.split(
        r"\b(?:and tell|tell me|which is|for rent|to rent)\b",
        match.group(1),
        maxsplit=1,
        flags=re.I,
    )[0]
    place = place.strip(" .?")
    return place or None


def _listing_followup_chips(place, listings):
    if len(listings) == 1:
        title = listings[0].get("title") or listings[0].get("name") or "this rental"
        return [
            f"What is the yearly rent of {title}?",
            f"How many bedrooms does {title} have?",
            f"What is the size of {title}?",
        ]

    where = f" in {place}" if place else ""
    return [
        f"Compare the first two rentals{where} and tell me which is best.",
        "Which of these has the lowest yearly rent?",
        "Which of these are 1-bedroom apartments?",
    ]


def _followup_questions(prepared):
    question = prepared.get("question") or ""

    if prepared.get("small_talk"):
        return _unused_followups(prepared, _STARTER_CHIPS)

    summary = prepared.get("structured_summary") or {}
    if summary.get("transaction_followup") or "transactions" in _question_topics(question):
        answer = prepared.get("fast_answer") or ""
        if "Recent Transactions" in answer or summary.get("transaction_followup"):
            return _unused_followups(
                prepared,
                [
                    _MORE_TRANSACTION_FOLLOWUP,
                    "What are their starting prices?",
                    "What are their handover dates?",
                    "Who is the developer of each of these projects?",
                ],
                pinned=_MORE_TRANSACTION_FOLLOWUP,
            )

    projects = [
        project
        for project in (summary.get("projects") or [])
        if isinstance(project, dict)
    ]
    listings = [
        listing
        for listing in (summary.get("listings") or [])
        if isinstance(listing, dict)
    ]
    place = _followup_place(prepared)

    if listings or asks_for_listings(question):
        return _unused_followups(
            prepared,
            _listing_followup_chips(place, listings),
        )

    if len(projects) == 1:
        name = projects[0].get("name") or "this project"
        return _unused_followups(
            prepared,
            [
                _TRANSACTION_FOLLOWUP,
                f"What is the starting price of {name}?",
                f"What is the handover date of {name}?",
                f"Who is the developer of {name}?",
                f"What amenities does {name} offer?",
                f"What is the bedroom range of {name}?",
            ],
            pinned=_TRANSACTION_FOLLOWUP,
        )

    if len(projects) >= 2:
        return _unused_followups(
            prepared,
            [chip for _, chip in _PROJECT_FOLLOWUPS],
            pinned=_TRANSACTION_FOLLOWUP,
        )

    if place and re.search(r"\bprojects?\b", question, flags=re.I):
        return _unused_followups(
            prepared,
            [
                _TRANSACTION_FOLLOWUP,
                f"What are the starting prices of projects in {place}?",
                f"What are the handover dates of projects in {place}?",
                f"Compare the first and second projects in {place}.",
            ],
            pinned=_TRANSACTION_FOLLOWUP,
        )

    return []


def _slim_stream_response(response):
    """Drop brochure essays from the stream so the last event stays small."""

    if not isinstance(response, dict):
        return response

    slim = dict(response)
    data = dict(slim.get("data") or {})
    projects = []
    for project in data.get("projects") or []:
        if not isinstance(project, dict):
            continue
        item = dict(project)
        item.pop("description", None)
        photos = item.get("photos")
        if isinstance(photos, list):
            item["photos"] = photos[:4]
        projects.append(item)
    data["projects"] = projects
    listings = []
    for listing in data.get("listings") or []:
        if not isinstance(listing, dict):
            continue
        item = dict(listing)
        photos = item.get("photos")
        if isinstance(photos, list):
            item["photos"] = photos[:1]
        listings.append(item)
    data["listings"] = listings
    comparison = []
    for side in data.get("comparison") or []:
        if not isinstance(side, dict):
            continue
        item = dict(side)
        photos = item.get("photos")
        if isinstance(photos, list):
            item["photos"] = photos[:1]
        comparison.append(item)
    data["comparison"] = comparison
    slim["data"] = data

    summary = slim.get("ai_summary")
    if isinstance(summary, dict) and isinstance(summary.get("data"), dict):
        summary = dict(summary)
        summary_data = dict(summary["data"])
        summary_data["projects"] = projects
        summary["data"] = summary_data
        slim["ai_summary"] = summary
    return slim


# Voice mode: set while a turn is being prepared, so the agent can start the spoken reply
# from its search results instead of waiting for the whole written answer (saves 1-4s).
_agent_results_hook = contextvars.ContextVar("agent_results_hook", default=None)


async def stream_acot(
    question,
    components,
    mode="text",
    language=None,
):
    """Yield typed stream events: meta, delta, followups, then done.

    mode="voice" also yields `speech` events (a short spoken reply written in
    parallel with the chat answer) and one `speech_done` before `done`.

    Database answers are split into small pieces so the client can show
    them like a typewriter. Model answers keep the provider's own tokens.
    Memory is updated only after the full answer is assembled.
    """

    # No stdout redirect: the API installs a logging-backed sys.stdout at
    # startup, so pipeline output is captured without mutating global state
    # per request (which interleaved output across concurrent requests).
    early = {}
    hook_token = None
    # Document and investment requests wait for the full result: brochures and market signals are added after the agent.
    if mode == "voice" and not _DOC_ASK.search(question or "") and not _INVEST_ASK.search(question or ""):
        loop = asyncio.get_running_loop()

        def start_early(tool_outputs):
            facts = "- Search results:\n" + "\n".join(tool_outputs)[:MAX_FACT_CHARS]
            seed = {"context": facts, "conversation_memory": components.get("conversation_memory")}
            # Runs in the agent's thread; the reply's task must start on the event loop.
            loop.call_soon_threadsafe(lambda: early.setdefault("speech", start_spoken_reply(
                components["rag_chain"].llm, question, seed, components.get("user_name"), language)))

        hook_token = _agent_results_hook.set(start_early)
    preparing = asyncio.create_task(_prepare_acot_turn(question, components))
    try:
        # Send the early spoken reply while the written answer is still being made.
        while not preparing.done():
            await asyncio.wait({preparing}, timeout=0.05)
            for text in early["speech"].drain() if "speech" in early else ():
                yield ("speech", {"type": "speech", "text": text})
        prepared = preparing.result()
    except BaseException:
        if "speech" in early:
            early["speech"].cancel()
        raise
    finally:
        if not preparing.done():
            preparing.cancel()
        if hook_token is not None:
            _agent_results_hook.reset(hook_token)

    yield (
        "meta",
        {
            "type": "meta",
            "sources": _stream_sources(prepared),
        }
    )

    speech = early.get("speech") or (
        start_spoken_reply(
            prepared["rag_chain"].llm,
            question,
            prepared,
            components.get("user_name"),
            language,
        )
        if mode == "voice"
        else None
    )

    def spoken_so_far():
        for text in speech.drain() if speech else ():
            yield ("speech", {"type": "speech", "text": text})

    try:
        answer_started = time.perf_counter()
        if prepared["use_fast"]:
            answer = prepared["fast_answer"] or ""
            for piece in _iter_typewriter_pieces(answer):
                yield (
                    "delta",
                    {"type": "delta", "text": piece}
                )
                for event in spoken_so_far():
                    yield event
                await asyncio.sleep(0.02)
        else:
            parts = []

            async for delta in prepared["rag_chain"].stream_answer(
                question=prepared["standalone_question"],
                context=prepared["context"],
                question_type=prepared["question_type"]
            ):
                if not delta:
                    continue

                parts.append(delta)
                yield (
                    "delta",
                    {"type": "delta", "text": delta}
                )
                for event in spoken_so_far():
                    yield event

            answer = "".join(parts).strip()

            if not answer:
                raise RuntimeError(
                    "LLM returned an empty answer."
                )

        followups = _followup_questions(prepared)
        if followups:
            yield (
                "followups",
                {"type": "followups", "questions": followups}
            )

        if speech:
            # Finish the spoken reply before `done`, so its tokens are billed with this turn.
            for event in spoken_so_far():
                yield event
            async for text in speech.rest():
                yield ("speech", {"type": "speech", "text": text})
            yield ("speech_done", {"type": "speech_done", "ok": speech.ok})

        yield (
            "done",
            _finalize_acot_turn(
                prepared,
                answer,
                answer_started,
            )
        )
    finally:
        if speech:
            speech.cancel()


def _elapsed_ms(started):
    return int(round((time.perf_counter() - started) * 1000))


# =========================================================
# SEQUENTIAL ANSWER OUTPUT
# =========================================================

def print_answer_sequentially(answer, delay=0.004):
    """
    Display the final ACOT answer progressively instead of printing
    the entire response at once.

    The answer itself is still generated by the same RAG pipeline.
    Only the terminal presentation is changed.
    """

    for char in str(answer):
        print(
            char,
            end="",
            flush=True
        )

        if char == "\n":
            continue

        time.sleep(delay)


# =========================================================
# MAIN CHAT LOOP
# =========================================================

def main():

    # Hide initialization logs/warnings from the terminal.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        components = initialize_acot()

    while True:

        try:
            question = input("\nQuestion: ").strip()

        except KeyboardInterrupt:
            print()
            break

        except EOFError:
            print()
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit"
        }:
            print()
            break

        try:
            # Hide all internal ACOT/RAG/retriever/library output.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                response = asyncio.run(ask_acot(
                    question,
                    components
                ))

            print("\nACOT:")
            print_answer_sequentially(
                response.get("answer", "")
            )
            print(
                "\nACOT timing_ms:",
                response.get("meta", {}).get("timing_ms", {}),
            )

            #print("\n\nFRONTEND RESPONSE:")
            #print(
                #json.dumps(
                    #response,
                    #indent=2,
                    #ensure_ascii=False
                #)
            #)
            print()

        except Exception as e:
            print(f"\nACOT ERROR: {e}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()