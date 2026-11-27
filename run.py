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

from app.listings.router import is_rent_browse
from app.listings.tools import _find_projects, top_areas_by_sales
from app.answers.intent_route import (
    extract_place,
    is_incomplete_home_browse,
    is_no_preference_reply,
    is_show_more,
    parse_bedrooms,
    parse_budget_aed,
    prior_anchor_question,
    reuse_last_search,
    wants_more_results,
)
from app.listings.agent import EARLIER_NAME, ListingAgent, ListingsUnavailable, _ID_TAG as _AGENT_TAG
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
    repair_typed_question,
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
    # Spoken false starts must stay small-talk, not full hybrid RAG.
    "uh", "um", "erm", "ah", "oh", "hmm", "hmmm", "like", "just",
    "please", "can", "could", "would", "will", "me", "my", "name",
    "is", "this", "that", "here", "okay", "ok", "yeah", "yep", "yes",
}
_SMALL_TALK_REAL_ASK = re.compile(
    r"\b(?:show|find|list|search|projects?|propert(?:y|ies)|apartments?|"
    r"villas?|townhouses?|listings?|agents?|brokers?|price|prices|budget|"
    r"bedrooms?|handover|invest|buy|rent|compare|rera|escrow|marina|jvc|"
    r"jlt|downtown|palm|bay)\b",
    re.I,
)


_TRANSACTION = re.compile(
    r"\b(?:"
    r"transactions?|transcations?|trancations?|transctions?|"
    r"sale history|sales history|"
    # Demo wording: "recent sales", "last 5 sales", "were sold in November"
    r"(?:recent|latest|last|newest|recorded|past)\s+sales?|"
    r"sales?\s+(?:in|of|for|from|under|over|between|with)|"
    r"last\s+\d+\s+sales?|"
    r"how many\b[\s\S]{0,60}?\bsold\b|"
    r"\bsold\b[\s\S]{0,40}?\b(?:in|during|this|last|november|october|september|august|july|june|may|april|march|february|january|20\d{2})"
    r")\b",
    re.I,
)
_TRANSACTION_BROWSE = re.compile(
    r"\b(?:show|list|find|give|tell|latest|recent|last|newest|more|how many|sold)\b",
    re.I,
)
_TRANSACTION_FILLER = re.compile(
    r"\b(?:show|me|tell|find|list|give|the|top|best|latest|recent|last|newest|"
    r"recorded|please|can|you|could|what|are|is|any|all|how|many|"
    r"transactions?|transcations?|trancations?|transctions?|sales?|sold|"
    r"sale history|sales history|details?|info|information|about|"
    r"for|in|at|of|dubai|apartments?|bedroom|bedrooms?|studio|studios?)\b",
    re.I,
)
_MORE_TRANSACTIONS = re.compile(
    r"\b(?:more|another|next|additional|further)\b",
    re.I,
)
_TRANSACTION_PAGE_SIZE = 5
_MORE_TRANSACTION_FOLLOWUP = "Can you show me more transactions"
_LAST_N_SALES = re.compile(r"\blast\s+(\d{1,2})\s+sales?\b", re.I)
# Bedroom-only / "same but 2 beds" after an active sales task (no "sales" word needed).
_SALES_SLOT_TWEAK = re.compile(
    r"\b(?:same|again|instead|also|but|now|what about|how about)\b|"
    r"\b(?:\d+|one|two|three|four|five|studio)\s*-?\s*bed(?:room)?s?\b|"
    r"\bstudio\b",
    re.I,
)


def _parse_sale_limit(question: str | None) -> int | None:
    """'last 5 sales' → 5; None when the user did not ask for a fixed last-N."""
    match = _LAST_N_SALES.search(question or "")
    if not match:
        return None
    return max(1, min(int(match.group(1)), 15))


def _sale_list_limit(question: str | None = None, state: dict | None = None) -> int:
    """Prefer explicit last-N, then carried dialogue_state.limit, else default 8."""
    from_q = _parse_sale_limit(question)
    if from_q is not None:
        return from_q
    if isinstance(state, dict) and state.get("limit") is not None:
        try:
            return max(1, min(int(state["limit"]), 15))
        except (OverflowError, TypeError, ValueError):
            pass
    prior = None
    if isinstance(state, dict):
        prior = state.get("source_question")
    from_prior = _parse_sale_limit(prior)
    if from_prior is not None:
        return from_prior
    return 8
_SALE_COUNT_ASK = re.compile(r"\bhow many\b[\s\S]{0,80}\bsold\b", re.I)
# Extreme class only — whether this is a sales aggregate comes from
# turn understanding + dialogue context, not from synonym lists.
_CHEAPEST_SALE_ASK = re.compile(
    r"\b(?:cheapest|lowest|least\s+expensive)\b",
    re.I,
)
_HIGHEST_SALE_ASK = re.compile(
    r"\b(?:highest|most\s+expensive|priciest|maximum)\b",
    re.I,
)
_AVERAGE_SALE_ASK = re.compile(
    r"\b(?:average|median|mean)\b",
    re.I,
)
_WHICH_AMONG_SHOWN = re.compile(
    r"\b(?:which\s+one|which\s+(?:project|of\s+(?:these|those|them))|"
    r"among\s+(?:these|those)|(?:of|from)\s+(?:these|those|them))\b",
    re.I,
)
_PROJECT_COUNT_ASK = re.compile(
    r"\bhow many\b[\s\S]{0,40}\b(?:residential\s+)?projects?\b",
    re.I,
)
_DEVELOPER_RANK_ASK = re.compile(
    r"\b(?:which|top|best)\b[\s\S]{0,50}\bdevelopers?\b|"
    r"\bdevelopers?\b[\s\S]{0,60}\b(?:highest|most|top|largest)\b[\s\S]{0,40}"
    r"\b(?:sales?|value|volume|revenue)\b",
    re.I,
)
# Strict wording for the early route: a ranking of developers by sales value, citywide. The wider pattern above stays
# as the late safety net.
_DEVELOPER_VALUE_RANK = re.compile(
    r"\b(?:top|leading|biggest|largest)\s+(?:\d{1,2}\s+)?developers?\b|"
    r"\bdevelopers?\b[\s\S]{0,60}\b(?:highest|most|largest|biggest|top)\b[\s\S]{0,40}\b(?:sales?|value|volume|revenue|sold)\b",
    re.I,
)
# "Best investment properties under AED 2M with high rental yield": projects ranked by gross yield.
_YIELD_RANK_ASK = re.compile(r"\b(?:yield(?:s|ing)?|roi|rental\s+returns?|return\s+on\s+investment)\b", re.I)
_YIELD_RANK_NOUN = re.compile(r"\b(?:propert(?:y|ies)|projects?|apartments?|flats?|buildings?|towers?|units?)\b", re.I)
_YIELD_RANK_WORDS = re.compile(r"\b(?:best|top|highest|high|good|great|strong|better|most)\b", re.I)
# "Show me in a bar chart": draw the numbers of the projects the last answer showed.
_CHART_ASK = re.compile(
    r"\b(?:bar\s*-?\s*(?:chart|chat|graph)|chart|graph|plot|diagram|visuali[sz]e|visuali[sz]ation|visual\s+representation)\b",
    re.I,
)
# "Separate developer projects from resale" (ACOT QA AI-012, R07)
_SPLIT_ASK = re.compile(r"\b(?:separate|split|launch(?:es)?|resale|primary|secondary market)\b", re.I)
_CONSTRAINT_ASK = re.compile(r"\b(?:constraints?|retain\w*|carr(?:y|ied) over|keeping|which (?:filters|criteria))\b", re.I)
_VILLA_ASK = re.compile(r"\b(?:villas?|townhouses?|town\s+houses?)\b", re.I)
_READY_ONLY_ASK = re.compile(
    r"\b(?:ready(?:\s+only)?|only\s+ready|completed\s+only|only\s+completed)\b",
    re.I,
)
_PRICE_FLOOR_ASK = re.compile(
    r"\b(?:above|over|starting\s+(?:above|from|at)|greater\s+than|more\s+than)\b",
    re.I,
)
_PRICE_CEILING_ASK = re.compile(
    r"\b(?:under|below|up\s+to|max(?:imum)?|no\s+more\s+than|less\s+than)\b",
    re.I,
)
# After a count/sales answer: "ok show me", "yes show", "show them", "list those sales"
_SHOW_PRIOR_SALES = re.compile(
    r"^\s*(?:(?:ok|okay|alright|sure|yes|yep|yeah|please|pls|then|and)\s+){0,3}"
    r"(?:"
    r"show(?:\s+me)?(?:\s+(?:them|those|it|the(?:\s+\w+){0,3}))?|"
    r"list(?:\s+(?:them|those|it|the(?:\s+\w+){0,3}))?|"
    r"see(?:\s+(?:them|those|it|the(?:\s+\w+){0,3}))?|"
    r"(?:the\s+)?(?:sales?|transactions?|details?|list)"
    r")"
    r"\s*[.!]?\s*$",
    re.I,
)


def _transaction_place(question):
    """Area or project named in a transaction ask, after stripping show/latest/transactions."""
    asked = (question or "").strip()
    match = _BROKER_PLACE.search(asked)
    if match:
        place = match.group(1).strip(" .?")
        place = re.split(
            r"\b(?:and tell|tell me|please)\b",
            place,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" .?")
        place = " ".join(_TRANSACTION_FILLER.sub(" ", place).split()).strip(" .?")
        if place:
            return place
    leftover = " ".join(_TRANSACTION_FILLER.sub(" ", asked).split()).strip(" .?")
    return leftover or None


_TX_DEIXIS = re.compile(
    r"\b(?:these|those|them|this list|the projects|the ones|on screen)\b",
    re.I,
)
# Soft follow-ups that keep project cards "live" for a later sales ask.
_TX_SOFT_FOLLOWUP = re.compile(
    r"\b(?:more|handover|developer|bedroom|bedrooms|price|prices|"
    r"amenities|size|compare|which of|transaction|transactions|"
    r"sales?|sold|details?)\b",
    re.I,
)
# Hard topic changes — after two of these, bare "transaction details" must clarify.
_TX_TOPIC_BREAK = re.compile(
    r"\b(?:rent|rental|listings?|agents?|brokers?|famil(?:y|ies)|"
    r"best (?:areas?|communities)|yield|roi|invest(?:ment|ing)?|"
    r"where should i buy|show me (?:projects?|apartments?|villas?)\b)",
    re.I,
)


def _dialogue_action_name(conversation_memory) -> str:
    if conversation_memory is None:
        return ""
    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state() or {}
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    return str((state or {}).get("action") or "").strip().lower()


def _project_card_context_is_live(question, conversation_memory) -> bool:
    """True when on-screen project cards are still a safe subject for sales.

    Users can ask anything anytime. Cards stay live for deixis, an active sales
    thread, or at most one hard topic-break since the cards were shown. After
    two+ unrelated asks (rent / agents / new browse / …), bare transaction
    asks must clarify which community or project — not invent sticky cards.
    """
    if _TX_DEIXIS.search(question or ""):
        return True
    action = _dialogue_action_name(conversation_memory)
    if action in {"sales_list", "sales_count"}:
        return True
    items = getattr(conversation_memory, "active_candidates", None) or []
    projects = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    if not projects:
        return False
    names = {
        str(item.get("name") or "").strip().lower()
        for item in projects
        if item.get("name")
    }
    turns = getattr(conversation_memory, "turns", None) or []
    last_card_idx = -1
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        cands = turn.get("candidate_names") or []
        hits = sum(1 for name in cands if str(name or "").strip().lower() in names)
        if hits >= min(2, len(names)) or (
            len(names) == 1 and hits == 1
        ):
            last_card_idx = index
    if last_card_idx < 0:
        # Cards exist but not yet mirrored into turns — still search/recommend.
        return action in {"search", "recommend", ""}
    breaks = 0
    for turn in turns[last_card_idx + 1 :]:
        if not isinstance(turn, dict):
            continue
        user_q = str(turn.get("user_question") or "")
        if not user_q.strip():
            continue
        lower = user_q.lower()
        if any(name and name in lower for name in names):
            continue
        if _TRANSACTION.search(user_q) or _TX_SOFT_FOLLOWUP.search(user_q):
            # Attribute / sales follow-ups about the same shortlist stay soft
            # unless they clearly open a new topic (rent/agents/new browse).
            if not _TX_TOPIC_BREAK.search(user_q):
                continue
        if _TX_TOPIC_BREAK.search(user_q):
            breaks += 1
        elif len(user_q.split()) >= 4:
            breaks += 1
        if breaks >= 2:
            return False
    return True


def _last_turn_project_stub(question, conversation_memory):
    """The single project the last answer was about, when cards were not mirrored.

    Agent answers (e.g. "What is the starting price of X?") can show one project
    card without leaving it in active_candidates; "show me the recent transaction
    details" must still mean that project, not a place-less clarify.
    """
    if conversation_memory is None:
        return None
    # A named place is its own ask ("sales in Business Bay").
    try:
        from app.retrieval.propai_catalog import place_named
        from app.answers.turn_resolve import sales_place_from_text

        if place_named(question) or sales_place_from_text(question):
            return None
    except Exception:
        pass
    entities = getattr(conversation_memory, "active_entities", None) or {}
    name = str(entities.get("project") or "").strip()
    turns = getattr(conversation_memory, "turns", None) or []
    last = turns[-1] if turns and isinstance(turns[-1], dict) else {}
    if not name:
        names = [str(n).strip() for n in (last.get("candidate_names") or []) if str(n or "").strip()]
        if len(names) == 1:
            name = names[0]
    if not name:
        # The last ask itself named one project ("starting price of Serenz by Danube").
        last_q = str(last.get("user_question") or "").strip()
        if last_q:
            try:
                from app.retrieval.propai_catalog import project_named_in_question

                hit = str(project_named_in_question(last_q) or "").strip()
                if hit and hit.lower() in last_q.lower():
                    name = hit
            except Exception:
                name = ""
    if not name:
        return None
    return [{
        "name": name,
        "entity_type": "project",
        "community": entities.get("community"),
        "developer_name": entities.get("developer"),
    }]


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
    if not projects:
        # Bare "recent transaction details" right after a single-project answer.
        stub = _last_turn_project_stub(question, conversation_memory)
        if stub:
            return stub
    asked = (question or "").lower()
    named = [project for project in projects if str(project["name"]).lower() in asked]
    if named:
        return named[:8]
    # A named area ("sales in Business Bay") must use area-wide latest_sales,
    # not eight random projects from that area. Only on-screen projects apply
    # when the ask does not name a different place.
    from app.retrieval.propai_catalog import place_named
    from app.answers.turn_resolve import sales_place_from_text

    place = place_named(question) or sales_place_from_text(question)
    if place:
        on_screen_same_place = any(
            place.lower() in " ".join(
                str(project.get(k) or "") for k in ("community", "sub_community", "area", "name")
            ).lower()
            for project in projects
        )
        if not on_screen_same_place:
            return None
    if not _project_card_context_is_live(question, conversation_memory):
        return None
    return projects[:8] or None


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
    # Newest sale date first overall — project blocks follow the freshest activity.
    for block in grouped.values():
        block["sales"].sort(
            key=lambda sale: str(sale.get("date") or ""),
            reverse=True,
        )
    order = sorted(
        grouped.keys(),
        key=lambda key: str((grouped[key]["sales"][0] or {}).get("date") or ""),
        reverse=True,
    )
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
    # Grounding line: say what the list is (newest first, how many, which dates).
    dates = sorted(
        str(sale.get("date"))[:10] for sale in sales if sale.get("date")
    )
    span = ""
    if dates:
        span = (
            f" from {dates[0]} to {dates[-1]}" if dates[0] != dates[-1] else f" on {dates[0]}"
        )
    lines.append(
        f"These are the {len(sales)} most recent recorded sale"
        f"{'s' if len(sales) != 1 else ''} for {intro}{span} "
        "(newest first). Source: Dubai Land Department sale records."
    )
    return "\n".join(lines).rstrip()


def _remember_sale_projects(conversation_memory, sales) -> list[dict]:
    """Keep unique project stubs from a sales answer for 'developer of these' follow-ups."""
    stubs: list[dict] = []
    seen: set[str] = set()
    for sale in sales or []:
        name = str((sale or {}).get("project") or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        stubs.append({"name": name, "entity_type": "project"})
    if conversation_memory is not None:
        conversation_memory.last_sale_projects = stubs
    return stubs


def _hydrate_projects_with_developers(projects: list[dict]) -> list[dict]:
    """Fill developer_name from Prop-AI when stubs only have a project name."""
    if not projects:
        return []
    from app.retrieval.propai_catalog import agent_projects

    hydrated = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        name = str(project.get("name") or "").strip()
        if not name:
            continue
        row = dict(project)
        if not str(row.get("developer_name") or row.get("developer") or "").strip():
            try:
                matches, _ = agent_projects({"name": name}, limit=1)
            except Exception:
                matches = []
            if matches:
                match = matches[0]
                row["developer_name"] = match.get("developer_name") or match.get("developer")
                row.setdefault("community", match.get("community"))
                row.setdefault("id", match.get("id"))
        row["entity_type"] = "project"
        hydrated.append(row)
    return hydrated


def _transaction_reply(question, answer, sales=None, followup=True, conversation_memory=None):
    summary = _empty_structured_summary()
    if followup:
        summary["transaction_followup"] = True
    # The sales themselves, so the app can show them by unit type and over time (data.transactions).
    summary["transactions"] = sales or []
    _remember_sale_projects(conversation_memory, sales)
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
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _bed_label(bed) -> str:
    if bed == 0:
        return "studio"
    if bed is None:
        return "homes"
    return "1-bedroom" if int(bed) == 1 else f"{int(bed)}-bedroom"


def _sale_query_from_question(question) -> dict | None:
    """Place / beds / month filters from a sales or sale-count ask."""
    from app.retrieval.propai_catalog import (
        parse_sale_month_year,
        place_named,
        resolve_dubai_place,
    )

    place = (
        extract_place(question)
        or place_named(question)
        or _transaction_place(question)
    )
    resolved = resolve_dubai_place(place) if place else None
    place = resolved or place
    beds = parse_bedrooms(question)
    bed = beds[0] if beds else None
    month, year = parse_sale_month_year(question)
    if not place and bed is None and year is None:
        return None
    return {
        "place": place,
        "bedrooms": bed,
        "year": year,
        "month": month,
        "source_question": question,
    }


def _set_dialogue_state(
    conversation_memory,
    updates: dict | None,
    *,
    replace: bool = False,
) -> dict | None:
    """Write task slots onto session memory (Phase 1 dialogue state)."""
    if conversation_memory is None or not updates:
        return None
    from app.memory.dialogue_state import (
        normalize_dialogue_state,
        patch_dialogue_state,
        sale_filters_from_state,
    )

    if replace:
        state = normalize_dialogue_state(updates)
        if hasattr(conversation_memory, "set_dialogue_state"):
            return conversation_memory.set_dialogue_state(state)
        conversation_memory.dialogue_state = state
        conversation_memory.sale_query = sale_filters_from_state(state)
        return state
    if hasattr(conversation_memory, "patch_dialogue_state"):
        return conversation_memory.patch_dialogue_state(updates)
    current = getattr(conversation_memory, "dialogue_state", None)
    state = patch_dialogue_state(current, updates)
    conversation_memory.dialogue_state = state
    conversation_memory.sale_query = sale_filters_from_state(state)
    return state


def _remember_sale_query(conversation_memory, query: dict | None, *, replace: bool = True) -> None:
    """Persist sales filters into dialogue_state (and legacy sale_query mirror)."""
    if conversation_memory is None or not query:
        return
    action = "sales_count" if query.get("count") is not None and not query.get("listed") else "sales_list"
    if query.get("action") in {"sales_count", "sales_list"}:
        action = query["action"]
    updates = {
        "action": action,
        "place": query.get("place"),
        "beds": query.get("bedrooms") if "bedrooms" in query else query.get("beds"),
        "month": query.get("month"),
        "year": query.get("year"),
        "source_question": query.get("source_question"),
        "offset": int(query.get("offset") or 0),
        "budget_aed": None,
        "ready_only": False,
        "prefer_cheaper": False,
        "goal": None,
        "exclude_names": [],
    }
    if query.get("limit") is not None:
        updates["limit"] = query.get("limit")
    elif query.get("source_question"):
        parsed_limit = _parse_sale_limit(query.get("source_question"))
        if parsed_limit is not None:
            updates["limit"] = parsed_limit
    if query.get("count") is not None:
        updates["last_result"] = {"count": query.get("count")}
    # Follow-up list after a count: keep prior last_result count when replace=False.
    _set_dialogue_state(conversation_memory, updates, replace=replace)


def _prior_sale_query(conversation_memory) -> dict | None:
    if conversation_memory is None:
        return None
    from app.memory.dialogue_state import sale_filters_from_state

    if hasattr(conversation_memory, "get_dialogue_state"):
        from_state = sale_filters_from_state(conversation_memory.get_dialogue_state())
        if from_state and (
            from_state.get("place")
            or from_state.get("bedrooms") is not None
            or from_state.get("year") is not None
        ):
            return from_state
    stored = getattr(conversation_memory, "sale_query", None)
    if isinstance(stored, dict) and stored:
        return stored
    # Plain test doubles may only have dialogue_state.
    from_state = sale_filters_from_state(getattr(conversation_memory, "dialogue_state", None))
    if from_state and (
        from_state.get("place")
        or from_state.get("bedrooms") is not None
        or from_state.get("year") is not None
    ):
        return from_state
    prior = (
        getattr(conversation_memory, "last_user_question", None)
        or getattr(conversation_memory, "last_standalone_question", None)
    )
    if not prior:
        return None
    if _SALE_COUNT_ASK.search(prior) or _TRANSACTION.search(prior):
        return _sale_query_from_question(prior)
    return None


def _dialogue_action(conversation_memory) -> str | None:
    if conversation_memory is None:
        return None
    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state()
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    action = (state or {}).get("action")
    return str(action) if action else None


def _sales_fingerprint(place=None, bed=None, year=None, month=None) -> str:
    """Stable paging key for sales count/list follow-ups."""
    return (
        f"SALES|{(place or 'Dubai').strip().upper()}"
        + (f"|B{bed}" if bed is not None else "")
        + (f"|{year}-{int(month):02d}" if year and month else "")
    )


def _is_show_prior_sales_ask(question, conversation_memory=None) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    # A fresh place/budget ask is not "show the previous sales".
    if parse_budget_aed(text):
        return False
    place = extract_place(text)
    if place and place.lower() not in {"dubai"}:
        return False
    if _SHOW_PRIOR_SALES.search(text):
        return True
    # After a sales answer, plain "show me more" continues that sales list
    # from dialogue_state — do not rewrite it into a project-name search.
    if _dialogue_action(conversation_memory) in {"sales_count", "sales_list"}:
        if is_show_more(text) or wants_more_results(text):
            return True
        if _MORE_TRANSACTIONS.search(text) and len(text.split()) <= 8:
            return True
    return False


def _sale_count_turn(question, conversation_memory=None):
    """How many X were sold in Month Year — a count, not a sales list."""
    if not _SALE_COUNT_ASK.search(question or ""):
        return None
    from app.retrieval.propai_catalog import count_sales

    query = _sale_query_from_question(question) or {}
    place = query.get("place")
    bed = query.get("bedrooms")
    month = query.get("month")
    year = query.get("year")
    stats = count_sales(place=place, bedrooms=bed, year=year, month=month)
    n = int(stats.get("count") or 0)
    where = place or "Dubai"
    bed_bit = f"{_bed_label(bed)} " if bed is not None else ""
    when = ""
    if month and year:
        when = f" in {datetime(year, month, 1).strftime('%B %Y')}"
    elif year:
        when = f" in {year}"
    if n == 0:
        answer = (
            f"I found no recorded {bed_bit}sales in {where}{when}."
        )
    else:
        answer = (
            f"There were **{n:,}** recorded {bed_bit}sales in {where}{when}. "
            "Say **show me** if you want the list."
        )
    _remember_sale_query(
        conversation_memory,
        {
            **query,
            "action": "sales_count",
            "bedrooms": bed,
            "count": n,
            "source_question": question,
            "offset": 0,
        },
    )
    # Keep transaction paging ready for the follow-up list.
    if conversation_memory is not None:
        conversation_memory.transaction_names = _sales_fingerprint(
            place, bed, year, month
        )
        conversation_memory.transaction_offset = 0
    return _transaction_reply(question, answer, followup=True)


def _execute_sales_from_state(question, conversation_memory, *, more: bool = False):
    """Run sales_list / sales_count from dialogue_state slots."""
    from app.retrieval.propai_catalog import count_sales, latest_sales

    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state()
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    # Plain test doubles / legacy sale_query mirror.
    prior = _prior_sale_query(conversation_memory) or {}
    action = state.get("action") or (
        "sales_list" if prior else None
    )
    place = state.get("place") or prior.get("place")
    bed = state.get("beds")
    if bed is None:
        bed = prior.get("bedrooms")
    year = state.get("year") if state.get("year") is not None else prior.get("year")
    month = state.get("month") if state.get("month") is not None else prior.get("month")
    if action == "sales_count" and not more:
        stats = count_sales(place=place, bedrooms=bed, year=year, month=month)
        n = int(stats.get("count") or 0)
        where = place or "Dubai"
        bed_bit = f"{_bed_label(bed)} " if bed is not None else ""
        when = ""
        if month and year:
            when = f" in {datetime(year, month, 1).strftime('%B %Y')}"
        elif year:
            when = f" in {year}"
        answer = (
            f"There were **{n:,}** recorded {bed_bit}sales in {where}{when}. "
            "Say **show me** if you want the list."
            if n else
            f"I found no recorded {bed_bit}sales in {where}{when}."
        )
        _remember_sale_query(
            conversation_memory,
            {
                "action": "sales_count",
                "place": place,
                "bedrooms": bed,
                "month": month,
                "year": year,
                "count": n,
                "source_question": state.get("source_question") or question,
                "offset": 0,
            },
        )
        if conversation_memory is not None:
            conversation_memory.transaction_names = _sales_fingerprint(
                place, bed, year, month
            )
            conversation_memory.transaction_offset = 0
        return _transaction_reply(question, answer, followup=True)

    fingerprint = _sales_fingerprint(place, bed, year, month)
    prev_fp = getattr(conversation_memory, "transaction_names", None)
    prev_offset = int(getattr(conversation_memory, "transaction_offset", 0) or 0)
    offset = prev_offset if more and prev_fp == fingerprint else 0
    limit = _sale_list_limit(question, state)
    payload = latest_sales(
        place=place,
        limit=limit,
        offset=offset,
        bedrooms=bed,
        year=year,
        month=month,
    )
    names = list(dict.fromkeys(
        str(sale.get("project") or "").strip()
        for sale in (payload.get("sales") or [])
        if sale.get("project")
    ))
    intro = place or "Dubai"
    if bed == 0:
        intro = f"studio sales in {intro}"
    elif bed is not None:
        intro = f"{bed}-bedroom sales in {intro}"
    if month and year:
        intro = f"{intro} in {datetime(year, month, 1).strftime('%B %Y')}"
    if state.get("limit") or _parse_sale_limit(question) or _parse_sale_limit(
        state.get("source_question")
    ):
        intro = f"the last {limit} {intro}"
    if conversation_memory is not None:
        conversation_memory.transaction_names = fingerprint
        next_offset = offset + limit if payload.get("sales") else offset
        conversation_memory.transaction_offset = next_offset
        count = None
        if isinstance(state.get("last_result"), dict):
            count = state["last_result"].get("count")
        _remember_sale_query(
            conversation_memory,
            {
                "action": "sales_list",
                "listed": True,
                "place": place,
                "bedrooms": bed,
                "month": month,
                "year": year,
                "limit": limit,
                "source_question": state.get("source_question") or question,
                "offset": next_offset,
                "count": count,
            },
            replace=False,
        )
    if more and offset > 0 and not (payload.get("sales") or []):
        return _transaction_reply(
            question,
            f"There are no more recorded sales for {intro} in the transaction data.",
            followup=False,
            conversation_memory=conversation_memory,
        )
    return _transaction_reply(
        question,
        _format_transactions(names, payload, label=intro, more=more and offset > 0),
        payload.get("sales") or [],
        conversation_memory=conversation_memory,
    )


def _applied_filters(payload, beds, place):
    """The filters a project search actually used, in words (ACOT QA R07)."""
    return [b for b in (
        "ready only" if payload.get("ready_only") else None,
        _bed_label(beds) if beds is not None else None,
        "villa / townhouse" if payload.get("property_types") else None,
        ("developer " + " or ".join(d.title() for d in payload["developers"])) if payload.get("developers") else None,
        f"in {place}" if place and str(place).strip().lower() != "dubai" else None,
        f"under AED {int(payload['max_price']):,}" if payload.get("max_price") is not None else None,
        f"above AED {int(payload['min_price']):,}" if payload.get("min_price") is not None else None,
    ) if b]


def _relax_offer(payload, beds, place):
    """When filters exclude everything, name the ones that can be relaxed instead of a dead end (ACOT QA R07)."""
    options = [opt for cond, opt in (
        (payload.get("developers"), "include other developers"),
        (payload.get("property_types"), "include apartments"),
        (beds is not None, "change the number of bedrooms"),
        (payload.get("max_price") is not None, "raise the budget"),
        (payload.get("ready_only"), "include off-plan"),
        (place and str(place).strip().lower() != "dubai", "look in nearby communities"),
    ) if cond]
    if not options:
        return ""
    joined = options[0] if len(options) == 1 else ", ".join(options[:-1]) + " or " + options[-1]
    return f"\n\nTo widen the search I can relax one filter: {joined}. Which would you like me to change?"


def _execute_search_from_state(question, conversation_memory, *, more: bool = False):
    """Re-run project browse from dialogue_state slots."""
    from app.retrieval.propai_catalog import browse_projects_near_budget

    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state()
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    place = state.get("place")
    beds = state.get("beds")
    budget = state.get("budget_aed")
    payload: dict = {}
    if place and str(place).strip().lower() != "dubai":
        payload["communities"] = [place]
    if beds is not None:
        payload["bedrooms"] = beds
    from app.retrieval.propai_catalog import developers_in_question

    developers = developers_in_question(f"{state.get('source_question') or ''} {question or ''}")
    if developers:
        payload["developers"] = developers
    if _VILLA_ASK.search(f"{state.get('source_question') or ''} {question or ''}"):
        payload["property_types"] = ["villa"]
    if budget is not None:
        if state.get("prefer_cheaper") or not re.search(
            r"\b(?:above|over|starting)\b", str(state.get("source_question") or ""), re.I
        ):
            payload["max_price"] = budget
        else:
            payload["min_price"] = budget
    if state.get("ready_only"):
        payload["ready_only"] = True
    skip = list(state.get("exclude_names") or [])
    if more:
        items = getattr(conversation_memory, "active_candidates", None) or []
        skip = list(dict.fromkeys(skip + [
            str(item.get("name") or "").strip()
            for item in items
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]))
    sort = "more" if skip else (
        "cheapest" if payload.get("max_price") is not None or state.get("prefer_cheaper")
        else "priciest" if payload.get("min_price") is not None
        else "latest_handover"
    )

    try:
        rows, total, budget_meta = browse_projects_near_budget(
            payload, limit=10, sort=sort, skip_names=skip or None
        )
    except Exception as exc:
        print(f"dialogue-state search unavailable: {exc}")
        return None
    if not rows:
        where = place or "Dubai"
        return {
            "question": question,
            "standalone_question": question,
            "question_type": "search",
            "structured_summary": _empty_structured_summary(),
            "documents": [],
            "investment_analysis": None,
            "retrieval_result": {
                "question_type": "search",
                "structured_data": {"community": [], "projects": [], "sub_communities": []},
                "documents": [],
            },
            "use_fast": True,
            "fast_answer": (
                f"I found no more matching projects for {where} "
                "with the same filters."
                if more
                else (
                    f"I don't have projects for {where} with those exact filters ("
                    + (", ".join(_applied_filters(payload, beds, place)) or "no filters")
                    + ") in the last 12 months of recorded sales."
                    + (_relax_offer(payload, beds, place) or " Try a nearby area, a higher budget, or fewer filters.")
                )
            ),
            "context": None,
            "small_talk": None,
        }
    summary = _empty_structured_summary()
    summary["projects"] = rows
    summary["catalog_place"] = str(place or "Dubai").strip()
    asked = f"{state.get('source_question') or ''} {question or ''}"
    summary["applied_filters"] = _applied_filters(payload, beds, place)
    summary["split_launches"] = bool(payload.get("developers") or _SPLIT_ASK.search(asked))
    if _CONSTRAINT_ASK.search(question or ""):
        summary["constraint_note"] = ("**Filters kept for this search:** "
                                      + (", ".join(summary["applied_filters"]) or "none") + ". Anything not listed was not carried over.")
    if budget_meta and budget_meta.get("budget_miss"):
        summary["budget_miss"] = True
        summary["asked_max_price"] = budget_meta.get("asked_max_price")
        summary["lowest_price"] = budget_meta.get("lowest_price")
        # Show the related page only — keep the miss explanation, not "I found N".
        summary["project_total"] = len(rows)
    else:
        summary["project_total"] = total or len(rows)
    shown = skip + [str(row.get("name") or "").strip() for row in rows if row.get("name")]
    _set_dialogue_state(
        conversation_memory,
        {
            **state,
            "action": "search",
            "exclude_names": [name for name in shown if name][:40],
            "last_result": {"project_total": summary["project_total"]},
            "source_question": state.get("source_question") or question,
        },
        replace=True,
    )
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
                "projects": rows,
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": _build_fast_structured_answer(question, summary),
        "context": None,
        "small_talk": None,
    }


def _explain_recommend_from_state(question, conversation_memory):
    """Explain on-screen recommend picks from state — no fresh SQL search."""
    from app.answers.recommend import format_explain_on_screen, rank_projects

    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state()
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    items = getattr(conversation_memory, "active_candidates", None) or []
    rows = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    if len(rows) < 2:
        return None
    ranked = rank_projects(
        rows[:12],
        wanted_beds=state.get("beds"),
        budget_aed=state.get("budget_aed"),
        attach_sales=True,
        goal=state.get("goal"),
    )
    answer = format_explain_on_screen(
        ranked,
        goal=state.get("goal"),
        place=state.get("place"),
        beds=state.get("beds"),
    )
    summary = _empty_structured_summary()
    ordered = []
    by_name = {str(row.get("name") or "").lower(): row for row in rows}
    for item in ranked.get("ranked") or []:
        row = by_name.get(str(item.get("name") or "").lower())
        if row and row not in ordered:
            ordered.append(row)
    summary["projects"] = (ordered or rows)[:3]
    summary["project_total"] = len(summary["projects"])
    summary["recommendation"] = ranked.get("recommendation")
    if state.get("place"):
        summary["catalog_place"] = str(state["place"]).strip()
    return {
        "question": question,
        "standalone_question": state.get("source_question") or question,
        "question_type": "search",
        "structured_summary": summary,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "search",
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


def _execute_recommend_from_state(
    question, conversation_memory, *, more: bool = False, mode: str | None = None
):
    """Re-run or explain recommend from dialogue_state slots (Phase 3)."""
    from app.answers.intent_route import classify_intent, goal_for_route
    from app.answers.intent_turns import recommend_data_turn

    if mode == "explain":
        explained = _explain_recommend_from_state(question, conversation_memory)
        if explained is not None:
            return explained
    if hasattr(conversation_memory, "get_dialogue_state"):
        state = conversation_memory.get_dialogue_state()
    else:
        state = getattr(conversation_memory, "dialogue_state", None) or {}
    prior = state.get("source_question") or question
    state_kind = str(state.get("kind") or "").lower() or None
    # After a split, user picked studios/villas → deepen that path, do not
    # re-classify the original conflict question as split again.
    if state_kind == "filtered" and state.get("beds") is not None:
        beds = state.get("beds")
        route = {
            "intent": "recommend",
            "kind": "filtered",
            "subject": state.get("place"),
            "bedrooms": beds,
            "budget_aed": state.get("budget_aed"),
            "source_question": prior,
            # Path choice after a split: studio vs villa product.
            "property_type": (
                "apartment" if beds == 0
                else ("villa" if isinstance(beds, int) and beds >= 3 else None)
            ),
        }
    elif state_kind == "split":
        route = {
            "intent": "recommend",
            "kind": "split",
            "subject": state.get("place"),
            "bedrooms": state.get("beds"),
            "budget_aed": state.get("budget_aed"),
            "source_question": prior,
        }
    else:
        route = classify_intent(prior, None)
        if route.get("intent") != "recommend":
            route = {
                "intent": "recommend",
                "kind": state_kind or "filtered",
                "subject": state.get("place"),
                "bedrooms": state.get("beds"),
                "budget_aed": state.get("budget_aed"),
                "source_question": prior,
            }
        else:
            route = dict(route)
            if state.get("place"):
                route["subject"] = state.get("place")
            if state.get("beds") is not None:
                route["bedrooms"] = state.get("beds")
            if state.get("budget_aed") is not None:
                route["budget_aed"] = state.get("budget_aed")
            if state_kind:
                route["kind"] = state_kind
            route["source_question"] = prior
    if more:
        route["more"] = True
    if state.get("prefer_cheaper"):
        route["prefer_cheaper"] = True
    if state.get("ready_only"):
        route["ready_only"] = True
    if isinstance(state.get("focus"), dict):
        route["focus"] = dict(state["focus"])
    # Latest user text can refine family ranking (kids / not towers).
    if str(route.get("kind") or state_kind or "").lower() == "family_areas":
        from app.answers.intent_route import merge_family_area_focus

        route["focus"] = merge_family_area_focus(question, route)
        asked = str(question or "").strip()
        if asked and asked.lower() != str(prior or "").strip().lower():
            from app.answers.intent_route import family_area_focus

            if any(family_area_focus(asked).values()):
                route["source_question"] = asked
    route.setdefault("goal", goal_for_route(route) or state.get("goal"))
    return recommend_data_turn(question, conversation_memory, route)


def _execute_dialogue_state(
    question, conversation_memory, *, more: bool = False, mode: str | None = None
):
    """Dispatch the active dialogue_state.action."""
    action = _dialogue_action(conversation_memory)
    if action in {"sales_list", "sales_count"}:
        return _execute_sales_from_state(question, conversation_memory, more=more)
    if action in {"search", "project_count"}:
        # project_count + "show me" → list Dubai projects (same filters/slots).
        if mode == "explain":
            explained = _explain_recommend_from_state(question, conversation_memory)
            if explained is not None:
                return explained
        return _execute_search_from_state(question, conversation_memory, more=more)
    if action == "recommend":
        return _execute_recommend_from_state(
            question, conversation_memory, more=more, mode=mode
        )
    return None


def _show_prior_sales_turn(question, conversation_memory):
    """Legacy bridge: phrase continue → execute from state (Phase 2 prefers resolver)."""
    if not _is_show_prior_sales_ask(question, conversation_memory):
        return None
    if not _prior_sale_query(conversation_memory):
        return None
    more = bool(
        is_show_more(question)
        or wants_more_results(question)
        or (
            _MORE_TRANSACTIONS.search(question or "")
            and not _SHOW_PRIOR_SALES.search(question or "")
        )
    )
    return _execute_sales_from_state(question, conversation_memory, more=more)


def _llm_generate_from(conversation_memory, components=None):
    """Shared LLM generate callback for turn/follow-up resolve."""
    client = getattr(conversation_memory, "llm_client", None) if conversation_memory else None
    if client is None and isinstance(components, dict):
        planner = components.get("query_planner")
        client = getattr(planner, "_llm_client", None) if planner else None
    if client is not None and hasattr(client, "generate"):
        return client.generate
    return None


_PROJECT_TURN_SKIP = re.compile(r"\b(?:compar\w*|vs\.?|versus|agents?|brokers?|charts?|graphs?)\b", re.I)
_AMENITY_ASK = re.compile(r"\b(?:amenit\w*|facilit\w*|features?)\b", re.I)
_ON_SCREEN_WORDS = re.compile(r"\b(?:these|those|them|their|each|all|this|it|its)\b", re.I)

_PROJECT_FACTS_PROMPT = """You are ACOT, a Dubai property assistant. Answer the buyer's question about {subject} \
using ONLY the facts below (Prop-AI project details from the developer, and Dubai Land Department sales and \
rent registrations).
- Answer exactly what was asked first, in one or two sentences, then a short bulleted list of the most relevant facts.
- "Starting price" means the lowest layout starting_price_from (developer price). Also give the 12-month median \
recorded sale when present, labelled as recorded sales.
- Features or amenities: list the amenities and highlights as given; mention project_details points when useful.
- Sales or rent questions: use the sales and rents sections (AED, dates as Month YYYY, newest first).
- If what was asked is not in the facts, say it is not on record for this project. Never invent facts.
- Markdown bullets, no headings, AED amounts with thousands separators, under 220 words.

Facts (JSON):
{facts}

Question: {question}"""


async def _named_project_turn(question, conversation_memory, components=None):
    """A question that names one project, or asks amenities of the projects on screen.

    Runs before the on-screen / follow-up handlers so "starting price of Serenz by Danube" answers
    Serenz instead of re-listing the last browse. Facts: Prop-AI details (amenities, layouts,
    payment plan) plus that project's own sales and rents.
    """
    from app.retrieval.hybrid.question_text import comparison_pair
    from app.retrieval.propai_catalog import _id_index, project_fact_sheet, project_in_question

    text = question or ""
    if _PROJECT_TURN_SKIP.search(text) or comparison_pair(text):
        return None
    llm_generate = _llm_generate_from(conversation_memory, components)
    if llm_generate is None:
        return None
    project = await asyncio.to_thread(project_in_question, text)
    if project:
        targets = [(project["name"], project["key"])]
    elif _AMENITY_ASK.search(text) and _ON_SCREEN_WORDS.search(text):
        ids = await asyncio.to_thread(_id_index)
        targets = [
            (item["name"], ids.get(str(item.get("id") or "")))
            for item in _active_projects(conversation_memory)[:8]
        ]
        targets = [(name, key) for name, key in targets if key]
        if not targets:
            return None
    else:
        return None

    facts = {}
    for name, key in targets:
        sheet = await asyncio.to_thread(project_fact_sheet, key)
        if len(targets) > 1:
            sheet = {k: sheet[k] for k in ("executive_summary", "amenities", "highlights", "project_details") if k in sheet}
        facts[name] = sheet or {"note": "nothing on record"}
    subject = targets[0][0] if len(targets) == 1 else "the projects on screen"
    prompt = _PROJECT_FACTS_PROMPT.format(
        subject=subject, question=text.strip(),
        facts=json.dumps(facts, default=str, ensure_ascii=False)[:24000],
    )
    try:
        answer = (await llm_generate(prompt) or "").strip()
    except Exception as exc:
        print(f"named project answer failed: {exc}")
        return None
    if not answer:
        return None
    if len(targets) == 1:
        turn = await asyncio.to_thread(_project_detail_turn, text, conversation_memory, targets[0][0])
        if turn is None:
            turn = _fast_answer(text, answer, question_type="search")
        turn["fast_answer"] = answer
        return turn
    return _fast_answer(text, answer, projects=_active_projects(conversation_memory)[:8], question_type="search")


async def _dialogue_followup_turn(question, conversation_memory, components=None):
    """Phase 2: continue the active task from dialogue_state (LLM resolve + execute)."""
    from app.answers.followup_resolve import (
        apply_followup,
        looks_like_state_followup,
        resolve_followup,
    )

    if conversation_memory is None:
        return None
    # Casual speech never continues a sticky recommend/search task.
    try:
        from app.answers.turn_act import is_casual_speech

        if is_casual_speech(question):
            return None
    except Exception:
        pass
    # Compare / best-of-shown about live cards is a new turn — never sticky afford.
    if _on_screen_compare_or_pick(question, conversation_memory):
        return None
    state = (
        conversation_memory.get_dialogue_state()
        if hasattr(conversation_memory, "get_dialogue_state")
        else getattr(conversation_memory, "dialogue_state", None)
    )
    if not looks_like_state_followup(question, state):
        return None
    # Intent-shifting chips/asks after another task are new turns — never a
    # sticky search/recommend patch (e.g. agents chip → "200 projects in JVC").
    prior_action = str((state or {}).get("action") or "")
    asked = question or ""
    if re.search(r"\b(?:agents?|brokers?)\b", asked, re.I) and prior_action not in {
        "agents", "",
    }:
        return None
    if re.search(
        r"\b(?:transactions?|transcations?|(?:recent|latest)\s+sales?|sale history)\b",
        asked,
        re.I,
    ) and prior_action not in {"sales_list", "sales_count", "sales_stat", ""}:
        return None

    resolve_started = time.perf_counter()
    llm_generate = _llm_generate_from(conversation_memory, components)
    resolution = await resolve_followup(
        question, state, llm_generate=llm_generate
    )
    resolve_ms = _elapsed_ms(resolve_started)
    if not resolution or resolution.get("kind") != "continue":
        return None
    if (
        resolution.get("mode") == "rerun"
        and not resolution.get("more")
        and not resolution.get("patch")
        and str(resolution.get("action") or prior_action) == prior_action
    ):
        # Same task, nothing changed: running it again only repeats the last answer.
        return None
    new_state = apply_followup(state, resolution)
    if not new_state:
        return None
    if hasattr(conversation_memory, "set_dialogue_state"):
        conversation_memory.set_dialogue_state(new_state)
    else:
        conversation_memory.dialogue_state = new_state
    turn = await asyncio.to_thread(
        _execute_dialogue_state,
        question,
        conversation_memory,
        more=bool(resolution.get("more")),
        mode=resolution.get("mode"),
    )
    if isinstance(turn, dict):
        timing = dict(turn.get("timing_ms") or {})
        timing["followup_resolve"] = resolve_ms
        turn["timing_ms"] = timing
    return turn


def _route_from_turn_resolution(resolution: dict, question: str | None = None) -> dict:
    """Build a recommend route dict from Phase 5 turn resolution."""
    from app.answers.intent_route import (
        goal_for_route,
        merge_family_area_focus,
    )

    if resolution.get("route") and isinstance(resolution["route"], dict):
        route = dict(resolution["route"])
    else:
        route = {
            "intent": "recommend",
            "kind": resolution.get("kind") or "filtered",
            "subject": resolution.get("place"),
            "bedrooms": resolution.get("beds"),
            "budget_aed": resolution.get("budget_aed"),
            "source_question": resolution.get("source_question"),
        }
    route.setdefault("intent", "recommend")
    if resolution.get("place"):
        route["subject"] = resolution["place"]
    if resolution.get("beds") is not None:
        route["bedrooms"] = resolution["beds"]
    if resolution.get("budget_aed") is not None:
        route["budget_aed"] = resolution["budget_aed"]
    if resolution.get("kind"):
        route["kind"] = resolution["kind"]
    route["source_question"] = (
        resolution.get("source_question") or route.get("source_question")
    )
    route["goal"] = resolution.get("goal") or goal_for_route(route)
    # LLM resolve omits focus — attach family ranking flags from the user text.
    if str(route.get("kind") or "").lower() == "family_areas":
        if isinstance(resolution.get("focus"), dict):
            route["focus"] = resolution["focus"]
        route["focus"] = merge_family_area_focus(
            question or route.get("source_question"),
            route,
        )
    return route


async def _understood_turn(question, conversation_memory, components=None):
    """Phase 5–7: understand → confidence gate → state → execute.

    Confident search / sales / project_count / recommend run on SQL/tools.
    Weak understanding clarifies once instead of guessing.
    Other actions fall through to specialized paths (agents, predict, agent LLM).
    """
    from app.answers.clarify import clarify_turn_payload, should_clarify
    from app.answers.turn_resolve import resolve_turn, state_updates_from_resolution

    # Live cards + compare/best beat sticky afford re-runs from LLM resolve.
    compare_early = await asyncio.to_thread(
        _compare_shown_turn, question, conversation_memory
    )
    if compare_early is not None:
        return compare_early
    shortlist_early = await asyncio.to_thread(
        _shortlist_recommend_turn, question, conversation_memory
    )
    if shortlist_early is not None:
        return shortlist_early

    # Two areas ("Compare JVC and Dubai Marina", "Business Bay or Downtown?") on the same market numbers.
    from app.answers.area_comparison import parse_area_comparison

    area_pair = parse_area_comparison(question)
    if area_pair:
        return await asyncio.to_thread(_area_pair_turn, question, *area_pair)

    # Named project compares must fall through to hybrid retrieval — never
    # clarify for area/beds/budget ("Compare Serenz and Aura Heights").
    try:
        from app.retrieval.hybrid.question_text import comparison_pair

        if comparison_pair(question):
            return None
    except Exception:
        pass

    resolve_started = time.perf_counter()
    llm_generate = _llm_generate_from(conversation_memory, components)
    resolution = await resolve_turn(
        question, conversation_memory, llm_generate=llm_generate
    )
    resolve_ms = _elapsed_ms(resolve_started)
    if not resolution:
        return None

    def _stamp_resolve(turn):
        if not isinstance(turn, dict):
            return turn
        timing = dict(turn.get("timing_ms") or {})
        timing["turn_resolve"] = resolve_ms
        timing.setdefault("resolve_source", resolution.get("source"))
        turn["timing_ms"] = timing
        return turn

    if components is not None:
        # Later fallbacks read this: a guidance/general question must reach the agent, not a project list.
        components["_resolved_action"] = resolution.get("action")
    if resolution.get("action") == "handovers":
        return _stamp_resolve(
            await asyncio.to_thread(_handover_turn, question, resolution, conversation_memory)
        )
    if resolution.get("action") == "guidance":
        # Rules, fees, visas, restaurants, metro: not in our records, so the web answers (with sources).
        # No web (search off or failed): the agent's general guidance answers as before.
        web = await asyncio.to_thread(_web_answer_turn, question, resolution)
        if web is not None:
            return _stamp_resolve(web)
    direct = {
        "predict": _predict_turn,
        "rent_stat": _rent_stat_turn, "area_market": _area_market_turn,
        "area_rank": _area_rank_turn, "top_projects": _top_projects_turn,
    }.get(resolution.get("action"))
    if direct is not None:
        turn = await asyncio.to_thread(direct, question, resolution)
        if turn is not None:
            return _stamp_resolve(turn)
    if resolution.get("action") == "sales_count" and str(resolution.get("place") or "dubai").strip().lower() == "dubai" \
            and (resolution.get("month") or resolution.get("year")):
        # "How many sold in Dubai in October?" is a citywide count, not a reason to ask for an area.
        from datetime import date
        from app.retrieval.propai_catalog import count_sales

        month, year = resolution.get("month"), resolution.get("year") or date.today().year
        counted = await asyncio.to_thread(count_sales, None, resolution.get("beds"), year, month)
        when = date(year, month, 1).strftime("%B %Y") if month else str(year)
        return _stamp_resolve(_fast_answer(
            question,
            f"There were **{counted['count']:,}** recorded {_beds_label(resolution.get('beds'))}property sales in "
            f"Dubai in {when} (all property types, Dubai Land Department records).",
        ))
    if resolution.get("action") == "project_detail" and resolution.get("project"):
        turn = await asyncio.to_thread(
            _project_detail_turn, question, conversation_memory, resolution["project"]
        )
        if turn is not None:
            return _stamp_resolve(turn)

    if resolution.get("action") == "data_freshness":
        fresh = await asyncio.to_thread(_data_freshness_turn, question, True)
        if fresh is not None:
            return _stamp_resolve(fresh)

    # Aggregates (cheapest / highest / average sale) go to the exact SQL handler.
    if resolution.get("action") == "sales_stat":
        stat_slots = {
            "stat": resolution.get("stat"),
            "project": resolution.get("project"),
            "place": resolution.get("place"),
            "beds": resolution.get("beds"),
            "months": resolution.get("months"),
        }
        stat_turn = await asyncio.to_thread(
            _cheapest_or_stat_turn, question, conversation_memory, stat_slots
        )
        if stat_turn is not None:
            return _stamp_resolve(stat_turn)

    # On-screen project cards beat a place-less sales clarify ("transaction details").
    # The LLM may label a bare "transaction details" as clarify/general; the
    # on-screen project still answers it.
    if resolution.get("action") in {"sales_list", "clarify", "general", "search", None} and (
        _TRANSACTION.search(question or "")
    ):
        on_screen = _transaction_subject(question, conversation_memory)
        if on_screen:
            return _stamp_resolve(
                await asyncio.to_thread(
                    _transaction_turn, question, conversation_memory
                )
            )

    # Live project cards / deixis give agents a place before the clarify gate.
    if resolution.get("action") == "agents" and not resolution.get("place"):
        live_place = _broker_place(question, conversation_memory)
        if live_place:
            from app.answers.clarify import score_confidence

            resolution = {**resolution, "place": live_place}
            resolution["confidence"] = score_confidence(resolution, question)

    # Attribute chips about on-screen projects beat a wrong clarify/search.
    # ("Who is the developer of each of these projects?")
    if resolution.get("action") in {"attribute", "clarify", "search", "general"}:
        shown_early = _shown_attribute_response(
            question, conversation_memory, {}, time.perf_counter(), None
        )
        if shown_early is not None:
            return _stamp_resolve(shown_early)

    # Phase 7: ask before guessing when slots/confidence are weak.
    if should_clarify(resolution, question):
        updates = state_updates_from_resolution(
            {**resolution, "action": "clarify"}
        )
        _set_dialogue_state(conversation_memory, updates, replace=True)
        return _stamp_resolve(clarify_turn_payload(question, resolution))

    action = resolution.get("action")
    # Leave these to existing specialized turns / agent.
    if action in {
        None, "general", "small_talk", "predict", "guidance",
    }:
        return None

    updates = state_updates_from_resolution(resolution)

    # Top agents / brokers for a community — never fall through to project browse.
    if action == "agents":
        _set_dialogue_state(conversation_memory, updates, replace=True)
        return _stamp_resolve(
            await asyncio.to_thread(_broker_turn, question, conversation_memory)
        )

    # Developer / price / handover of the projects already on screen (or last sales).
    if action == "attribute":
        shown = _shown_attribute_response(
            question, conversation_memory, {}, time.perf_counter(), None
        )
        if shown is not None:
            return _stamp_resolve(shown)
        return None

    if action == "developer_rank":
        return _stamp_resolve(
            await asyncio.to_thread(_developer_rank_turn, question)
        )

    place = str(resolution.get("place") or "").strip().lower()
    # Every project count (per area, per developer, off-plan / ready) uses the same counter, so wordings agree.
    if action == "project_count" and (
        place not in {"", "dubai"} or resolution.get("developer") or resolution.get("kind") in {"off_plan", "ready"}
    ):
        from app.retrieval.propai_catalog import count_projects

        status = resolution.get("kind")
        n = await asyncio.to_thread(count_projects, resolution.get("place"), resolution.get("developer"), status)
        label = {"off_plan": "off-plan ", "ready": "ready (completed) "}.get(status, "")
        by = f" by {resolution['developer']}" if resolution.get("developer") else ""
        _set_dialogue_state(conversation_memory, {**updates, "action": "project_count"}, replace=True)
        return _stamp_resolve(_fast_answer(
            question,
            f"There are **{n:,}** {label}residential projects{by}{_where(resolution.get('place'))} in our "
            "Dubai Land Department project records. Say **show me** if you want the list.",
        ))
    if action == "project_count" and place in {"", "dubai"}:
        return _stamp_resolve(
            await asyncio.to_thread(
                _project_count_turn, question, conversation_memory, True
            )
        )
    # A count in one community ("how many projects in Dubai Hills") runs the search
    # executor below, which answers "I found N projects in X" for every wording.

    if action == "recommend":
        from app.answers.intent_turns import recommend_data_turn

        route = _route_from_turn_resolution(resolution, question=question)
        if route.get("focus"):
            updates = {**updates, "focus": route["focus"]}
        _set_dialogue_state(conversation_memory, updates, replace=True)
        turn = await asyncio.to_thread(
            recommend_data_turn, question, conversation_memory, route
        )
        return _stamp_resolve(turn)

    if action in {"search", "project_count", "sales_list", "sales_count"}:
        # After a project shortlist, "recent transaction details" must use the
        # on-screen cards — not area-wide sales from an inherited place.
        if action == "sales_list":
            on_screen = _transaction_subject(question, conversation_memory)
            if on_screen:
                return _stamp_resolve(
                    await asyncio.to_thread(
                        _transaction_turn, question, conversation_memory
                    )
                )
        # Persist last-N so follow-ups ("same … 2 bedrooms") keep that count.
        sale_limit = _parse_sale_limit(question)
        if sale_limit is not None:
            updates = {**updates, "limit": sale_limit}
        _set_dialogue_state(conversation_memory, updates, replace=True)
        if action == "sales_count":
            # Keep listed=False semantics via action sales_count on state.
            _set_dialogue_state(
                conversation_memory,
                {"action": "sales_count"},
                replace=False,
            )
        elif action == "sales_list":
            extra = {"action": "sales_list"}
            if sale_limit is not None:
                extra["limit"] = sale_limit
            _set_dialogue_state(
                conversation_memory,
                extra,
                replace=False,
            )
        return _stamp_resolve(
            await asyncio.to_thread(
                _execute_dialogue_state,
                question,
                conversation_memory,
                more=False,
                mode="rerun",
            )
        )
    return None


def _sale_stat_kind(question) -> str | None:
    """Map repaired text to a price-extreme kind. Context decides if it runs."""
    try:
        text = repair_typed_question(question or "")
    except Exception:
        text = question or ""
    # Prefer shared understanding helper when available.
    try:
        from app.answers.turn_resolve import _stat_kind_from_text

        return _stat_kind_from_text(text)
    except Exception:
        pass
    if _CHEAPEST_SALE_ASK.search(text):
        return "cheapest"
    if _HIGHEST_SALE_ASK.search(text):
        return "highest"
    if _AVERAGE_SALE_ASK.search(text):
        return "average"
    return None


def _cheapest_or_stat_turn(question, conversation_memory=None, slots=None):
    """Exact aggregate over recorded sales (cheapest / highest / average).

    `slots` is the understanding layer's output (stat / project / place / beds /
    months). When it is present the question wording does not matter.
    """
    question = repair_typed_question(question)
    return _cheapest_sale_turn(
        question, slots=slots, conversation_memory=conversation_memory
    )


def _on_screen_sale_extreme(projects, *, kind: str, bedrooms=None, months=12):
    """Lowest/highest recorded sale among the projects already on screen."""
    from app.retrieval.propai_catalog import cheapest_sale

    best = None
    for item in projects or []:
        name = str((item or {}).get("name") or "").strip()
        if not name:
            continue
        try:
            sale = cheapest_sale(
                project=name,
                bedrooms=bedrooms,
                months=months,
                highest=(kind == "highest"),
            )
        except Exception:
            sale = None
        if not sale or sale.get("price") is None:
            continue
        if best is None:
            best = sale
            continue
        price = int(sale["price"])
        champ = int(best["price"])
        if kind == "highest" and price > champ:
            best = sale
        elif kind != "highest" and price < champ:
            best = sale
    return best


def _cheapest_sale_turn(question, slots=None, conversation_memory=None):
    """Cheapest / highest / average sale for a named project or place — not a list.

    Slots from the LLM resolver win; phrase regexes are only the offline fallback.
    After a sales list, "which one has cheapest transactions" compares on-screen
    projects and names a clear winner.
    """
    slots = slots if isinstance(slots, dict) else {}
    # Slots from the LLM win. Regex is only used when slots did not say.
    kind = slots.get("stat") or _sale_stat_kind(question)
    if kind is None and slots:
        # Understood as sales_stat but model omitted stat → cheapest is the
        # usual meaning of a "what would I pay / lowest" ask.
        kind = "cheapest"
    if kind is None:
        return None
    from app.retrieval.propai_catalog import (
        cheapest_sale,
        parse_lookback_months,
        place_named,
        project_named_in_question,
        resolve_dubai_place,
        sale_window_stats,
    )

    # Project: the model's named building, checked against the project table, so
    # any wording works; fall back to finding a project name inside the question.
    project = None
    if slots.get("project"):
        project = project_named_in_question(str(slots["project"]))
    if not project:
        project = project_named_in_question(question)
    place = None
    if not project:
        place = (
            slots.get("place")
            or extract_place(question)
            or place_named(question)
            or _transaction_place(question)
        )
        place = resolve_dubai_place(place) if place else place
    elif slots.get("place"):
        # Keep the area for the evidence line ("... in Hanover Square (JVC)").
        place = resolve_dubai_place(slots["place"]) or slots["place"]
    if slots.get("beds") is not None:
        bed = int(slots["beds"])
    else:
        beds = parse_bedrooms(question)
        bed = beds[0] if beds else None
    months = slots.get("months") or parse_lookback_months(question, default=12)

    on_screen = _active_projects(conversation_memory)
    prior_action = _dialogue_action(conversation_memory)
    among_shown = bool(
        not project
        and len(on_screen) >= 2
        and (
            _WHICH_AMONG_SHOWN.search(question or "")
            or (
                prior_action in {"sales_list", "sales_count", "search", "recommend"}
                and not place
            )
            or (
                prior_action in {"sales_list", "sales_count"}
                and _sale_stat_kind(question) is not None
            )
        )
    )
    if among_shown and kind in {"cheapest", "highest"}:
        sale = _on_screen_sale_extreme(
            on_screen, kind=kind, bedrooms=bed, months=months
        )
        names = [str(row.get("name") or "").strip() for row in on_screen if row.get("name")]
        label = "cheapest" if kind == "cheapest" else "highest"
        bed_bit = f"{_bed_label(bed)} " if bed is not None else ""
        window = f" in the last {months} months" if months else ""
        understood = (
            f"Understood: you asked which of these {len(names)} on-screen projects "
            f"has the {label} recorded {bed_bit}transaction{window}."
        )
        if not sale or sale.get("price") is None:
            return _transaction_reply(
                question,
                f"{understood}\n\n"
                "I could not find a matching recorded sale among these projects "
                f"{window or 'in the current window'}.",
                followup=False,
            )
        size = (
            f", {int(sale['size_sqft']):,} sqft"
            if sale.get("size_sqft")
            else ""
        )
        date = str(sale.get("date") or "")[:10]
        date_bit = f" on {date}" if date else ""
        bed_sale = ""
        if sale.get("bedrooms") == 0:
            bed_sale = " (studio)"
        elif sale.get("bedrooms") is not None:
            bed_sale = f" ({_bed_label(sale.get('bedrooms'))})"
        answer = (
            f"{understood}\n\n"
            f"**Acot Recommend:** **{sale.get('project')}** has the {label} recorded "
            f"{bed_bit}sale among these projects{window}: "
            f"**AED {int(sale['price']):,}**{date_bit}{size}{bed_sale}.\n\n"
            "That is a single Dubai Land Department sale record — not a brochure "
            "starting price. Want the highest sale next, or only studios / 1-beds?"
        )
        return _transaction_reply(question, answer, sales=[sale], followup=False)

    # Inherit sticky place from the sales thread when the ask has no new area.
    if not project and not place and conversation_memory is not None:
        getter = getattr(conversation_memory, "get_dialogue_state", None)
        state = getter() if callable(getter) else getattr(
            conversation_memory, "dialogue_state", None
        )
        if isinstance(state, dict) and state.get("place"):
            place = resolve_dubai_place(state["place"]) or state["place"]

    subject = project or place or "Dubai"
    bed_bit = f"{_bed_label(bed)} " if bed is not None else ""
    window = f" in the last {months} months" if months else ""
    area_bit = f" ({place})" if project and place else ""
    # "average" asks are answered with the median, so say median.
    word = {"cheapest": "lowest", "highest": "highest", "average": "median"}[kind]
    # Explain-first: restate exactly what was searched before the number.
    understood = (
        f"Understood: the {word} recorded {bed_bit}sale price in {subject}{window}."
    )
    try:
        stats = sale_window_stats(
            project=project, place=place, bedrooms=bed, months=months
        )
    except Exception:
        stats = {}
    n = int(stats.get("count") or 0)
    plural = "s" if n != 1 else ""

    if kind == "average":
        if not n or not stats.get("median"):
            return _transaction_reply(
                question,
                f"{understood}\n\nI found no recorded {bed_bit}sales for "
                f"{subject}{window}, so I cannot quote an average.",
                followup=False,
            )
        mean_note = ""
        answer = (
            f"{understood}\n\n"
            f"Across **{n:,}** recorded {bed_bit}sale{plural} in **{subject}**{window}, "
            f"the median price is **AED {int(stats['median']):,}**{mean_note}. "
            f"Prices ranged from AED {int(stats['low']):,} to AED {int(stats['high']):,}.\n\n"
            "Source: Dubai Land Department sale records."
        )
        return _transaction_reply(question, answer, followup=False)

    sale = cheapest_sale(
        project=project,
        place=place,
        bedrooms=bed,
        months=months,
        highest=(kind == "highest"),
    )
    if not sale or sale.get("price") is None:
        return _transaction_reply(
            question,
            f"{understood}\n\n"
            f"I found no recorded {bed_bit}sales for {subject}{window}, "
            f"so there is no {'cheapest' if kind == 'cheapest' else 'highest'} sale to quote. "
            "Try a longer window or a nearby project.",
            followup=False,
        )
    size = (
        f", {int(sale['size_sqft']):,} sqft"
        if sale.get("size_sqft")
        else ""
    )
    date = str(sale.get("date") or "")[:10]
    date_bit = f" on {date}" if date else ""
    label = "cheapest" if kind == "cheapest" else "highest"
    answer = (
        f"{understood}\n\n"
        f"**Acot Recommend:** The {label} recorded {bed_bit}sale in "
        f"**{sale.get('project') or subject}**{window} is "
        f"**AED {int(sale['price']):,}**{date_bit}{size}."
    )
    # Evidence: how many sales this is the extreme of, so the number can be trusted.
    if n:
        evidence = (
            f"That is the {word} of {n:,} recorded {bed_bit}sale{plural} "
            f"in {subject}{area_bit}"
        )
        if stats.get("median") and n > 1:
            other = stats["high"] if kind == "cheapest" else stats["low"]
            evidence += (
                f", where prices ran {'up to' if kind == 'cheapest' else 'down to'} "
                f"AED {int(other):,} (median AED {int(stats['median']):,})"
            )
        answer += f"\n\n{evidence}. Source: Dubai Land Department sale records."
    return _transaction_reply(question, answer, sales=[sale], followup=False)


def _transaction_turn(question, conversation_memory):
    question = repair_typed_question(question)
    # Aggregates (cheapest / highest / average sale) never become a sales list.
    if _sale_stat_kind(question) is not None:
        stat_turn = _cheapest_sale_turn(
            question, conversation_memory=conversation_memory
        )
        if stat_turn is not None:
            return stat_turn
    if not _TRANSACTION.search(question or ""):
        # Short "ok show me" after a count/sales answer — reuse stored filters.
        show_prior = _show_prior_sales_turn(question, conversation_memory)
        if show_prior is not None:
            return show_prior
        # "same give in 2 bedrooms" after sales_list — patch beds, keep place/limit.
        if (
            _dialogue_action(conversation_memory) in {"sales_list", "sales_count"}
            and _SALES_SLOT_TWEAK.search(question or "")
            and (
                parse_bedrooms(question)
                or re.search(r"\b(?:same|again|instead)\b", question or "", re.I)
            )
        ):
            beds = parse_bedrooms(question)
            if beds:
                _set_dialogue_state(
                    conversation_memory,
                    {"beds": beds[0], "action": "sales_list", "offset": 0},
                    replace=False,
                )
            return _execute_sales_from_state(
                question, conversation_memory, more=False
            )
        return None
    count_turn = _sale_count_turn(question, conversation_memory)
    if count_turn is not None:
        return count_turn
    cheap_turn = _cheapest_sale_turn(
        question, conversation_memory=conversation_memory
    )
    if cheap_turn is not None:
        return cheap_turn
    from app.retrieval.propai_catalog import (
        is_outside_dubai,
        latest_sales,
        recent_sales,
        resolve_dubai_place,
    )
    # On-screen project cards beat a stale "show prior sales" continue.
    projects = _transaction_subject(question, conversation_memory)
    if not projects:
        show_prior = _show_prior_sales_turn(question, conversation_memory)
        if show_prior is not None:
            return show_prior
    more = bool(_MORE_TRANSACTIONS.search(question or ""))
    prev_fp = getattr(conversation_memory, "transaction_names", None)
    prev_offset = int(getattr(conversation_memory, "transaction_offset", 0) or 0)
    names = []
    intro = "Dubai"
    payload = None
    fingerprint = None
    if projects:
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
        offset = prev_offset if more and prev_fp == fingerprint else 0
        payload = recent_sales(names, per_project=_TRANSACTION_PAGE_SIZE, offset=offset)
    else:
        if not _TRANSACTION_BROWSE.search(question or ""):
            return None
        # Prefer extract_place / place_named — _transaction_place often keeps
        # trailing words ("JVC, with date and price") and then resolve fails.
        from app.retrieval.propai_catalog import place_named

        asked = (
            extract_place(question)
            or place_named(question)
            or _transaction_place(question)
        )
        listings_on_screen = bool(_active_listings(conversation_memory))
        projects_on_screen = bool(_active_projects(conversation_memory))
        if listings_on_screen and not projects_on_screen and not asked:
            return _transaction_reply(
                question,
                "There are no recorded transactions for these rental listings. "
                "Advertised listings do not include sale history.",
                followup=False,
            )
        if asked and is_outside_dubai(asked):
            return _transaction_reply(
                question,
                "I only have recorded sales in Dubai. "
                "Ask about an area such as Dubai Marina, Business Bay, or Palm Jumeirah.",
            )
        resolved = resolve_dubai_place(asked) if asked else None
        place = resolved
        if asked and not resolved:
            named = recent_sales(asked, per_project=_TRANSACTION_PAGE_SIZE, offset=0)
            if named.get("sales"):
                payload = named
                names = [asked]
                intro = asked
                fingerprint = str(asked).strip().upper()
                offset = 0
        if payload is None:
            beds = parse_bedrooms(question)
            bed = beds[0] if beds else None
            # Carry beds/place/limit from an active sales task when this ask
            # only tweaks a slot ("same give in 2 bedrooms").
            prior = _prior_sale_query(conversation_memory) or {}
            if bed is None and prior.get("bedrooms") is not None:
                bed = prior.get("bedrooms")
            if not place and prior.get("place"):
                place = prior.get("place")
            fingerprint = _sales_fingerprint(place, bed)
            offset = prev_offset if more and prev_fp == fingerprint else 0
            # "last 5 sales" → limit 5; carry prior limit on "same" follow-ups.
            prior_state = (
                conversation_memory.get_dialogue_state()
                if conversation_memory and hasattr(conversation_memory, "get_dialogue_state")
                else getattr(conversation_memory, "dialogue_state", None)
            )
            limit = _sale_list_limit(question, prior_state if isinstance(prior_state, dict) else prior)
            payload = latest_sales(
                place=place, limit=limit, offset=offset, bedrooms=bed
            )
            names = list(dict.fromkeys(
                str(sale.get("project") or "").strip()
                for sale in (payload.get("sales") or [])
                if sale.get("project")
            ))
            intro = place or "Dubai"
            if bed == 0:
                intro = f"studio sales in {intro}"
            elif bed is not None:
                intro = f"{bed}-bedroom sales in {intro}"
            if _parse_sale_limit(question) or (
                isinstance(prior_state, dict) and prior_state.get("limit")
            ) or _parse_sale_limit(prior.get("source_question")):
                intro = f"the last {limit} {intro}"
            _remember_sale_query(
                conversation_memory,
                {
                    "action": "sales_list",
                    "listed": True,
                    "place": place,
                    "bedrooms": bed,
                    "year": None,
                    "month": None,
                    "limit": limit,
                    "source_question": question,
                    "offset": offset,
                },
            )
    if conversation_memory is not None:
        conversation_memory.transaction_names = fingerprint
        page = limit if "limit" in locals() else _TRANSACTION_PAGE_SIZE
        conversation_memory.transaction_offset = (
            offset + page if payload.get("sales") else offset
        )
    # Text only: do not attach project rows, or the app draws property cards under the sales list.
    # Project names are still remembered on memory for "developer of these" follow-ups.
    return _transaction_reply(
        question,
        _format_transactions(names, payload, label=intro, more=more and offset > 0),
        payload.get("sales") or [],
        conversation_memory=conversation_memory,
    )


_BROKER = re.compile(
    r"\b(?:brokers?|agents?|agenc(?:y|ies)|licensed agents?|rera agents?|real[\s-]?estate agents?|top agents?)\b",
    re.I,
)
_BROKER_NATIONALITIES = (
    ("sri lankan", "Sri Lankan"),
    ("pakistani", "Pakistani"),
    ("filipino", "Filipino"),
    ("emirati", "Emirati"),
    ("egyptian", "Egyptian"),
    ("iranian", "Iranian"),
    ("lebanese", "Lebanese"),
    ("british", "British"),
    ("indian", "Indian"),
    ("india", "Indian"),
)
_BROKER_FILTER_WORDS = re.compile(
    r"\b(?:sri lankan|pakistani|pakistan|filipino|emirati|egyptian|iranian|lebanese|"
    r"british|indian|india|nationality|nationalities|gender|female|male|"
    r"women|woman|ladies|lady|men|man|both)\b",
    re.I,
)
# Budget/price clauses are not place names ("Dubai starting above AED 5 million").
_BROKER_BUDGET_TAIL = re.compile(
    r"\b(?:starting(?:\s+(?:above|from|at))?|above|over|under|below|up\s+to|"
    r"around|about|budget|max(?:imum)?|min(?:imum)?|price|prices|"
    r"aed|dh|dhs)\b.*$",
    re.I,
)
_BROKER_BUDGET_TOKENS = re.compile(
    r"\b(?:starting|above|over|under|below|up\s+to|around|about|budget|"
    r"max(?:imum)?|min(?:imum)?|price|prices|aed|dh|dhs|million|millions|"
    r"\d[\d,]*(?:\.\d+)?)\b",
    re.I,
)
_BROKER_FILLER = re.compile(
    r"\b(?:show|me|tell|find|list|the|top|best|licensed|rera|dubai|please|who|are|is|"
    r"can|help|with|this|these|those|property|properties|project|projects|community|"
    r"brokers?|agents?|agenc(?:y|ies)|real[\s-]?estate|for|in|at|of|only|just|from|"
    r"you|your|a|an|i|need|want|get|looking|someone|some|good|any|give|recommend|suggest|could|would|"
    r"details?|info|information|about|"
    r"sri lankan|pakistani|pakistan|filipino|emirati|egyptian|iranian|lebanese|"
    r"british|indian|india|nationality|nationalities|gender|female|male|"
    r"women|woman|ladies|lady|men|man|both)\b",
    re.I,
)
_BROKER_PLACE = re.compile(
    r"\b(?:for|in|at)\s+(.+?)$",
    re.I,
)
_BROKER_POINTER = re.compile(
    r"\b(?:this|these|those|that|them|here|above|shown)\b|"
    r"\b(?:projects?|propert(?:y|ies)|areas?|ones?|options?|listings?|"
    r"communities?|places?)\b",
    re.I,
)


def _strip_broker_budget(text: str | None) -> str:
    """Drop price/budget tails so place stays 'Dubai' / 'Business Bay'."""
    cleaned = _BROKER_BUDGET_TAIL.sub("", str(text or ""))
    return cleaned.strip(" .?,;:'\"")


def _broker_identity_filters(question):
    """Nationality or gender the user asked for. DLD rows we store do not have these fields."""
    text = question or ""
    nationality = None
    for needle, label in _BROKER_NATIONALITIES:
        if re.search(rf"\b{re.escape(needle)}\b", text, re.I):
            nationality = label
            break
    female = bool(re.search(r"\b(?:female|women|woman|ladies|lady)\b", text, re.I))
    male = bool(re.search(r"\b(?:male|men|man)\b", text, re.I))
    if female and male:
        return nationality, None, True
    if female:
        return nationality, "female", False
    if male:
        return nationality, "male", False
    return nationality, None, False


def _broker_identity_unsupported(question) -> bool:
    nationality, gender, _both = _broker_identity_filters(question)
    return bool(nationality or gender)


_BROKER_SWITCH_AWAY = re.compile(
    r"\b(?:projects?|listings?|communities?|developers?|handover|brochure|"
    r"propert(?:y|ies)|apartments?|villas?|townhouses?|penthouses?|"
    r"rentals?|for sale|off[- ]?plan|"
    r"sales?|sold|transactions?|sale history|sales history|price|prices|budget)\b",
    re.I,
)


def _last_asked_about_agents(conversation_memory) -> bool:
    if conversation_memory is None:
        return False
    texts = [
        getattr(conversation_memory, "last_user_question", None),
        getattr(conversation_memory, "last_standalone_question", None),
    ]
    for turn in reversed((getattr(conversation_memory, "turns", None) or [])[-4:]):
        if isinstance(turn, dict):
            texts.append(turn.get("user_question"))
            texts.append(turn.get("standalone_question"))
    return any(_BROKER.search(str(text or "")) for text in texts)


def _is_broker_question(question, conversation_memory=None) -> bool:
    text = question or ""
    if _TRANSACTION.search(text):
        return False
    if _BROKER.search(text):
        return True
    if asks_for_listings(text) or _BROKER_SWITCH_AWAY.search(text):
        return False
    # A bare nationality / gender filter ("only indian female") only makes sense
    # for agents, even with no memory of the last answer — never a project search.
    if _broker_identity_unsupported(text):
        leftover = _BROKER_FILTER_WORDS.sub(" ", text)
        leftover = re.sub(
            r"\b(?:only|just|show|me|the|and|or|also|please|with|a|an|only)\b",
            " ",
            leftover,
            flags=re.I,
        )
        # Nothing but identity words and filler left → an agent filter.
        if not re.findall(r"[a-z0-9]+", leftover.lower()):
            return True
    # After an agents answer, only short continuations stay on agents
    # ("female", "show more", "Indians"). A new sales/project ask must leave.
    if _last_asked_about_agents(conversation_memory):
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        if _broker_identity_unsupported(text):
            return True
        if len(words) <= 6 and not re.search(
            r"\b(?:sales?|sold|projects?|listings?|price|budget|buy|rent)\b",
            text,
            re.I,
        ):
            return True
        return False
    return False


_AMBIGUOUS_FILLER = re.compile(
    r"\b(?:show|me|tell|find|list|see|look|looking|please|can|you|could|"
    r"what|whats|about|any|anything|there|available|"
    r"in|at|for|of|the|a|an|is|are)\b",
    re.I,
)
_AMBIGUOUS_BLOCK = re.compile(
    r"\b(?:bed(?:room)?s?|studio|budget|aed|sqft|sq\s*ft|price|rent|buy|"
    r"compare|vs|versus)\b",
    re.I,
)
# "what is dubai" / "tell me about marina" = explain, not "show me that place".
_DEFINITIONAL_ASK = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"what(?:'s|\s+is|\s+are)\s+(?:a\s+|an\s+|the\s+)?|"
    r"whats\s+(?:a\s+|an\s+|the\s+)?|"
    r"tell\s+(?:me\s+)?about\s+|"
    r"explain\s+(?:what\s+)?|"
    r"define\s+|"
    r"describe\s+"
    r")(.+)$",
    re.I,
)
_DEFINITIONAL_FIELD = re.compile(
    r"^(?:the\s+)?(?:price|prices|cost|costs|handover|bedroom|bedrooms|"
    r"rent|rents|size|sizes|developer|status|budget|budgets)\b",
    re.I,
)
_CHOICE_AGENTS = re.compile(r"^\s*(?:show\s+me\s+)?(?:the\s+)?(?:agents?|brokers?)\s*[.!]?\s*$", re.I)
_CHOICE_PROPERTIES = re.compile(
    r"^\s*(?:show\s+me\s+)?(?:the\s+)?(?:propert(?:y|ies)|projects?|listings?)\s*[.!]?\s*$",
    re.I,
)


def _is_definitional_question(question: str) -> bool:
    """True for normal explain asks ('what is dubai'), not price/handover field asks."""
    text = (question or "").strip()
    matched = _DEFINITIONAL_ASK.match(text)
    if not matched:
        return False
    rest = matched.group(1).strip(" .?!")
    if not rest:
        return False
    if _DEFINITIONAL_FIELD.match(rest):
        return False
    return True


def _memory_question_texts(conversation_memory) -> list[str]:
    if conversation_memory is None:
        return []
    texts = [
        getattr(conversation_memory, "last_user_question", None),
        getattr(conversation_memory, "last_standalone_question", None),
    ]
    for turn in reversed((getattr(conversation_memory, "turns", None) or [])[-4:]):
        if isinstance(turn, dict):
            texts.append(turn.get("user_question"))
            texts.append(turn.get("standalone_question"))
    return [str(text or "") for text in texts if text]


def _last_asked_about_properties(conversation_memory) -> bool:
    for text in _memory_question_texts(conversation_memory):
        if asks_for_listings(text) or _BROKER_SWITCH_AWAY.search(text):
            return True
    items = getattr(conversation_memory, "active_candidates", None) or []
    return any(
        isinstance(item, dict)
        and str(item.get("entity_type") or "").lower() in {"project", "listing"}
        and item.get("name")
        for item in items
    )


def _place_choice_reply(question):
    text = (question or "").strip()
    if not text:
        return None
    if _CHOICE_AGENTS.fullmatch(text):
        return "agents"
    if _CHOICE_PROPERTIES.fullmatch(text):
        return "properties"
    if _BROKER.search(text) and not _BROKER_SWITCH_AWAY.search(text):
        return "agents"
    if asks_for_listings(text) or _BROKER_SWITCH_AWAY.search(text):
        return "properties"
    return None


def _bare_place(question):
    asked = (question or "").strip()
    if not asked:
        return None
    match = _BROKER_PLACE.search(asked)
    if match:
        place = match.group(1).strip(" .?")
        place = re.split(
            r"\b(?:and tell|tell me|please)\b",
            place,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" .?")
        place = " ".join(_BROKER_FILTER_WORDS.sub(" ", place).split()).strip(" .?")
        if place:
            return place
    leftover = " ".join(_AMBIGUOUS_FILLER.sub(" ", asked).split()).strip(" .?")
    leftover = " ".join(_BROKER_FILTER_WORDS.sub(" ", leftover).split()).strip(" .?")
    return leftover or None


def _ambiguous_place_question(question, conversation_memory):
    """Known Dubai place with no noun, on a first turn that has no inherited intent."""
    text = repair_typed_question(question or "")
    if _is_broker_question(text, conversation_memory):
        return None
    if _last_asked_about_properties(conversation_memory):
        return None
    # "what is dubai" is an explanation ask, not a bare place browse.
    if _is_definitional_question(text):
        return None
    if (
        _BROKER.search(text)
        or _BROKER_SWITCH_AWAY.search(text)
        or asks_for_listings(text)
        or _TRANSACTION.search(text)
        or _BROKER_FILTER_WORDS.search(text)
        or _AMBIGUOUS_BLOCK.search(text)
    ):
        return None
    place = _bare_place(text)
    if not place:
        return None
    from app.retrieval.propai_catalog import is_outside_dubai, resolve_dubai_place
    if is_outside_dubai(place) or is_outside_dubai(text):
        return False
    return resolve_dubai_place(place)


def _apply_pending_place_choice(question, conversation_memory):
    pending = getattr(conversation_memory, "pending_place_choice", None) if conversation_memory else None
    if not pending:
        return question
    choice = _place_choice_reply(question)
    named = bool(re.search(r"\b(?:in|at|for)\b", question or "", re.I) and _bare_place(question))
    if choice == "agents":
        conversation_memory.pending_place_choice = None
        if _BROKER.search(question or "") and named:
            return question
        return f"Show me agents in {pending}"
    if choice == "properties":
        conversation_memory.pending_place_choice = None
        if named and (_BROKER_SWITCH_AWAY.search(question or "") or asks_for_listings(question or "")):
            return question
        kind = "listings" if asks_for_listings(question or "") else "projects"
        return f"Show me {kind} in {pending}"
    return question


def _place_choice_turn(question, place, conversation_memory):
    if conversation_memory is not None:
        conversation_memory.pending_place_choice = place
    summary = _empty_structured_summary()
    summary["place_choice"] = True
    summary["understood_subject"] = place
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
        "fast_answer": f"Are you looking for properties or agents in {place}?",
        "context": None,
        "small_talk": None,
    }


def _inherit_place_for_home_browse(question, conversation_memory, prior=None):
    """Attach the last Dubai place when the user only names beds/type.

    After "projects in JVC", "show me two-bedroom apartments" must become
    "Show me 2 bedroom apartments in Jumeirah Village Circle" for SQL browse.
    """
    if not is_incomplete_home_browse(question):
        return question
    locked = extract_place(question)
    if not locked or locked.lower() == "dubai":
        locked = extract_place(prior) if prior else None
    if not locked or str(locked).lower() == "dubai":
        locked = _on_screen_broker_place(conversation_memory)
    if not locked or str(locked).lower() == "dubai":
        return question
    beds = parse_bedrooms(question)
    bed_label = f"{beds[0]} bedroom " if beds else ""
    home = "apartments" if re.search(
        r"\b(?:apartments?|flats?)\b", question or "", re.I
    ) else "projects"
    return f"Show me {bed_label}{home} in {locked}"


def _on_screen_broker_place(conversation_memory):
    """First area or project already on screen, so agent follow-ups stay on the last answer."""
    items = getattr(conversation_memory, "active_candidates", None) or []
    communities = []
    names = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("entity_type") or "project").lower()
        name = str(item.get("name") or item.get("title") or "").strip()
        community = str(
            item.get("community") or item.get("official_area") or ""
        ).strip()
        if kind == "community":
            if name and name not in communities:
                communities.append(name)
            continue
        if community and community not in communities:
            communities.append(community)
        if kind in {"project", "listing"} and name:
            names.append(name)
    if communities:
        return communities[0]
    if len(names) == 1:
        return names[0]
    return None


def _agents_on_screen_place_allowed(question, conversation_memory) -> bool:
    """Same freshness gate as sales: named/deixis/live cards stick; cold clarifies."""
    if _TX_DEIXIS.search(question or "") or _BROKER_POINTER.search(question or ""):
        return True
    if _dialogue_action_name(conversation_memory) == "agents":
        return True
    return _project_card_context_is_live(question, conversation_memory)


def _canonical_broker_place(place: str | None) -> str | None:
    """Strip chip aliases like '(JVC)' and map to a known Dubai community/project."""
    if not place or not str(place).strip():
        return None
    raw = str(place).strip()
    try:
        from app.retrieval.propai_catalog import known_dubai_label, resolve_dubai_place

        return known_dubai_label(raw) or resolve_dubai_place(raw) or raw
    except Exception:
        cleaned = re.sub(r"\s*\([^)]{1,40}\)", "", raw).strip()
        return cleaned or raw


def _broker_place(question, conversation_memory):
    """Community or project a broker follow-up refers to."""
    asked = (question or "").strip()
    match = _BROKER_PLACE.search(asked)
    if match:
        place = match.group(1).strip(" .?")
        place = re.split(
            r"\b(?:and tell|tell me|please)\b",
            place,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" .?")
        place = _strip_broker_budget(place)
        place = " ".join(_BROKER_FILTER_WORDS.sub(" ", place).split()).strip(" .?")
        # "the jvc" -> "Jumeirah Village Circle", so answers read properly and the search uses the full name.
        place = re.sub(r"^(?:the|a)\s+", "", place, flags=re.I)
        # "these projects" / "those areas" points at the last answer, not a new place.
        pointed = " ".join(_BROKER_POINTER.sub(" ", place).split()).strip(" .?")
        if pointed:
            from app.listings.tools import ALIASES
            return _canonical_broker_place(ALIASES.get(pointed.lower(), pointed))
    leftover = " ".join(_BROKER_FILLER.sub(" ", asked).split())
    leftover = " ".join(_BROKER_FILTER_WORDS.sub(" ", leftover).split()).strip(" .?")
    leftover = " ".join(_BROKER_POINTER.sub(" ", leftover).split()).strip(" .?")
    leftover = _strip_broker_budget(leftover)
    leftover = " ".join(_BROKER_BUDGET_TOKENS.sub(" ", leftover).split()).strip(" .?")
    if leftover:
        from app.retrieval.propai_catalog import is_outside_dubai, known_dubai_label
        if is_outside_dubai(leftover) or known_dubai_label(leftover):
            return _canonical_broker_place(leftover)
    if not _agents_on_screen_place_allowed(question, conversation_memory):
        return None
    return _on_screen_broker_place(conversation_memory)


def _broker_lookup(question, conversation_memory):
    """Name or BRN to search, or None for the ranked list."""
    leftover = _BROKER_FILLER.sub(" ", question or "")
    leftover = re.sub(r"[^a-zA-Z0-9\s]", " ", leftover)
    leftover = " ".join(leftover.split())
    leftover = " ".join(_BROKER_BUDGET_TOKENS.sub(" ", leftover).split())
    place = _broker_place(question, conversation_memory)
    if place:
        # Word by word: "the" is already gone from the leftover, so "the jvc" as a phrase would not match.
        from app.listings.tools import ALIASES
        short = {key for key, full in ALIASES.items() if full.lower() == place.lower()}  # "jvc" for JVC
        for word in set(re.findall(r"[a-z0-9]+", place.lower())) | {place.lower()} | short:
            leftover = re.sub(rf"\b{re.escape(word)}\b", " ", leftover, flags=re.I)
        leftover = " ".join(leftover.split())
    if re.fullmatch(r"\d{3,12}", leftover or ""):
        return None, leftover
    if leftover and len(leftover) >= 3:
        return leftover, None
    return None, None


def _broker_expiry(value):
    text = str(value or "")[:10]
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%b %Y")
    except ValueError:
        return text


def _project_is_full(project) -> bool:
    """Enough structured fields to send the highest-ranked brokers."""
    if not isinstance(project, dict):
        return False
    has_price = bool(project.get("price"))
    has_beds = project.get("bedroom_min") is not None or project.get("bedroom_max") is not None
    size = project.get("size_sqft") if isinstance(project.get("size_sqft"), dict) else {}
    has_size = (
        project.get("size_min") is not None
        or project.get("size_max") is not None
        or size.get("min") is not None
        or size.get("max") is not None
    )
    has_handover = bool(project.get("handover_time") or project.get("handover"))
    photos = project.get("photos") or []
    has_photo = isinstance(photos, list) and any(photos)
    has_sales = bool(project.get("sales_count") or project.get("recent_sales_count"))
    return bool(has_price and has_beds and has_size and has_handover and (has_photo or has_sales))


def _shown_projects_for_place(conversation_memory, place=None):
    items = getattr(conversation_memory, "active_candidates", None) or []
    projects = [
        item
        for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    if not place:
        return projects
    needle = str(place).strip().lower()
    matched = [
        project
        for project in projects
        if needle in str(project.get("community") or "").lower()
        or needle in str(project.get("name") or "").lower()
    ]
    return matched or projects


def _broker_band(conversation_memory, place=None) -> str:
    projects = _shown_projects_for_place(conversation_memory, place)
    if any(_project_is_full(project) for project in projects):
        return "top"
    return "mid"


def _broker_reply(question, answer, brokers=None):
    summary = _empty_structured_summary()
    summary["brokers"] = brokers or []
    summary["broker_followup"] = True
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
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _format_brokers(rows, label=None) -> str:
    from app.answers.explain_contract import build_agents_narrative, scrub_user_facing

    if not rows:
        if label:
            return (
                f"Understood: you asked for licensed agents for {label}.\n\n"
                f"I could not find matching licensed agents for {label} in our DLD-linked data. "
                "Try a nearby community such as Dubai Marina, Business Bay, or Palm Jumeirah."
            )
        return (
            "Understood: you asked for licensed agents in Dubai.\n\n"
            "I could not find matching licensed agents in our DLD-linked data right now."
        )
    lines = [build_agents_narrative(place=label, count=len(rows)), ""]
    for index, row in enumerate(rows, start=1):
        name = row.get("name") or row.get("name_en") or "Agent"
        bits = []
        if row.get("agency_name"):
            bits.append(f"agency {row['agency_name']}")
        if row.get("card_rank"):
            bits.append(f"card rank {row['card_rank']}")
        if row.get("brn"):
            bits.append(f"BRN {row['brn']}")
        if row.get("license_number"):
            bits.append(f"licence {row['license_number']}")
        expiry = _broker_expiry(row.get("card_expiry_date"))
        if expiry:
            bits.append(f"card valid until {expiry}")
        if row.get("awards_count"):
            bits.append(f"{int(row['awards_count'])} recorded awards")
        detail = "; ".join(str(bit) for bit in bits if bit)
        lines.append(f"{index}. **{name}**")
        if detail:
            lines.append(f"   {detail}.")
        lines.append("")
    lines.append(
        "Want agents for a nearby community, or should I match projects in the same area next?"
    )
    return scrub_user_facing("\n".join(lines).rstrip())


_OUTSIDE_SKIP = re.compile(
    r"\b(?:agents?|brokers?|transactions?|transcations?|sales?\s+in|rent(?:al|als)?)\b",
    re.I,
)


def _outside_dubai_market_reply(question):
    """Buy / area / project asks about another city: Dubai-only, said plainly."""
    text = question or ""
    if not text.strip() or _OUTSIDE_SKIP.search(text):
        return None
    from app.answers.advice_turn import advice_kind

    if advice_kind(text):
        return None  # "Compare Dubai and London without assuming my tax residence" wants a framework (AI-058)
    try:
        from app.retrieval.propai_catalog import outside_dubai_city

        city = outside_dubai_city(text)
    except Exception:
        return None
    if not city:
        return None
    return (
        f"ACOT only covers Dubai, so I don't have {city} data. "
        "I can rank Dubai areas for families, check what a budget buys, or compare "
        "projects — for example Dubai Hills Estate, Arabian Ranches, Dubai Marina, "
        "Business Bay or JVC. Which would you like?"
    )


def _outside_coverage_reply(question, asked=None):
    from app.retrieval.propai_catalog import outside_dubai_city
    city = asked or outside_dubai_city(question)
    if not city:
        return None
    answer = (
        "I can only show licensed agents in Dubai. "
        "Ask about an area such as Dubai Marina, Business Bay, or Palm Jumeirah."
    )
    return _broker_reply(question, answer)


# "What communities do you have?" / "full list of communities": the whole list from the data, not a top 10.
# Tolerates common typos (communintes, comunities); skips questions that narrow it (best, near, under...).
_COMMUNITY = r"com+u?n+\w{0,3}t(?:y|ies|ie|es|is|s)?"
_COMMUNITY_LIST = re.compile(
    rf"\b(?:full|complete|entire|whole)\s+list\b.*\b{_COMMUNITY}\b"
    rf"|\b(?:all|every)\s+(?:the\s+)?{_COMMUNITY}\b"
    rf"|\blist\s+(?:of\s+)?(?:all\s+)?(?:the\s+)?{_COMMUNITY}\b"
    rf"|\b{_COMMUNITY}\s+(?:do\s+)?(?:we|you|acot)\s+(?:have|cover|support|got)\b",
    re.I,
)
_COMMUNITY_NARROWED = re.compile(
    r"\b(?:best|top|cheap\w*|afford\w*|family|families|near|nearby|under|below|above|with|for|luxury|safe\w*|compare)\b",
    re.I,
)


def _project_detail_turn(question, conversation_memory=None, name=None):
    """Full detail for a named / on-screen project from Prop-AI. No model call.

    `name`: the project the turn resolver understood; otherwise it is read from the question.
    """
    from app.answers.intent_route import project_detail_subject
    from app.retrieval.propai_catalog import agent_projects

    name = name or project_detail_subject(question, conversation_memory)
    if not name:
        return None
    shown = None
    needle = name.strip().lower()
    for item in _active_projects(conversation_memory):
        label = str(item.get("name") or "").strip()
        if label.lower() == needle or needle in label.lower():
            shown = item
            if label.lower() == needle:
                break
    rows = []
    try:
        rows, _ = agent_projects({"name": shown.get("name") if shown else name}, limit=6)
    except Exception:
        rows = []
    wanted = str((shown or {}).get("name") or name).strip().lower()
    exact = [
        row for row in rows
        if str(row.get("name") or "").strip().lower() == wanted
    ]
    picked = exact[0] if exact else (rows[0] if len(rows) == 1 else None)
    if shown:
        merged = dict(shown)
        if picked:
            for key, value in picked.items():
                if value in (None, "", [], {}):
                    continue
                if key not in merged or merged.get(key) in (None, "", [], {}):
                    merged[key] = value
                elif key in {
                    "price", "developer_name", "community", "handover_time",
                    "description", "photos", "sales_count", "price_kind",
                    "bedroom_min", "bedroom_max", "size_min", "size_max",
                }:
                    merged[key] = value
        picked = merged
    if not picked:
        summary = _empty_structured_summary()
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
                    "projects": [],
                    "sub_communities": [],
                },
                "documents": [],
            },
            "use_fast": True,
            "fast_answer": (
                f'I do not have a project matching "{name}". '
                "Ask about one of the projects already on screen, or name the area."
            ),
            "context": None,
            "small_talk": None,
        }
    picked.setdefault("entity_type", "project")
    summary = _empty_structured_summary()
    summary["projects"] = [picked]
    summary["project_total"] = 1
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
                "projects": [picked],
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": _build_fast_structured_answer(question, summary),
        "context": None,
        "small_talk": None,
    }


def _place_detail_turn(question, conversation_memory=None):
    """Recorded sales and example projects for one community. No model call."""
    from app.answers.intent_route import place_detail_subject
    from app.answers.predict import (
        _place_keys,
        attach_latest_project_sales,
        fetch_place_trend,
        format_place_details,
        projects_for_place,
    )
    from app.retrieval.propai_catalog import agent_projects

    place = place_detail_subject(question, conversation_memory)
    if not place:
        return None
    try:
        trend = fetch_place_trend(place)
    except Exception:
        trend = None
    try:
        pool, _total = agent_projects({"communities": _place_keys(place)}, limit=40, sort="cheapest")
    except Exception:
        pool = []
    matched = attach_latest_project_sales(projects_for_place(place, pool))
    projects = matched if len(matched) <= 15 else matched[:8]
    answer = format_place_details(place, trend, projects, total=len(matched))
    summary = _empty_structured_summary()
    summary["projects"] = projects or []
    summary["project_total"] = len(matched)
    summary["community"] = [{"name": place, "entity_type": "community"}]
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
                "community": summary["community"],
                "projects": projects or [],
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


# Meta asks about catalog calendar / max sale date — not "show me recent sales".
# Require "data" (or today/latest-sale-date) so listing browses stay on sales_list.
_DATA_FRESHNESS = re.compile(
    r"(?:"
    r"today'?s\s+date|"
    r"what(?:'s|\s+is)\s+today|"
    r"current\s+date|"
    r"latest\s+sale\s+date|"
    r"how\s+(?:fresh|current|up[\s-]?to[\s-]?date)\s+(?:is\s+)?(?:your\s+)?data|"
    r"as\s+of\s+when|"
    r"until\s+when\s+(?:is\s+)?(?:your\s+)?(?:sales?\s+)?data|"
    r"how\s+recent\s+(?:is\s+)?(?:your\s+|the\s+)?(?:sales?\s+)?data\b|"
    r"what(?:'s|\s+is)\s+the\s+(?:most\s+)?recent\s+data\b|"
    r"(?:most\s+)?recent.{0,40}?data.{0,80}?\b(?:sales?|dubai)\b|"
    r"(?:most\s+)?recent.{0,40}?(?:dubai\s+)?sales?\s+data\b|"
    r"latest.{0,40}?data.{0,80}?\b(?:sales?|dubai)\b|"
    r"data\s+(?:we|you)\s+have.{0,60}?\bsales?\b|"
    r"sales?\s+data\b.{0,60}?\b(?:most\s+)?recent\b|"
    r"freshest\s+(?:sales?\s+)?data\b"
    r")",
    re.I | re.S,
)


_DATE_ONLY_ASK = re.compile(
    r"\btoday'?s?\s+date\b|\bdate\s+today\b|\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:date|day)\b|"
    r"\bwhat\s+day\s+is\s+(?:it|today)\b|\bcurrent\s+date\b",
    re.I,
)
_DATA_WORDS = re.compile(
    r"\b(?:data|sales?|sold|rents?|rental|records?|transactions?|latest|recent|updated?|info(?:rmation)?)\b", re.I
)


def _data_freshness_turn(question, understood=False):
    """Calendar today + latest recorded sale and rent dates. Never invent a scrape snapshot as today.

    `understood`: the turn resolver already decided this is a freshness ask, so skip the wording check.
    """
    # Runs before follow-up handling, so date asks never re-run an on-screen search.
    if not understood and not (_DATA_FRESHNESS.search(question or "") or _DATE_ONLY_ASK.search(question or "")):
        return None
    from datetime import date
    from app.retrieval.propai_catalog import fetch

    today = date.today()
    if _DATE_ONLY_ASK.search(question or "") and not _DATA_WORDS.search(question or ""):
        # "What is today's date?" gets the date, not a data report.
        answer = f"Today is **{today.strftime('%A, %d %B %Y')}**."
        summary = _empty_structured_summary()
        summary["data_freshness"] = {"today": today.isoformat()}
        return {
            "question": question, "standalone_question": question, "question_type": "general",
            "structured_summary": summary, "documents": [], "investment_analysis": None,
            "retrieval_result": {"question_type": "general",
                                 "structured_data": {"community": [], "projects": [], "sub_communities": []},
                                 "documents": []},
            "use_fast": True, "fast_answer": answer, "context": None, "small_talk": None,
        }
    try:
        row = fetch(
            "select (select max(instance_date) from propai_sales"
            "         where deleted_at is null and instance_date <= current_date)::date as latest, "
            "       (select max(contract_start_date) from propai_rents"
            "         where deleted_at is null and contract_start_date <= current_date)::date as rent"
        )
        # fetch() returns ISO text; turn it back into dates so they read "18 Nov 2026".
        latest, rent = (
            date.fromisoformat(str(row[0][key])[:10]) if row and row[0].get(key) else None
            for key in ("latest", "rent")
        )
    except Exception as exc:
        print(f"data freshness lookup failed: {exc}")
        latest = rent = None
    if hasattr(latest, "isoformat"):
        latest_text = latest.strftime("%d %b %Y")
        latest_iso = latest.isoformat()
    elif latest:
        latest_text = str(latest)[:10]
        latest_iso = latest_text
    else:
        latest_text = None
        latest_iso = None
    today_text = today.strftime("%d %b %Y")
    rent_text = None
    if rent:
        rent_text = rent.strftime("%d %b %Y") if hasattr(rent, "strftime") else str(rent)[:10]
    if latest_text:
        lines = [f"- **Sales:** Dubai Land Department sale transactions, up to **{latest_text}**"]
        if rent_text:
            lines.append(f"- **Rents:** registered rent contracts (Ejari), up to **{rent_text}**")
        lines.append("- **Projects:** the Dubai Land Department project register (developer, status, completion)")
        lines.append("- **Agents:** Dubai Land Department broker licence records")
        answer = (
            f"My data comes from official Dubai Land Department records (today is {today_text}):\n\n"
            + "\n".join(lines)
            + "\n\nEvery price and count I give comes from these records, so nothing after these dates is included."
        )
    else:
        answer = (
            f"Today is {today_text}. "
            "I could not read the latest sale date from the sales table right now."
        )
    summary = _empty_structured_summary()
    summary["data_freshness"] = {
        "today": today.isoformat(),
        "latest_sale_date": latest_iso,
    }
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
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _fast_answer(question, answer, projects=None, question_type="general"):
    """Response shape for an answer built straight from the data (no model call)."""
    summary = _empty_structured_summary()
    summary["projects"] = list(projects or [])
    return {
        "question": question, "standalone_question": question, "question_type": question_type,
        "structured_summary": summary, "documents": [], "investment_analysis": None,
        "retrieval_result": {"question_type": question_type,
                             "structured_data": {"community": [], "projects": list(projects or []),
                                                 "sub_communities": []},
                             "documents": []},
        "use_fast": True, "fast_answer": answer, "context": None, "small_talk": None,
    }


_COMMUNITY_DEVELOPER_QUESTION = re.compile(
    r"(?:who\s+(?:(?:is|was)\s+(?:the\s+)?(?:master\s+)?developer\s+(?:of|for)|developed)"
    r"|(?:the\s+)?(?:master\s+)?developer\s+(?:of|for))\s+"
    r"(?:jvc|jumeirah village circle(?:\s*\(jvc\))?)(?:\s+in dubai)?[?.!]*",
    re.I,
)


def _community_developer_turn(question):
    if not _COMMUNITY_DEVELOPER_QUESTION.fullmatch((question or "").strip()):
        return None
    # Nakheel confirms its role for the master community:
    # https://www.nakheel.com/en/media-centre/blogs/detail/nakheel-neighbourhoods-in-dubai-with-high-roi-for-real-estate-buyers
    return _fast_answer(
        question,
        "**Nakheel** is the master developer of **Jumeirah Village Circle (JVC)**. "
        "Individual projects in JVC can have their own developers.",
    )


def _aed(value) -> str:
    return f"AED {int(round(float(value))):,}"


def _where(place) -> str:
    return f" in {place}" if place and str(place).strip().lower() != "dubai" else " in Dubai"


def _beds_label(beds) -> str:
    if beds is None:
        return ""
    return "studio " if int(beds) == 0 else f"{int(beds)}-bedroom "


def _rent_stat_turn(question, resolution):
    """Median / lowest / highest yearly rent for a place, from registered rent contracts."""
    from app.retrieval.propai_catalog import rent_stats

    place, beds, stat = resolution.get("place"), resolution.get("beds"), resolution.get("stat") or "average"
    if not place or str(place).strip().lower() == "dubai":
        return None  # "cheapest rents in Dubai" is an area ranking, handled by area_rank
    try:
        row = rent_stats([place], beds, beds)
    except Exception as exc:
        print(f"rent lookup failed: {exc}")
        row = None
    what = f"{_beds_label(beds)}rent{_where(place)}"
    if not row:
        return _fast_answer(question, f"I couldn't find registered rent contracts for a {what}.")
    low, high = row.get("typical_range_annual_aed") or [None, None]
    n = row.get("listings")
    # The single lowest / highest contract is usually a data error, so quote the cheaper / upper quarter.
    if stat == "cheapest" and low:
        lead = f"At the cheaper end, a yearly {what} costs about **{_aed(low)}** or less (lowest quarter of contracts)."
    elif stat == "highest" and high:
        lead = f"At the top end, a yearly {what} costs about **{_aed(high)}** or more (top quarter of contracts)."
    else:
        lead = f"The typical yearly {what} is **{_aed(row['median_annual_rent_aed'])}** (median)."
    detail = f"Most contracts fall between **{_aed(low)}** and **{_aed(high)}** a year" if low and high else ""
    answer = (
        f"{lead}\n\n"
        + (f"- {detail}\n" if detail else "")
        + f"- Based on **{n:,}** registered rent contracts since Sep 2025 (Dubai Land Department / Ejari)"
    )
    return _fast_answer(question, answer)


def _market_numbers(place):
    """One area's figures: our recorded sales (price/sq ft, change, activity) plus Prop-AI's matched rental
    yield and 3-year price growth. Prop-AI wins for yield; our gross yield is the fallback."""
    from app.answers.area_comparison import market_signals, official_area
    from app.retrieval.propai_live import area_returns

    official = official_area(place)
    signals = dict(market_signals(official) or {}) if official else {}
    live = area_returns(official) if official else None
    if live and live.get("rental_yield_pct") is not None:
        signals["gross_yield_pct"] = live["rental_yield_pct"]
    if live:
        signals["growth_3y_pct"] = live.get("growth_3y_pct")
        signals["growth_years"] = live.get("growth_years")
    if (signals.get("sales_12m") or 0) < _MIN_SALES:
        # A median from a handful of sales swings wildly (Al Qusais: "+47%" from one sale).
        for key in ("price_psf", "price_psf_change_pct", "sales_change_pct", "growth_3y_pct", "gross_yield_pct"):
            signals[key] = None
    yld, growth = signals.get("gross_yield_pct"), signals.get("growth_3y_pct")
    signals["roi_pct"] = round(yld + growth, 1) if yld is not None and growth is not None else None
    return signals


_MIN_SALES = 30  # fewer recorded sales in 12 months than this: too few to quote a price or trend


def _pct(value) -> str:
    return f"{'+' if value > 0 else ''}{value}%"


_ROI_NOTE = ("ROI here = rental yield + yearly price growth, for apartments, before service charges, fees, "
             "vacancy and mortgage costs. Past growth does not guarantee future returns.")


def _area_market_turn(question, resolution):
    """Rental yield, ROI, price per sq ft, price trend or an investment summary for one area."""
    from app.answers.area_comparison import areas_in

    named = areas_in(question)
    if len(named) >= 2:
        return _area_pair_turn(question, named[0], named[1])
    place = resolution.get("place")
    if not place or str(place).strip().lower() == "dubai":
        return None
    signals = _market_numbers(place)
    kind = resolution.get("kind") or "invest"
    if kind == "profile":
        profile = _area_profile_turn(question, place, signals)
        if profile is not None:
            return profile
        kind = "invest"  # web search unavailable: the investment view from our records still answers
    if not signals.get("sales_12m") and signals.get("gross_yield_pct") is None:
        return _fast_answer(question, f"I don't have enough recorded sales{_where(place)} to measure that.")
    psf, change = signals.get("price_psf"), signals.get("price_psf_change_pct")
    yld, roi, growth = signals.get("gross_yield_pct"), signals.get("roi_pct"), signals.get("growth_3y_pct")
    sales, sales_change = signals.get("sales_12m"), signals.get("sales_change_pct")
    years = signals.get("growth_years")
    span = f" ({years[0]} → {years[1]})" if years else ""
    trend = (f"**up {change}%**" if change and change > 0 else f"**down {abs(change)}%**" if change and change < 0
             else "**flat**") if change is not None else None
    where = _where(place)

    if kind == "roi":
        if roi is None:
            return _fast_answer(question, f"I can't estimate an ROI{where}: I need both rental yield and price "
                                          "history, and one of them is missing.")
        answer = (
            f"The estimated return on an apartment{where} is about **{roi}% a year**:\n\n"
            f"- **Rental income:** {yld}% rental yield\n"
            f"- **Capital growth:** {_pct(growth)} a year in price per sq ft{span}\n"
            + (f"- **Last 12 months:** price {_pct(change)}\n" if change is not None else "")
            + f"\n{_ROI_NOTE}"
        )
        return _fast_answer(question, answer)

    lines = []
    if kind == "yield":
        if yld is None:
            return _fast_answer(question, f"I can't measure a rental yield{where}: not enough rents and sales on record.")
        lead = f"The rental yield{where} is about **{yld}%** a year (apartments): the rent you earn as a share of the price."
        if roi is not None:
            lines.append(f"- **Total return (ROI):** about {roi}% a year once price growth ({_pct(growth)} a year) is added")
    elif kind == "psf":
        lead = f"Homes{where} sell for a median **AED {psf:,} per sq ft** (last 12 months)."
    elif kind == "trend":
        lead = (f"Prices{where} are {trend} over the last 12 months (median price per sq ft, now AED {psf:,})."
                if trend else f"I can't measure a price trend{where} yet.")
        if growth is not None:
            lines.append(f"- **Over 3 years:** {_pct(growth)} a year{span}")
    else:
        lead = f"Here is how **{place}** looks as an investment (apartments, recorded sales and rents):"
        if roi is not None:
            lines.append(f"- **Estimated ROI:** about {roi}% a year (rental yield + price growth)")
    if kind != "psf" and psf:
        lines.append(f"- **Price:** AED {psf:,} per sq ft (median, last 12 months)")
    if kind not in ("trend",) and trend:
        lines.append(f"- **Price change:** {trend} vs the previous 12 months")
    if kind not in ("yield",) and yld is not None:
        lines.append(f"- **Rental yield:** about {yld}%")
    if sales:
        move = f" ({_pct(sales_change)} vs the year before)" if sales_change is not None else ""
        lines.append(f"- **Sales activity:** {sales:,} recorded sales in 12 months{move}")
    note = f"\n\n{_ROI_NOTE}" if kind == "invest" and roi is not None else ""
    return _fast_answer(question, lead + ("\n\n" + "\n".join(lines) if lines else "") + note)


_PROFILE_FOCUS = (  # (words in the question, words in the bullet that answers it)
    (r"metro|transport|commut|traffic|getting around|bus", r"metro|transport|getting around"),
    (r"school|nurser|educat", r"school"),
    (r"downside|cons\b|disadvantage|drawback|problem", r"downside|drawback"),
    (r"amenit|mall|shop|restaurant|park|beach", r"amenit|mall|shop"),
    (r"who (is it|it'?s) for|suit|famil|couple", r"suit|famil"),
)


def _area_profile_turn(question, place, signals):
    """"Tell me about JVC" / "what is it like to live in": a web-sourced community profile plus 2-3 key
    figures from our records, with sources. None when the web search is unavailable."""
    from app.answers.area_comparison import official_area
    from app.retrieval.web_context import area_profile

    profile = area_profile(place, key=official_area(place) or place)
    if not profile:
        return None
    psf, change, yld = signals.get("price_psf"), signals.get("price_psf_change_pct"), signals.get("gross_yield_pct")
    figures = []
    if psf:
        figures.append(f"- **Price:** AED {psf:,} per sq ft (median, last 12 months"
                       + (f", {_pct(change)} on the year)" if change is not None else ")"))
    if yld is not None:
        figures.append(f"- **Rental yield:** about {yld}% a year (apartments)")
    if signals.get("sales_12m"):
        figures.append(f"- **Sales activity:** {signals['sales_12m']:,} recorded sales in 12 months")
    bullets = profile["text"].split("\n")
    asked = [rx for words, rx in _PROFILE_FOCUS if re.search(words, question, re.I)]
    # "Is there a metro in JVC?": the bullet that answers it goes first; the rest of the profile follows.
    bullets.sort(key=lambda line: not any(re.search(rx, line, re.I) for rx in asked))
    answer = f"**{place}** at a glance:\n\n" + "\n".join(bullets)
    if figures:
        answer += "\n\n**Key figures** (Dubai Land Department records):\n" + "\n".join(figures)
    # ponytail: sources are kept in the web cache but not shown (the chat UI printed raw markdown links);
    # add them back as source chips if the frontend ever renders them.
    return _fast_answer(question, answer)


def _web_answer_turn(question, resolution):
    """A question our records cannot answer ("best restaurant near JVC?", golden visa rules), from the web."""
    from app.retrieval.web_context import web_answer

    found = web_answer(question, resolution.get("place"))
    return _fast_answer(question, found["text"]) if found else None


def _area_pair_turn(question, first, second):
    """Two areas side by side on the same market numbers ("Business Bay or Downtown for investment?")."""
    from app.answers.area_comparison import market_signals, official_area

    first, second = (n.title() if n.isupper() else n for n in (first, second))
    picks = {name: _market_numbers(name) for name in (first, second)}

    def row(label, key, fmt):
        values = [picks[name].get(key) for name in (first, second)]
        return f"- **{label}:** " + " vs ".join(
            f"{name} {fmt(v) if v is not None else 'n/a'}" for name, v in zip((first, second), values)
        )

    lines = [
        row("Price per sq ft", "price_psf", lambda v: f"AED {v:,}"),
        row("Price change (12 months)", "price_psf_change_pct", lambda v: f"{'+' if v > 0 else ''}{v}%"),
        row("Rental yield", "gross_yield_pct", lambda v: f"{v}%"),
        row("Estimated ROI (yield + price growth)", "roi_pct", lambda v: f"{v}% a year"),
        row("Sales in 12 months", "sales_12m", lambda v: f"{v:,}"),
    ]
    yields = {n: picks[n].get("gross_yield_pct") for n in (first, second)}
    growth = {n: picks[n].get("price_psf_change_pct") for n in (first, second)}
    verdict = []
    if all(v is not None for v in yields.values()) and yields[first] != yields[second]:
        verdict.append(f"**{max(yields, key=yields.get)}** earns more rent for the price")
    if all(v is not None for v in growth.values()) and growth[first] != growth[second]:
        verdict.append(f"**{max(growth, key=growth.get)}** has risen more in price over the last year")
    rois = {n: picks[n].get("roi_pct") for n in (first, second)}
    if all(v is not None for v in rois.values()) and rois[first] != rois[second]:
        verdict.append(f"**{max(rois, key=rois.get)}** has the higher estimated total return")
    answer = (
        f"**{first} vs {second}**, from the last 12 months of recorded sales and rents:\n\n" + "\n".join(lines)
        + ("\n\n" + "; ".join(verdict) + "." if verdict else "")
        + "\n\n" + _ROI_NOTE
    )
    return _fast_answer(question, answer)


def _predict_turn(question, resolution):
    """Price outlook for one area: the recorded trend and a projection from past growth, never a made-up price.
    Says plainly when there is not enough data."""
    place = resolution.get("place")
    if not place or str(place).strip().lower() == "dubai":
        return _fast_answer(question, "Which area should I look at? A price outlook needs one community, "
                                      "for example JVC, Business Bay or Dubai Creek Harbour.")
    signals = _market_numbers(place)
    psf, change = signals.get("price_psf"), signals.get("price_psf_change_pct")
    growth, years = signals.get("growth_3y_pct"), signals.get("growth_years")
    if psf is None and growth is None:
        return _fast_answer(question, f"We don't have enough recorded sales{_where(place)} to show a price trend, "
                                      "so I can't make a prediction there.")
    lines = []
    if change is not None:
        lines.append(f"- **Last 12 months:** {_pct(change)} (median AED {psf:,} per sq ft)")
    if growth is not None:
        lines.append(f"- **Last 3 years:** {_pct(growth)} a year ({years[0]} → {years[1]})")
    if signals.get("sales_12m"):
        move = signals.get("sales_change_pct")
        lines.append(f"- **Sales activity:** {signals['sales_12m']:,} sales in 12 months"
                     + (f" ({_pct(move)} vs the year before)" if move is not None else ""))
    if growth is not None:
        outlook = (f"If the 3-year pace continued, prices{_where(place)} would be about **{_pct(growth)}** in a "
                   "year. That is a projection from past growth, not a forecast: rates, new supply and the "
                   "wider market can change it.")
    else:
        outlook = "There is not enough price history for a projection, only the last 12 months above."
    answer = (f"No one can predict prices exactly, but here is the recorded trend for apartments{_where(place)}:\n\n"
              + "\n".join(lines) + "\n\n" + outlook)
    return _fast_answer(question, answer, question_type="prediction")


_AREA_RANK_TEXT = {
    "cheapest_rent": ("lowest", "rent", "median yearly rent"),
    "priciest_rent": ("highest", "rent", "median yearly rent"),
    "cheapest_buy": ("lowest", "buy", "median apartment price"),
    "priciest_buy": ("highest", "buy", "median apartment price"),
    "best_yield": ("best", "yield", "gross rental yield"),
}


def _area_rank_turn(question, resolution):
    """Dubai areas ranked by rent, price or gross yield (apartments, last 12 months)."""
    from app.retrieval.propai_catalog import area_ranking

    kind = resolution.get("kind") if resolution.get("kind") in _AREA_RANK_TEXT else None
    if kind is None:
        return None
    beds = resolution.get("beds")
    rows = area_ranking(kind, beds=beds, limit=8)
    word, _, measure = _AREA_RANK_TEXT[kind]
    homes = f"{_beds_label(beds)}apartments" if beds is not None else "apartments"
    if not rows:
        return _fast_answer(question, "I don't have enough recorded data to rank areas for that.")
    def line(i, r):
        if kind == "best_yield":
            return (f"{i}. **{r['place']}** — about **{r['yield_pct']}%** "
                    f"(rent {_aed(r['rent'])}/yr, price {_aed(r['price'])})")
        if "rent" in kind:
            return f"{i}. **{r['place']}** — **{_aed(r['rent'])}** a year ({r['rents']:,} contracts)"
        return f"{i}. **{r['place']}** — **{_aed(r['price'])}** ({r['sales']:,} sales)"
    title = {"best_yield": f"Dubai areas with the **best gross rental yield** for {homes}",
             }.get(kind, f"Dubai areas with the **{word} {measure}** for {homes}")
    note = ("\n\nYield = median yearly rent / median ready-home price, before service charges and vacancy. "
            if kind == "best_yield" else "\n\n")
    answer = (title + " (last 12 months):\n\n" + "\n".join(line(i, r) for i, r in enumerate(rows, 1))
              + note + "Areas need enough contracts and sales to be ranked, so one odd building can't top the list.")
    return _fast_answer(question, answer)


def _top_projects_turn(question, resolution):
    """Projects ranked by number of recorded sales."""
    from datetime import date
    from app.retrieval.propai_catalog import top_projects_by_sales

    year, place = resolution.get("year"), resolution.get("place")
    rows = top_projects_by_sales(place=place, year=year, limit=10)
    period = (f"in {year}" + (f" (to {date.today().strftime('%d %b')})" if year == date.today().year else "")
              if year else "in the last 12 months")
    if not rows:
        return _fast_answer(question, f"I found no recorded sales{_where(place)} {period}.")
    lines = [
        f"{i}. **{r['name']}** — {r['sales']:,} sales · median {_aed(r['median_price'])}"
        + (f" · {r['community']}" if r.get("community") else "")
        for i, r in enumerate(rows, 1)
    ]
    answer = (f"The **best-selling projects**{_where(place)} {period}, by number of recorded sales:\n\n"
              + "\n".join(lines) + "\n\nSource: Dubai Land Department sale records (off-plan and ready).")
    return _fast_answer(question, answer, question_type="search")


def _handover_turn(question, resolution, conversation_memory=None):
    """Most recently handed-over (or next to hand over) projects, straight from DLD project records."""
    from datetime import date
    from app.retrieval.propai_catalog import handover_projects

    year = resolution.get("year")
    when = "upcoming" if resolution.get("kind") == "upcoming" else "recent"
    if year:
        when = "upcoming" if int(year) >= date.today().year else "recent"
    place = resolution.get("place")
    where = f" in {place}" if place and str(place).strip().lower() != "dubai" else " in Dubai"
    try:
        projects = handover_projects(when, place, limit=10, year=year)
    except Exception as exc:
        print(f"handover lookup failed: {exc}")
        projects = []

    def month(row):
        value = row.get("handover_on")
        return date.fromisoformat(str(value)[:10]).strftime("%b %Y") if value else "date not recorded"

    def line(i, row):
        bits = [row.get("community"), row.get("developer_name")]
        detail = " · ".join(b for b in bits if b)
        if when == "upcoming":
            built = row.get("percent_completed")
            tail = f"planned **{month(row)}**" + (f" · {built:.0f}% built" if built is not None else "")
        else:
            tail = f"completed **{month(row)}**"
        return f"{i}. **{row.get('name')}** — {detail} · {tail}"

    if not projects:
        answer = (f"I couldn't find any recorded {'planned' if when == 'upcoming' else 'completed'} handovers"
                  f"{where}{f' in {year}' if year else ''}.")
    elif year:
        verb = "planned to hand over" if when == "upcoming" else "completed"
        answer = (
            f"Projects{where} {verb} in **{year}** (Dubai Land Department records):\n\n"
            + "\n".join(line(i, row) for i, row in enumerate(projects, 1))
            + ("\n\nPlanned dates often slip, so check the % built." if when == "upcoming" else "")
        )
    elif when == "upcoming":
        answer = (
            f"These projects{where} have the **nearest planned handover** (Dubai Land Department project records):\n\n"
            + "\n".join(line(i, row) for i, row in enumerate(projects, 1))
            + "\n\nPlanned dates often slip, so check the % built: a project at 10% is unlikely to hand over this month."
        )
    else:
        answer = (
            f"These are the **most recently handed-over** projects{where} (Dubai Land Department completion records):\n\n"
            + "\n".join(line(i, row) for i, row in enumerate(projects, 1))
            + f"\n\nThe newest completion on record is **{month(projects[0])}**. "
            "DLD records completions after handover, so a very recent one may not appear yet."
        )
    summary = _empty_structured_summary()
    summary["projects"] = projects
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "search",
        "structured_summary": summary,
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "search",
            "structured_data": {"community": [], "projects": projects, "sub_communities": []},
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _project_count_turn(question, conversation_memory=None, understood=False):
    """How many residential projects are in Dubai — a grounded count.

    Saves dialogue_state so a follow-up "show me" can list projects (same idea
    as sales_count → show me → sales_list). `understood`: the turn resolver already
    decided this is a count, so the wording check is skipped.
    """
    if not understood and not _PROJECT_COUNT_ASK.search(question or ""):
        return None
    if _SALE_COUNT_ASK.search(question or ""):
        return None
    from app.retrieval.propai_catalog import count_projects

    try:
        n = count_projects()
    except Exception as exc:
        print(f"project count lookup failed: {exc}")
        return None
    answer = (
        f"There are **{n:,}** distinct residential projects in our Dubai data. "
        "Say **show me** if you want the list."
    )
    summary = _empty_structured_summary()
    summary["project_total"] = n
    _set_dialogue_state(
        conversation_memory,
        {
            "action": "project_count",
            "goal": None,
            "place": "Dubai",
            "beds": None,
            "budget_aed": None,
            "month": None,
            "year": None,
            "ready_only": False,
            "prefer_cheaper": False,
            "exclude_names": [],
            "offset": 0,
            "last_result": {"count": n, "project_total": n},
            "source_question": question,
        },
        replace=True,
    )
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
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _developer_rank_turn(question):
    """Top developers by recorded sale value over a lookback window."""
    if not _DEVELOPER_RANK_ASK.search(question or ""):
        return None
    from app.retrieval.propai_catalog import (
        parse_lookback_months,
        top_developers_by_value,
    )

    want = re.search(r"\b(?:top|which)\s+(\d{1,2})\b", question or "", re.I)
    limit = max(1, min(int(want.group(1)), 20)) if want else 5
    months = parse_lookback_months(question, default=12) or 12
    try:
        rows = top_developers_by_value(limit=limit, months=months)
    except Exception as exc:
        print(f"developer ranking unavailable: {exc}")
        return None
    if not rows:
        answer = (
            "I could not rank developers from recorded sales right now."
        )
    else:
        lines = [
            f"Here are the top {len(rows)} developers by recorded sales value "
            f"in Dubai over the last {months} months:",
            "",
        ]
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"{index}. **{row['developer']}** — "
                f"AED {int(row['value']):,} across {int(row['sales']):,} sales"
            )
        answer = "\n".join(lines)
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "general",
        "structured_summary": _empty_structured_summary(),
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "general",
            "structured_data": {"community": [], "projects": [], "sub_communities": []},
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _chart_turn(question, conversation_memory):
    """A bar chart of the projects on screen: recent sale price by default, or sales count or yield when asked.
    The chart data goes out as data.chart and the app draws it; the answer text is one or two lines."""
    text = question or ""
    if not _CHART_ASK.search(text):
        return None
    kind, rows = _shown_compare_rows(text, conversation_memory)
    if kind != "project" or len(rows) < 2:
        # Just listed recent sales for a place: chart those sales (price of each) instead.
        prior = _prior_sale_query(conversation_memory) or {}
        if prior.get("place"):
            from app.retrieval.propai_catalog import latest_sales
            try:
                sales = (latest_sales(place=prior["place"], limit=10, offset=0, bedrooms=prior.get("bedrooms")) or {}).get("sales") or []
            except Exception as exc:
                print(f"sales chart unavailable: {exc}")
                sales = []
            bars = [{"label": f"{str(sale.get('project') or 'Sale').title()} ({str(sale.get('sold_on') or sale.get('date') or '')[:10]})",
                     "value": float(sale["price"]), "note": None}
                    for sale in sales if sale.get("price")]
            if len(bars) >= 2:
                bars.sort(key=lambda b: -b["value"])
                place = str(prior["place"])
                result = _fast_answer(
                    question,
                    f"Here are the {len(bars)} most recent sales in {place} as a bar chart, highest price first. "
                    f"**{bars[0]['label']}** is highest at AED {int(bars[0]['value']):,}.\n\n"
                    "Want to see more transactions, or only one bedroom count?",
                    question_type="general",
                )
                result["structured_summary"]["chart"] = {"title": f"Recent sales in {place}", "unit": "aed", "bars": bars,
                                                         "caption": "Each bar is one recorded sale, with its date."}
                return result
        return _fast_answer(question, "Ask me something first, for example recent sales in Dubai Marina, "
                                      "and then I can draw it as a bar chart.")

    def number(row, *keys):
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return None

    if re.search(r"\b(?:yields?|returns?|roi)\b", text, re.I):
        metric, title, unit, keys = "yield", "Gross rental yield", "percent", ("gross_yield_pct",)
    elif re.search(r"\b(?:sales?|sold|volume|activity|busiest|most)\b", text, re.I):
        metric, title, unit, keys = "sales", "Sales recorded in the last 12 months", "count", ("sales_count", "sales")
    else:
        metric, title, unit, keys = "price", "Recent sale price", "aed", ("price",)
    bars = [{"label": str(row.get("name")), "value": number(row, *keys),
             "note": (f"Recorded {str(row.get('last_sold_on') or row.get('last_sale'))[:10]}"
                      if metric == "price" and (row.get("last_sold_on") or row.get("last_sale")) else None)}
            for row in rows]
    charted = sorted((b for b in bars if b["value"] is not None), key=lambda b: -b["value"])[:12]
    left_out = len(bars) - len(charted)
    if len(charted) < 2:
        return _fast_answer(question, f"Most of these projects have no {title.lower()} on record, so there is not enough to chart. "
                                      "Try recent sale prices instead.")
    unit_text = {"aed": lambda v: f"AED {int(v):,}", "count": lambda v: f"{int(v):,} sales", "percent": lambda v: f"{v:.1f}%"}[unit]
    answer = (f"Here are the {len(charted)} projects as a bar chart, highest first. **{charted[0]['label']}** is highest at "
              f"{unit_text(charted[0]['value'])} and **{charted[-1]['label']}** lowest at {unit_text(charted[-1]['value'])}."
              + (f" {left_out} more have no figure on record, so they are not drawn." if left_out else "")
              + ("\n\nWant it by recent sale price instead, or compare two of these side by side?" if metric == "sales"
                 else "\n\nWant it by sales count instead, or compare two of these side by side?"))
    result = _fast_answer(question, answer, question_type="general")
    result["structured_summary"]["chart"] = {"title": title, "unit": unit, "bars": charted,
                                             "caption": "Median of each project's recorded sales in the last 12 months." if metric == "price" else None}
    return result


def _yield_projects_turn(question, conversation_memory=None):
    """Projects ranked by gross rental yield, from recorded sales and rent contracts. Only for a ranking ask
    ("best ... high rental yield"); the yield of one named place goes the existing area route."""
    text = question or ""
    if not (_YIELD_RANK_ASK.search(text) and _YIELD_RANK_NOUN.search(text) and _YIELD_RANK_WORDS.search(text)):
        return None
    if re.search(r"\b(?:compare|versus|vs)\b", text, re.I):
        return None
    from app.retrieval.propai_catalog import resolve_dubai_place, top_yield_projects

    place = extract_place(text)
    if place and str(place).strip().lower() == "dubai":
        place = None
    try:
        resolved = (resolve_dubai_place(place) or place) if place else None
    except Exception:
        resolved = place
    beds = parse_bedrooms(text)
    budget = parse_budget_aed(text)
    low = high = None
    if budget is not None:
        if _PRICE_FLOOR_ASK.search(text) and not _PRICE_CEILING_ASK.search(text):
            low = budget
        else:
            high = budget
    try:
        rows, total = top_yield_projects(max_price=high, min_price=low, place=resolved,
                                         bedrooms=beds[0] if beds else None, limit=6)
    except Exception as exc:
        print(f"yield ranking unavailable: {exc}")
        return None
    bits = (f" {_bed_label(beds[0])}" if beds else "") + _where(resolved)
    if high is not None:
        bits += f" under AED {int(high):,}"
    if low is not None:
        bits += f" above AED {int(low):,}"
    if not rows:
        answer = (f"I found no projects{bits} with enough recorded sales and rent contracts to measure a rental yield. "
                  "Try a wider budget or another community.")
    else:
        answer = (
            f"I found **{total:,}** projects{bits} with enough recorded sales and rents to measure a gross yield. "
            f"Here are the top {len(rows)}, highest yield first.\n\n"
            "Gross yield is the median yearly rent of recent contracts divided by the median recent sale price. It is a guide "
            "for comparing projects, before service charges, vacancy and fees, and it mixes unit types. "
            "Want me to compare these, or narrow it to a community or bedroom count?"
        )
    return _fast_answer(question, answer, projects=rows, question_type="search")


def _sql_project_browse_turn(question, conversation_memory=None):
    """Clear community project browse via Prop-AI SQL only — no rewrite/plan/agent.

    "Show me projects in JVC" and "2 bedroom apartments in Marina" were still
    paying hybrid planner / rewrite latency even though the answer is SQL.

    Phase 3: buy / best / family / afford stay on the recommend path — never
    answer those with a plain facts list.
    """
    text = question or ""
    from app.answers.intent_route import is_recommend_intent

    if is_recommend_intent(text, conversation_memory):
        return None
    if not re.search(
        r"\b(?:show|find|list|projects?|propert(?:y|ies)|apartments?|villas?|"
        r"townhouses?|flats?)\b",
        text,
        re.I,
    ):
        return None
    # Sales / sold asks are transaction answers, never a project gallery.
    if _TRANSACTION.search(text) or re.search(r"\b(?:sales?|sold)\b", text, re.I):
        return None
    if _PROJECT_COUNT_ASK.search(text) or _DEVELOPER_RANK_ASK.search(text):
        return None
    if re.search(
        r"\b(?:agents?|brokers?|compare|versus|\bvs\b|brochure|floor ?plans?|"
        r"invest(?:ment|ing)?|yield|roi)\b",
        text,
        re.I,
    ):
        return None
    if is_incomplete_home_browse(text):
        return None
    if asks_for_listings(text):
        return None

    place = extract_place(text)
    budget = parse_budget_aed(text)
    citywide = bool(
        re.search(r"\bin\s+dubai\b|\bdubai\b", text, re.I)
        and (not place or str(place).strip().lower() == "dubai")
    )
    if not place and not citywide:
        # Budget-only citywide: "projects starting above AED 5 million"
        if not (budget and re.search(r"\b(?:projects?|propert(?:y|ies)|apartments?)\b", text, re.I)):
            return None
        citywide = True

    from app.retrieval.propai_catalog import (
        browse_projects_near_budget,
        resolve_dubai_place,
    )

    try:
        resolved = resolve_dubai_place(place) or place if place else None
    except Exception:
        resolved = place
    if resolved and str(resolved).strip().lower() == "dubai":
        resolved = None
        citywide = True
    beds = parse_bedrooms(text)
    payload = {}
    if resolved:
        payload["communities"] = [resolved]
    if beds:
        payload["bedrooms"] = beds[0]
    from app.retrieval.propai_catalog import developers_in_question

    developers = developers_in_question(text)
    if developers:
        payload["developers"] = developers
    if _VILLA_ASK.search(text):
        payload["property_types"] = ["villa"]
    if budget is not None:
        if _PRICE_FLOOR_ASK.search(text) and not _PRICE_CEILING_ASK.search(text):
            payload["min_price"] = budget
        else:
            payload["max_price"] = budget
    if _READY_ONLY_ASK.search(text):
        payload["ready_only"] = True

    skip = []
    if wants_more_results(text):
        items = getattr(conversation_memory, "active_candidates", None) or []
        skip = [
            str(item.get("name") or "").strip()
            for item in items
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]

    if skip:
        sort = "more"
    elif payload.get("min_price") is not None:
        sort = "priciest"
    elif payload.get("max_price") is not None or beds:
        sort = "cheapest"
    else:
        sort = "latest_handover"

    try:
        rows, total, budget_meta = browse_projects_near_budget(
            payload,
            limit=10,
            sort=sort,
            skip_names=skip or None,
        )
    except Exception as exc:
        print(f"SQL project browse unavailable: {exc}")
        return None
    if not rows:
        # Honest empty answer for ready/budget filters instead of falling to the LLM.
        if (payload.get("ready_only") or payload.get("min_price") or payload.get("max_price")
                or payload.get("developers") or payload.get("property_types")):
            where = resolved or "Dubai"
            bits = []
            if payload.get("ready_only"):
                bits.append("ready")
            if beds:
                bits.append(_bed_label(beds[0]))
            if payload.get("property_types"):
                bits.append("villa / townhouse")
            if payload.get("developers"):
                bits.append("by " + " or ".join(d.title() for d in payload["developers"]))
            bits.append(f"in {where}")
            if payload.get("max_price") is not None:
                bits.append(f"under AED {int(payload['max_price']):,}")
            if payload.get("min_price") is not None:
                bits.append(f"starting above AED {int(payload['min_price']):,}")
            relax = [opt for cond, opt in (
                (payload.get("developers"), "include other developers"),
                (payload.get("property_types"), "include apartments"),
                (beds, "change the number of bedrooms"),
                (payload.get("max_price") is not None, "raise the budget"),
                (payload.get("ready_only"), "include off-plan"),
                (resolved, "look in nearby communities"),
            ) if cond]
            answer = (
                "I found no matching projects for "
                + " ".join(bits)
                + " in the last 12 months of recorded sales."
                # ACOT QA R07: when filters exclude everything, offer the one to relax instead of a dead end.
                + (f"\n\nTo widen the search I can relax one filter: {', '.join(relax[:-1]) + ' or ' + relax[-1] if len(relax) > 1 else relax[0]}. "
                   "Which would you like me to change?" if relax else "")
            )
            summary = _empty_structured_summary()
            summary["project_total"] = 0
            if resolved:
                summary["catalog_place"] = str(resolved).strip()
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
                        "projects": [],
                        "sub_communities": [],
                    },
                    "documents": [],
                },
                "use_fast": True,
                "fast_answer": answer,
                "context": None,
                "small_talk": None,
            }
        return None

    summary = _empty_structured_summary()
    summary["projects"] = rows
    summary["catalog_place"] = str(resolved or "Dubai").strip()
    applied = [b for b in (
        "ready only" if payload.get("ready_only") else None,
        _bed_label(beds[0]) if beds else None,
        "villa / townhouse" if payload.get("property_types") else None,
        ("developer " + " or ".join(d.title() for d in payload["developers"])) if payload.get("developers") else None,
        f"in {resolved}" if resolved else None,
        f"under AED {int(payload['max_price']):,}" if payload.get("max_price") is not None else None,
        f"above AED {int(payload['min_price']):,}" if payload.get("min_price") is not None else None,
    ) if b]
    summary["applied_filters"] = applied
    summary["split_launches"] = bool(payload.get("developers") or _SPLIT_ASK.search(text))
    if _CONSTRAINT_ASK.search(text):
        getter = getattr(conversation_memory, "get_dialogue_state", None)
        prior = (getter() if callable(getter) else getattr(conversation_memory, "dialogue_state", None)) or {}
        dropped = [b for b in (
            f"in {prior['place']}" if prior.get("place") and prior.get("place") != resolved else None,
            _bed_label(prior["beds"]) if prior.get("beds") is not None and (not beds or prior["beds"] != beds[0]) else None,
            f"budget AED {int(prior['budget_aed']):,}" if prior.get("budget_aed") and prior.get("budget_aed") != budget else None,
        ) if b]
        summary["constraint_note"] = (
            "**Kept from your request:** " + (", ".join(applied) or "no filters") + ". "
            + ("**Replaced from the last search:** " + ", ".join(dropped) + ", because you changed them."
               if dropped else "Nothing else was carried over from an earlier search.")
        )
    if budget_meta and budget_meta.get("budget_miss"):
        summary["budget_miss"] = True
        summary["asked_max_price"] = budget_meta.get("asked_max_price")
        summary["lowest_price"] = budget_meta.get("lowest_price")
        summary["project_total"] = len(rows)
    else:
        summary["project_total"] = total or len(rows)
    _set_dialogue_state(
        conversation_memory,
        {
            "action": "search",
            "place": resolved or ("Dubai" if citywide else None),
            "beds": beds[0] if beds else None,
            "budget_aed": budget,
            "month": None,
            "year": None,
            "ready_only": bool(payload.get("ready_only")),
            "prefer_cheaper": sort == "cheapest",
            "exclude_names": skip or [],
            "offset": 0,
            "last_result": {"project_total": summary["project_total"]},
            "source_question": question,
            "goal": None,
        },
        replace=True,
    )
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
                "projects": rows,
                "sub_communities": [],
            },
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": _build_fast_structured_answer(question, summary),
        "context": None,
        "small_talk": None,
    }


def _community_list_turn(question):
    text = question or ""
    if not _COMMUNITY_LIST.search(text) or _COMMUNITY_NARROWED.search(text):
        return None
    from app.database.postgres import fetch
    try:
        # Prop-AI only — never the retired Bayut `listings` table.
        projects = fetch(
            "select coalesce(nullif(btrim(master_project), ''), area) as name, count(*)::int as n "
            "from propai_projects "
            "where coalesce(name, '') <> '' "
            "and coalesce(nullif(btrim(master_project), ''), area) is not null "
            "group by 1"
        )
        places = fetch(
            "select btrim(label) as name, coalesce(sales_count, 0)::int as n "
            "from propai_places where level in ('area', 'master') "
            "and coalesce(btrim(label), '') <> ''"
        )
    except Exception:
        return None
    rows: dict[str, dict] = {}
    for kind, found in (("projects", projects), ("sales", places)):
        for row in found or []:
            key = re.sub(r"\s*\([^)]*\)", "", row["name"]).strip().lower()
            entry = rows.setdefault(key, {"name": row["name"], "projects": 0, "sales": 0})
            entry[kind] += row["n"]
    if not rows:
        return None
    ordered = sorted(rows.values(), key=lambda r: (-r["projects"], -r["sales"], r["name"]))
    lines = [
        f"We have {len(ordered)} communities in Prop-AI. Here is the full list, busiest first:",
        "",
        "| Community | New projects | Sales signal |",
        "|---|---|---|",
    ]
    for row in ordered:
        lines.append(f"| {row['name']} | {row['projects'] or '-'} | {row['sales'] or '-'} |")
    lines += ["", "Ask about any of them for prices, sales or projects, for example \"Tell me about Dubai Marina\"."]
    return {
        "question": question,
        "standalone_question": question,
        "question_type": "general",
        "structured_summary": _empty_structured_summary(),
        "documents": [],
        "investment_analysis": None,
        "retrieval_result": {
            "question_type": "general",
            "structured_data": {"community": [], "projects": [], "sub_communities": []},
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": "\n".join(lines),
        "context": None,
        "small_talk": None,
    }


def _broker_turn(question, conversation_memory):
    question = repair_typed_question(question)
    if not _is_broker_question(question, conversation_memory):
        return None
    from app.retrieval.propai_catalog import is_outside_dubai, known_dubai_label, list_brokers, resolve_dubai_place
    blocked = _outside_coverage_reply(question)
    if blocked:
        return blocked
    place = _broker_place(question, conversation_memory)
    if place and is_outside_dubai(place):
        return _outside_coverage_reply(question, place.strip())
    if place:
        needle = place.strip().lower()
        on_screen = False
        for item in getattr(conversation_memory, "active_candidates", None) or []:
            if not isinstance(item, dict):
                continue
            if needle in {
                str(item.get("community") or "").strip().lower(),
                str(item.get("name") or "").strip().lower(),
            }:
                on_screen = True
                break
        if not on_screen:
            resolved = resolve_dubai_place(place)
            if resolved is None:
                # Retry after budget noise (AED / starting above / million).
                cleaned = _strip_broker_budget(place)
                resolved = resolve_dubai_place(cleaned) if cleaned else None
            if resolved is None:
                return _broker_reply(
                    question,
                    "I will show only licensed agents in Dubai. "
                    "Ask about an area such as Dubai Marina, Business Bay, or Palm Jumeirah.",
                )
            place = resolved
        # Whole-market Dubai: list top Prop-AI agents citywide (not a fake community).
        if str(place).strip().lower() == "dubai":
            place = "Dubai"
    if _broker_identity_unsupported(question):
        nationality, gender, _both = _broker_identity_filters(question)
        wanted = []
        if nationality:
            wanted.append(nationality)
        if gender:
            wanted.append(gender)
        label = " and ".join(wanted) if wanted else "nationality or gender"
        answer = (
            "Our DLD agent records don't include nationality or gender, "
            f"so I can't show only {label} agents."
        )
        return _broker_reply(question, answer)
    name, brn = _broker_lookup(question, conversation_memory)
    if name and not brn and (is_outside_dubai(name) or known_dubai_label(name)):
        name = None
    follow_up = bool(place or re.search(r"\b(?:this|these|those|for)\b", question or "", re.I))
    limit = 5 if follow_up else 8
    band = _broker_band(conversation_memory, place) if follow_up and not name and not brn else None
    # "top / best agents" → Five Star Prop-AI ranks even for citywide Dubai.
    if not name and not brn and re.search(r"\b(?:top|best|five[\s-]?star)\b", question or "", re.I):
        band = "top"
    rows = list_brokers(name=name, brn=brn, limit=limit, place=place, band=band)
    if name and not rows:
        rows = list_brokers(limit=limit, place=place, band=band)
        label = place
        answer = (
            f'I couldn\'t find a licensed agent matching "{name}" in our DLD data. '
            + _format_brokers(rows, label=label)
        )
    else:
        answer = _format_brokers(rows, label=place)
    # Cards render each agent — keep the rich lead, drop the duplicated numbered list.
    if rows:
        kept = []
        for part in answer.split("\n\n"):
            if re.match(r"^\d+\.\s+\*\*", (part or "").strip()):
                break
            kept.append(part)
        answer = "\n\n".join(kept).rstrip()
        if "Want agents for a nearby" not in answer:
            answer += (
                "\n\nWant agents for a nearby community, or should I match projects "
                "in the same area next?"
            )
    return _broker_reply(question, answer, rows)


_SHORTLIST_RECOMMEND = re.compile(
    r"\b(?:"
    r"which (?:one |of these |of those )?(?:is|are) (?:the )?best|"
    r"which of these (?:is|are) better|"
    r"which (?:of these |ones? |of those )?(?:should|would) i|"
    r"what should i (?:pick|choose|buy)|"
    r"which (?:project|property) is (?:the )?best|"
    r"tell(?: me)? which (?:project|property|one) is (?:the )?best|"
    r"best (?:property|project|apartment|option) to (?:invest|buy|live)|"
    r"best to live|"
    r"tell(?: me)? (?:which |the )?best|"
    r"recommend (?:one|one of these|the best|among these)|"
    r"best (?:one|option|pick|project|apartment)(?:\s+of these)?|"
    r"pick (?:one|for me)|"
    r"which of these (?:do you )?recommend"
    r")\b",
    re.I,
)
_COMPARE_WORD = re.compile(r"\bcompar(?:e|ison|ing)\b", re.I)
_COMPARE_ALL = re.compile(r"\b(?:all|every|each)\b", re.I)
_SHOWN_RENTALS = re.compile(
    r"\b(?:apartments?|rentals?|listings?|studios?|to live)\b",
    re.I,
)
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
        and not _is_listing_row(item)
    ]


def _is_listing_row(item: dict) -> bool:
    kind = str(item.get("entity_type") or "").lower()
    if kind == "listing":
        return True
    if kind == "project":
        return False
    return bool(item.get("listing_id") or item.get("rent_frequency") or item.get("title"))


def _active_listings(conversation_memory) -> list[dict]:
    items = getattr(conversation_memory, "active_candidates", None) or []
    rows = []
    for item in items:
        if not isinstance(item, dict) or not _is_listing_row(item):
            continue
        if item.get("name") or item.get("title"):
            rows.append(item)
    return rows


def _rankable_listing(item: dict) -> dict:
    beds = item.get("beds")
    rent = item.get("rent_frequency") or item.get("price_kind")
    size = item.get("size_sqft") or item.get("size")
    if isinstance(size, dict):
        size = size.get("min") if size.get("min") is not None else size.get("max")
    property_type = item.get("property_type")
    if not property_type:
        types = item.get("property_types") or []
        if isinstance(types, list):
            property_type = next((value for value in types if value), None)
        elif types:
            property_type = types
    return {
        **item,
        "id": item.get("id") or item.get("listing_id"),
        "name": item.get("name") or item.get("title"),
        "price": item.get("price") or item.get("price_aed"),
        "price_kind": item.get("price_kind") or ("yearly rent" if rent else "price"),
        "size_sqft": size,
        "baths": item.get("baths") if item.get("baths") is not None else item.get("bathrooms"),
        "property_type": property_type,
        "bedroom_min": item.get("bedroom_min") if item.get("bedroom_min") is not None else beds,
        "bedroom_max": item.get("bedroom_max") if item.get("bedroom_max") is not None else beds,
        "entity_type": "listing",
    }


def _shown_compare_rows(question, conversation_memory) -> tuple[str, list[dict]]:
    projects = _active_projects(conversation_memory)
    listings = _active_listings(conversation_memory)
    rent_ask = bool(_SHOWN_RENTALS.search(question or ""))
    if listings and (rent_ask or len(projects) < 2):
        return "listing", [_rankable_listing(row) for row in listings]
    if len(projects) >= 2:
        return "project", projects
    if len(listings) >= 2:
        return "listing", [_rankable_listing(row) for row in listings]
    return "", []


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


def _on_screen_compare_or_pick(question, conversation_memory) -> bool:
    """True when the ask ranks/compares live cards (not explain/why follow-ups)."""
    _kind, available = _shown_compare_rows(question, conversation_memory)
    if len(available) < 2:
        return False
    # Compare-all / first-N table asks.
    if _shown_compare_limit(question):
        return True
    # "tell me the best" / which-is-best picks — not "why are these good".
    if _SHORTLIST_RECOMMEND.search(question or ""):
        return True
    return False


def _compare_shown_turn(question, conversation_memory):
    """Compare N (or all) on-screen projects or rentals in a table + Acot Recommend."""
    limit = _shown_compare_limit(question)
    if not limit:
        return None
    if _TRANSACTION.search(question or ""):
        return None
    kind, available = _shown_compare_rows(question, conversation_memory)
    rows = available[:limit]
    if len(rows) < 2:
        return None
    from app.answers.recommend import format_compare_table, rank_projects, wanted_bedrooms

    # Keep every on-screen option in the table; evidence still informs the score.
    ranked = rank_projects(
        rows,
        wanted_beds=wanted_bedrooms(question),
        attach_sales=(kind == "project"),
        goal="compare",
        require_evidence=False,
    )
    answer = format_compare_table(rows, ranked)
    summary = _empty_structured_summary()
    # Cards show the compared set; memory keeps the fuller shortlist via preserve logic.
    if kind == "listing":
        summary["listings"] = rows
        summary["listing_total"] = len(rows)
        data = {"community": [], "projects": [], "sub_communities": [], "listings": rows}
    else:
        summary["projects"] = rows
        summary["project_total"] = len(rows)
        data = {"community": [], "projects": rows, "sub_communities": []}
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
            "structured_data": data,
            "documents": [],
        },
        "use_fast": True,
        "fast_answer": answer,
        "context": None,
        "small_talk": None,
    }


def _shortlist_recommend_turn(question, conversation_memory):
    """Rank projects or rentals already on screen and return Acot Recommend."""
    if _shown_compare_limit(question):
        return None
    if _TRANSACTION.search(question or ""):
        return None
    try:
        from app.answers.turn_act import is_casual_speech

        if is_casual_speech(question):
            return None
    except Exception:
        pass
    from app.answers.intent_route import classify_intent

    route = classify_intent(question, conversation_memory)
    # "best project in Dubai Marina for family" is a fresh place-scoped recommend.
    # Do not re-rank whatever cards happen to be on screen just because the
    # sentence contains "best project".
    kind = str(route.get("kind") or "")
    subject = str(route.get("subject") or "").strip().lower()
    if (
        kind in {
            "filtered", "afford", "cheapest", "family_areas", "open_best",
            "yield_areas", "ready_offplan", "split",
        }
        and subject
        and subject != "on_screen"
    ):
        return None
    pick = bool(_SHORTLIST_RECOMMEND.search(question or ""))
    explain = bool(
        re.search(r"\b(?:why|explain|good for|suit|fit)\b", question or "", re.I)
    )
    # Attribute asks ("which of these have 2 beds") stay off this path.
    if not pick and not explain and route.get("kind") != "shortlist":
        return None
    if not pick and not explain:
        # classify shortlist alone is not enough unless it is a pick/explain ask.
        return None
    kind, available = _shown_compare_rows(question, conversation_memory)
    if len(available) < 2:
        return None
    from app.answers.recommend import (
        format_explain_on_screen,
        format_shortlist,
        rank_projects,
        wanted_bedrooms,
    )

    selected = available[:_COMPARE_ALL_LIMIT]
    state = {}
    if conversation_memory is not None:
        if hasattr(conversation_memory, "get_dialogue_state"):
            state = conversation_memory.get_dialogue_state() or {}
        else:
            state = getattr(conversation_memory, "dialogue_state", None) or {}
    # Prefer the ask's goal over stale browse state (e.g. "best to invest").
    q = question or ""
    if re.search(r"\binvest\b", q, re.I):
        goal = "invest"
    elif re.search(r"\bfamil", q, re.I):
        goal = "family"
    elif re.search(r"\b(?:afford|budget|cheap)\b", q, re.I):
        goal = "budget"
    elif re.search(r"\b(?:ready|handover|completed)\b", q, re.I):
        goal = "ready"
    else:
        goal = state.get("goal") or "compare"
    ranked = rank_projects(
        selected,
        wanted_beds=wanted_bedrooms(question) or state.get("beds"),
        budget_aed=state.get("budget_aed"),
        attach_sales=(kind == "project"),
        goal=goal,
    )
    # Compare-style "which is best" with many options still gets a table when asked to compare.
    if _COMPARE_WORD.search(question or ""):
        from app.answers.recommend import format_compare_table
        answer = format_compare_table(selected, ranked)
    elif re.search(r"\b(?:why|explain|good for|suit|fit)\b", question or "", re.I):
        goal = state.get("goal")
        if not goal and re.search(r"\bfamil", question or "", re.I):
            goal = "family"
        answer = format_explain_on_screen(
            ranked,
            goal=goal,
            place=state.get("place"),
            beds=state.get("beds"),
        )
    else:
        answer = format_shortlist(ranked, question=question)
    summary = _empty_structured_summary()
    ordered = []
    for item in ranked.get("ranked") or []:
        row = item.get("row")
        if row is not None and row not in ordered:
            ordered.append(row)
    if kind == "listing":
        summary["listings"] = ordered
        summary["listing_total"] = len(ordered)
        data = {"community": [], "projects": [], "sub_communities": [], "listings": ordered}
    else:
        summary["projects"] = ordered
        summary["project_total"] = len(ordered)
        data = {"community": [], "projects": ordered, "sub_communities": []}
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
            "structured_data": data,
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
    After a sales_list answer there are no project cards — use last_sale_projects.
    """
    try:
        from app.answers.turn_resolve import _is_shown_attribute_ask
    except Exception:
        _is_shown_attribute_ask = None
    if _is_shown_attribute_ask is not None:
        if not _is_shown_attribute_ask(question, conversation_memory):
            return None
    elif not is_group_attribute_question(question):
        return None
    items = getattr(conversation_memory, "active_candidates", None) or []
    projects = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("entity_type") or "project").lower() == "project"
        and item.get("name")
    ]
    from_sales = False
    if not projects:
        sale_stubs = getattr(conversation_memory, "last_sale_projects", None) or []
        projects = [
            item for item in sale_stubs
            if isinstance(item, dict) and item.get("name")
        ]
        from_sales = bool(projects)
    if not projects:
        return None
    # Named community that does not match the shortlist → leave to area browse.
    try:
        from app.answers.intent_route import extract_place

        named = extract_place(question)
    except Exception:
        named = None
    if named and str(named).strip().lower() not in {"", "dubai"}:
        needle = str(named).strip().lower()
        same_place = any(
            needle in " ".join(
                str(project.get(key) or "")
                for key in ("community", "sub_community", "area", "name")
            ).lower()
            for project in projects
        )
        if not same_place:
            return None
    # Sales stubs usually lack developer_name — hydrate from Prop-AI.
    if from_sales or any(
        not str(p.get("developer_name") or p.get("developer") or "").strip()
        for p in projects
    ):
        # Only hydrate when the ask needs developer (or stubs are name-only).
        if re.search(r"\bdevelopers?\b", question or "", re.I) or from_sales:
            projects = _hydrate_projects_with_developers(projects)
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


def _shown_attribute_response(question, conversation_memory, timing_ms, ask_started, rag_chain):
    """Fast structured answer for price/handover/bedroom asks about on-screen projects."""
    shown_followup = _shown_project_followup(question, conversation_memory)
    if shown_followup is None:
        return None
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


def _classify_small_talk(question):
    """Return a small-talk kind when the whole message is a greeting or identity question."""

    text = _normalize_chat_text(question)
    kind = _SMALL_TALK_KINDS.get(text)
    if kind:
        return kind

    words = re.findall(r"[a-z']+", text)
    if not words or len(words) > 8:
        # Still allow longer polite thanks via casual gate.
        try:
            from app.answers.turn_act import casual_kind, is_casual_speech

            if is_casual_speech(question):
                return casual_kind(question)
        except Exception:
            pass
        return None
    joined = " ".join(words)
    # Greetings / thanks leads first (preserve "Hello. So, uh," → greeting).
    for pattern, lead_kind in _SMALL_TALK_LEADS:
        if pattern.match(words[0]):
            rest = words[1:]
            if all(word in _SMALL_TALK_FILLER for word in rest):
                return lead_kind
            # "Hello. So, uh," / "hi there please" with no real ask → greeting.
            if len(words) <= 8 and not _SMALL_TALK_REAL_ASK.search(joined):
                return lead_kind
            return None
    # Casual speech gate (ack / cool / got it) — before any follow-up or data path.
    try:
        from app.answers.turn_act import casual_kind, is_casual_speech

        if is_casual_speech(question):
            return casual_kind(question)
    except Exception:
        pass
    # Casual thanks / soft acknowledgements that should not hit retrieval.
    if re.fullmatch(
        r"(?:ok(?:ay)?|alright|sure|yes|yep|yeah)?\s*"
        r"(?:thanks?|thx|ty|thank you)(?:\s+\w+){0,3}",
        joined,
    ):
        return "thanks"
    ack_words = {
        "ok", "okay", "alright", "sure", "yes", "yep", "yeah", "no", "nope",
        "nah", "hmm", "hmmm", "interesting", "really", "got", "it", "cool",
        "nice", "maybe", "later", "not", "now",
        "great", "perfect", "awesome", "amazing", "fine", "good",
    }
    if words and len(words) <= 3 and all(word in ack_words for word in words):
        return "ack"
    return None


def _small_talk_answer(kind, name=None, audience="buyer"):
    """name: the signed-in user's first name, when the app sent it.
    audience: buyer (default) or broker — Nest sends broker for agent Ask ACOT."""
    who = f" {name}" if name else ""
    broker = (audience or "buyer") == "broker"

    if kind == "identity":
        if broker:
            return (
                "I'm ACOT, your CRM companion. I can read your leads and pipeline, "
                "help you decide what to claim, and answer Dubai market questions — "
                "I never message a buyer without you."
            )
        return (
            "I'm ACOT, an AI assistant for Dubai real estate. I answer from "
            "project records and documents, including prices, developers, "
            "bedroom ranges, handover dates, and amenities."
        )

    if kind == "capabilities":
        if broker:
            return (
                "Ask me about follow-ups and your pipeline, draft a message to a lead, "
                "which projects to claim, or Dubai areas, prices and projects. "
                "Nothing is sent to a buyer unless you press Send."
            )
        return (
            "I can help you with Dubai project data:\n\n"
            "- Find projects in a community\n"
            "- List starting prices, bedroom ranges, and handover dates\n"
            "- Compare projects already on screen\n"
            "- Pull amenities from their documents"
        )

    if kind in ("GoodMorning", "Good Morning", "Morning"):
        if broker:
            return f"Good morning{who}. Ask about follow-ups, your pipeline, or the Dubai market."
        return (
            f"Good morning{who}, How can I help you with Dubai real estate today?"
        )

    if kind in ("GoodAfternoon", "Good Afternoon", "Afternoon"):
        if broker:
            return f"Good afternoon{who}. Ask about follow-ups, your pipeline, or the Dubai market."
        return (
            f"Good afternoon{who}, How can I help you with Dubai real estate today?"
        )

    if kind in ("GoodEvening", "Good Evening", "Evening"):
        if broker:
            return f"Good evening{who}. Ask about follow-ups, your pipeline, or the Dubai market."
        return (
            f"Good evening{who}, How can I help you with Dubai real estate today?"
        )

    if kind == "thanks":
        if broker:
            return (
                f"You're welcome{who}. Ask about a lead, your pipeline, or the market "
                "whenever you're ready."
            )
        return (
            f"You're welcome{who}. Ask me about a Dubai community or project "
            "whenever you're ready."
        )

    if kind == "ack":
        return (
            f"Glad that helped{who}. "
            "Ask anytime if you want another area, agents, recent sales, or a tighter shortlist."
        )

    if kind == "bye":
        if broker:
            return f"Goodbye{who}. I'll be here when you want to check leads or the market again."
        return (
            f"Goodbye{who}. Have a great day! I'll be here when you want to look at Dubai projects again."
        )

    if broker:
        return (
            f"Hello{who}, I'm ACOT — ask about follow-ups, your pipeline, what to claim, "
            "or the Dubai market."
        )
    return f"Hello{who}, I'm ACOT, a Dubai real estate assistant. How can I help you?"


def _empty_structured_summary():
    return {
        "community": [],
        "projects": [],
        "sub_communities": [],
        "listings": [],
        "brokers": [],
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


# Dubai buying/renting process terms the agent may answer from general knowledge.
# Hybrid RAG refuses these ("only ACOT evidence"), so they must not skip the agent.
_GUIDANCE_TOPIC = re.compile(
    r"\b(?:"
    r"rera|ejari|oqood|escrow|freehold|leasehold|"
    r"dld(?:\s+fees?)?|dubai\s+land\s+department|"
    r"title\s+deeds?|sale\s+(?:and\s+|\&\s+)?purchase\s+agreement|"
    r"payment\s+plans?|mortgages?|off[\s-]?plan|"
    r"service\s+charges?|transfer\s+fees?|"
    r"how\s+(?:do\s+i|to|does\s+one)\s+(?:buy|purchase|rent)|"
    r"buying\s+process|purchase\s+process|steps?\s+to\s+buy"
    r")\b",
    re.I,
)
_GUIDANCE_DATA_ASK = re.compile(
    r"\b(?:show\s+me|find|list|projects?|listings?|apartments?|villas?|"
    r"townhouses?|price|prices|handover|bedrooms?|agents?|brokers?)\b",
    re.I,
)


def _agent_owns_turn(components) -> bool:
    """The understanding step called this a guidance or off-topic question: the agent answers it,
    never the default project-list browse."""
    return isinstance(components, dict) and components.get("_resolved_action") in {"guidance", "general"}


def _is_dubai_guidance_question(question: str) -> bool:
    """True for process/knowledge asks (RERA, escrow, how to buy) — not data browse."""
    text = question or ""
    if not _GUIDANCE_TOPIC.search(text):
        return False
    # "off-plan projects in Marina" / "RERA agents" stay on their data paths.
    if _GUIDANCE_DATA_ASK.search(text):
        return False
    return True


def _propai_sql_browse(question, conversation_memory=None) -> bool:
    """True when Prop-AI SQL / hybrid should answer, not the listings LLM.

    Clear project browses (show me 2 beds in Marina) stay on SQL — cheaper and
    accurate. Agent keeps messy / guidance / incomplete-clarify turns.
    """
    if asks_for_listings(question):
        return False
    text = question or ""
    if re.search(r"\b(?:agents?|brokers?)\b", text, re.I):
        return False
    # Sales / sold / transactions are handled by _transaction_turn, not the agent.
    if _TRANSACTION.search(text):
        return False
    if _is_dubai_guidance_question(text):
        return False
    if _is_definitional_question(text):
        return False
    if is_incomplete_home_browse(text):
        return False
    if is_no_preference_reply(text):
        return False
    items = []
    if conversation_memory is not None:
        items = getattr(conversation_memory, "active_candidates", None) or []
    if items and is_group_attribute_question(question):
        return False
    return True


def _seed_agent_history_from_memory(history, conversation_memory):
    """Copy on-screen cards into agent history after a structured browse.

    The listings agent only knows items from its own '[internal note]'. A
    Prop-AI SQL list never writes that note, so 'show me more' had nothing
    to exclude.
    """
    if history:
        return
    items = getattr(conversation_memory, "active_candidates", None) or []
    shown = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = item.get("id") or item.get("listing_id") or item.get("project_id")
        shown.append(f"{name} (id {key})" if key else name)
    if not shown:
        return
    prior = (
        getattr(conversation_memory, "last_user_question", None)
        or getattr(conversation_memory, "last_standalone_question", None)
        or "previous search"
    )
    history.append({"role": "user", "content": str(prior)})
    history.append({
        "role": "assistant",
        "content": (
            "Here are the options on screen.\n"
            "[internal note, not part of the reply. Shown: "
            + "; ".join(shown)
            + "]"
        ),
    })


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
    # Simple Prop-AI browse is SQL, not the listings agent. The agent was
    # retrying search_projects until the round limit and burning tokens.
    if not _agent_owns_turn(components) and _propai_sql_browse(question, conversation_memory):
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
    _seed_agent_history_from_memory(history, conversation_memory)
    llm = components["rag_chain"].llm

    if _is_broker_question(question, conversation_memory):
        blocked = _outside_coverage_reply(question)
        if blocked:
            blocked["timing_ms"] = timing_ms
            blocked["ask_started"] = ask_started
            blocked["rag_chain"] = components.get("rag_chain")
            blocked["conversation_memory"] = conversation_memory
            return blocked

    started = time.perf_counter()
    # Same concurrency cap as every other model call, so the provider is not flooded.
    await llm._acquire_slot()
    try:
        result = await asyncio.to_thread(
            agent.answer, question, list(history), None, profile, _agent_results_hook.get(),
            components.get("page"), components.get("audience"),
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
    if "couldn't pin that down" in (result.answer or "").lower() and not result.cards:
        return None

    projects, listings, brokers, labels = [], [], [], {}
    for card in result.cards:
        if card.get("kind") == "broker":
            row = card.get("row") or {}
            brokers.append(row)
            labels[card["listing_id"]] = row.get("name") or row.get("name_en") or "this agent"
        elif card.get("kind") == "project":
            row = card["row"]
            projects.append(row)
            labels[card["listing_id"]] = row.get("name") or "this project"
        else:
            listings.append(_agent_listing_record(card))
            labels[card["listing_id"]] = card.get("title") or "this rental"

    # [[id]] tags become the project or rental name; the card is in the response data.
    # Follow-ups ("which one is cheapest") tag items from earlier answers; their names are in those notes.
    known = {key: name for turn in history for name, key in EARLIER_NAME.findall(str(turn.get("content") or ""))}
    known.update(labels)
    answer = _AGENT_TAG.sub(lambda m: known.get(m.group(1), ""), result.answer)
    answer = _collapse_doubled_names(answer, set(labels.values()))
    answer = _drop_repeated_name_lines(answer, set(labels.values()))
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()

    used = {tool["name"] for tool in result.tools if tool.get("ok")}
    # Phase 3: plain browse/search must not invent an Acot Recommend section.
    from app.answers.intent_route import is_recommend_intent

    wants_recommend = is_recommend_intent(question, conversation_memory) or bool(
        re.search(
            r"\b(?:invest|buy|recommend|best|compar(?:e|ison)|which should|suggest)\b",
            question or "",
            re.I,
        )
    )
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
    structured_summary["brokers"] = brokers
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


_LATEST_SALE_ASK = re.compile(r"\b(?:latest|most recent|last)\b[\s\S]{0,40}\b(?:sale|transaction)\b", re.I)
_COVERAGE_ASK = re.compile(r"\b(?:substantiat\w*|coverage|behind|lag|source|how far|up to date)\b", re.I)


def _area_clause(label):
    return ("(upper(area_name_prop_ai) = upper(%s) or upper(master_project_prop_ai) = upper(%s))", [label, label])


def _latest_sale_turn(question):
    """One dated recorded sale with its details, and how far the feed is behind today (ACOT QA AI-025 got a median)."""
    from datetime import date

    from app.answers.area_comparison import areas_in
    from app.database.postgres import CLEAN_SALE, fetch

    if not (_LATEST_SALE_ASK.search(question or "") and _COVERAGE_ASK.search(question or "")):
        return None
    named = areas_in(question)
    if not named:
        return None
    clause, params = _area_clause(named[0])
    rows = fetch(
        f"select instance_date, sale_price, bedrooms, procedure_area_sqft, project_name_prop_ai, property_type_en "
        f"from propai_sales where {CLEAN_SALE} and {clause} order by instance_date desc, sale_price desc limit 1",
        tuple(params),
    )
    feed = fetch(f"select max(instance_date) as d from propai_sales where {CLEAN_SALE}")
    if not rows:
        return None
    r, latest_any = rows[0], (feed[0]["d"] if feed else None)
    when = r["instance_date"]
    beds = r.get("bedrooms")
    beds_text = "studio" if beds == 0 else (f"{int(beds)}-bed" if beds is not None else "bedrooms not recorded")
    size = f", {int(r['procedure_area_sqft']):,} sq ft" if r.get("procedure_area_sqft") else ""
    today = date.today()
    lag = (today - date.fromisoformat(str(latest_any)[:10])).days if latest_any else None
    lines = [
        f"The latest recorded sale ACOT holds for **{named[0].title()}** is dated **{when}**: "
        f"**AED {int(r['sale_price']):,}**, {r.get('property_type_en') or 'property'}, {beds_text}{size}"
        + (f", in {r['project_name_prop_ai']}" if r.get("project_name_prop_ai") else "") + ".",
        "",
        "- **Source:** Dubai Land Department sale registrations, as loaded into ACOT (deleted and outlier-flagged records excluded).",
        f"- **Coverage:** the newest sale anywhere in the feed is dated {latest_any}"
        + (f", {lag} day{'s' if lag != 1 else ''} behind today ({today})." if lag is not None else "."),
        "- A registration date can trail the deal date; check the record on the DLD / Dubai REST transaction search before relying on it.",
    ]
    return _fast_answer(question, "\n".join(lines))


_SUPPLY_ASK = re.compile(r"\b(?:delivered|completed|completions?|handed over|handover)\b", re.I)
_PLANNED_ASK = re.compile(r"\b(?:announced|planned|upcoming|pipeline|expected|scheduled)\b", re.I)


def _supply_turn(question):
    """Projects completed in one year versus planned for another, from DLD project records (ACOT QA AI-023)."""
    from app.answers.area_comparison import areas_in
    from app.database.postgres import fetch

    text = question or ""
    years = [int(y) for y in re.findall(r"\b(20[1-4]\d)\b", text)]
    if not (_SUPPLY_ASK.search(text) and _PLANNED_ASK.search(text) and len(years) >= 2):
        return None
    named = areas_in(text)
    if not named:
        return None
    from app.answers.area_comparison import _PLAIN_NAMES

    place = named[0]
    # Project rows say "Dubai South" where sales say "Dubai South Residential District": match the buyer's words too.
    likes = [f"%{place}%"] + [f"%{k}%" for k in _PLAIN_NAMES if k in text.lower()]
    where = "(area ilike any(%s) or master_project ilike any(%s)) and not is_plot"
    rows = fetch(
        f"select extract(year from end_date)::int as y, "
        f"count(distinct upper(name)) filter (where development_status = 'FINISHED')::int as done, "
        f"count(distinct upper(name)) filter (where coalesce(development_status, '') not in ('FINISHED', 'CANCELLED'))::int as planned "
        f"from propai_projects where {where} and extract(year from end_date) = any(%s) group by 1",
        (likes, likes, years),
    )
    place = next((k.title() for k in _PLAIN_NAMES if k in text.lower()), place.title())
    by_year = {r["y"]: r for r in rows}
    lines = [f"From Dubai Land Department project records for **{place}** (counted in projects, not homes):", ""]
    for y in sorted(set(years)):
        r = by_year.get(y) or {"done": 0, "planned": 0}
        lines.append(f"- **{y}:** {r['done']} project{'s' if r['done'] != 1 else ''} recorded as completed, "
                     f"{r['planned']} still under construction or not started with that completion date")
    lines += ["", "Unit counts per project are not in ACOT's records, so the number of homes cannot be given. "
              "Planned dates are developer targets and often move. To verify a project, look it up on the DLD / "
              "Dubai REST project status service (completion percentage and escrow), and compare with the developer's notice."]
    return _fast_answer(question, "\n".join(lines))


_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_PERIOD_RE = re.compile(
    r"\b(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20\d\d)|q([1-4])\s*(20\d\d)|(?:in|during|for|of)\s+(20\d\d))\b",
    re.I)
_COHORT_METRIC = re.compile(r"\b(?:sales?|sold|median|price|count|rents?|lease|renewals?|new[\s-]contract|sq\.?\s*ft|psf|yield)\b", re.I)


def _periods(text):
    """Every month, quarter or year named in the question, as (label, start, end) with end exclusive."""
    from datetime import date

    out = []
    for m in _PERIOD_RE.finditer(text or ""):
        if m.group(1):
            mon, yr = _MONTH_NUM[m.group(1).lower()[:3]], int(m.group(2))
            start = date(yr, mon, 1)
            end = date(yr + (mon == 12), mon % 12 + 1, 1)
            label = start.strftime("%B %Y")
        elif m.group(3):
            q, yr = int(m.group(3)), int(m.group(4))
            start, end = date(yr, 3 * q - 2, 1), date(yr + (q == 4), (3 * q) % 12 + 1, 1)
            label = f"Q{q} {yr}"
        else:
            yr = int(m.group(5))
            start, end, label = date(yr, 1, 1), date(yr + 1, 1, 1), str(yr)
        if all(label != o[0] for o in out):
            out.append((label, start, end))
    return out


def _cohort_turn(question):
    """Dated cohorts from DLD records: the exact month / quarter / year, bedrooms, ready or off-plan, and the measure
    asked for (sale price, AED per sq ft, new vs renewal rents, matched gross yield). ACOT QA R05 / AI-016–AI-022
    got a trailing 12-month sale median whatever was asked."""
    from datetime import date

    from app.answers.area_comparison import areas_in
    from app.answers.intent_route import parse_bedrooms
    from app.database.postgres import CLEAN_SALE, fetch

    text = question or ""
    periods = _periods(text)
    named = areas_in(text)
    if not periods or not named or not _COHORT_METRIC.search(text) or re.search(r"\bhypothetical|sample\b", text, re.I):
        return None
    place = named[0]
    beds = parse_bedrooms(text)
    bed = beds[0] if beds else None
    clause, cparams = _area_clause(place)
    status = "Existing" if re.search(r"\bready\b", text, re.I) else "Off-Plan" if re.search(r"\boff[\s-]?plan\b", text, re.I) else None
    wants_rent = bool(re.search(r"\b(?:rents?|lease|renewals?|new[\s-]contract|ejari)\b", text, re.I))
    wants_psf = bool(re.search(r"\b(?:sq\.?\s*ft|psf|per square foot)\b", text, re.I))
    wants_yield = bool(re.search(r"\byield\b", text, re.I))
    apartments = bool(re.search(r"\bapartments?|flats?\b", text, re.I))
    today = date.today()
    head = ["Period", "Records", "Median", "Latest record"]
    rows, notes = [], []

    def sale_stats(start, end):
        where = [CLEAN_SALE, clause, "instance_date >= %s", "instance_date < %s"]
        params = [*cparams, start, end]
        if bed is not None:
            where.append("bedrooms = %s"); params.append(bed)
        if status:
            where.append("development_status = %s"); params.append(status)
        if apartments:
            where.append("property_sub_type_en = 'Flat'")
        col = "price_per_sqft" if wants_psf else "sale_price"
        r = fetch(f"select count(*)::int n, percentile_cont(0.5) within group (order by {col}) m, max(instance_date) last "
                  f"from propai_sales where {' and '.join(where)} and {col} > 0", tuple(params))
        return r[0] if r else {"n": 0}

    def rent_stats(start, end, kind=None):
        where = ["deleted_at is null", "coalesce(outlier_flag, '') <> '1'", clause.replace("area_name_prop_ai", "area_name_prop_ai"),
                 "contract_start_date >= %s", "contract_start_date < %s", "contract_months between 11 and 13", "annual_amount > 0"]
        params = [*cparams, start, end]
        if bed is not None:
            where.append("bedrooms = %s"); params.append(bed)
        if kind:
            where.append("contract_type = %s"); params.append(kind)
        r = fetch(f"select count(*)::int n, percentile_cont(0.5) within group (order by annual_amount) m, max(contract_start_date) last "
                  f"from propai_rents where {' and '.join(where)}", tuple(params))
        return r[0] if r else {"n": 0}

    aed = lambda v: f"AED {int(round(v)):,}" if v else "–"
    for label, start, end in periods:
        future = start > today
        partial = end > today
        if wants_rent and not wants_yield:
            for kind, name in (("New", "new contracts"), ("Renew", "renewals")):
                r = rent_stats(start, end, kind)
                rows.append([f"{label} · {name}", f"{r['n']:,}", aed(r.get("m")), str(r.get("last") or "–")])
            if start < date(2025, 9, 1):
                notes.append("Registered rent contracts in ACOT start on 1 Sep 2025, so earlier months are not covered.")
        else:
            r = sale_stats(start, end)
            median = (f"AED {int(round(r['m'])):,} per sq ft" if wants_psf and r.get("m") else aed(r.get("m")))
            rows.append([label + (" (so far)" if partial and not future else ""), f"{r['n']:,}", median, str(r.get("last") or "–")])
            if wants_yield:
                rr = rent_stats(start, end)
                y = (rr["m"] / r["m"] * 100) if r.get("m") and rr.get("m") and not wants_psf else None
                rows.append([f"{label} · rents (12-month contracts)", f"{rr['n']:,}", aed(rr.get("m")), str(rr.get("last") or "–")])
                rows.append([f"{label} · matched gross yield", "", f"{y:.2f}%" if y else "not enough matching records", ""])
                if start < date(2025, 9, 1):
                    notes.append("Rent records start on 1 Sep 2025, so the yield uses rents from that date only.")
        if future:
            notes.append(f"{label} has not happened yet, so there are no completed records.")
    mapped = fetch(f"select area_name_prop_ai a, count(*)::int n from propai_sales where {clause} and instance_date >= %s "
                   f"group by 1 order by 2 desc limit 3", (*cparams, min(p[1] for p in periods)))
    mapping = ", ".join(str(m["a"]).title() for m in mapped if m.get("a"))
    filt = [f"area: {place.title()}" + (f" (DLD area records: {mapping})" if mapping else ""), f"bedrooms: {bed}" if bed is not None else "all bedroom counts",
            {"Existing": "ready (existing) only", "Off-Plan": "off-plan only"}.get(status, "ready and off-plan"),
            "apartments only" if apartments else "all property types",
            "12-month contracts only (other lengths excluded)" if wants_rent or wants_yield else "deleted and outlier-flagged records excluded"]
    lines = [f"Dubai Land Department records for **{place.title()}**, exactly as filtered below:", "",
             "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |",
             *["| " + " | ".join(r) + " |" for r in rows], "",
             "**Filters:** " + "; ".join(filt) + ".",
             "**Measure:** " + ("median price per sq ft (DLD records area in square metres; converted at 1 m² = 10.764 sq ft; records without an area are excluded)" if wants_psf
                               else "median annual rent" if wants_rent and not wants_yield else "median sale price")
             + (". Gross yield = median rent ÷ median sale price for the same area, bedrooms and period (not the same units)." if wants_yield else "."),
             ]
    lines += [f"- {n}" for n in dict.fromkeys(notes)]
    return _fast_answer(question, "\n".join(lines))


_PULSE_ASK = re.compile(r"\bmarket\s+(?:pulse|overview)\b", re.I)


def _market_pulse_turn(question):
    """Explain and reconcile a Market Overview card count: same table, same filters (ACOT QA AI-017 read
    "market" as a project called The Market)."""
    from datetime import date, timedelta

    from app.answers.area_comparison import areas_in
    from app.database.postgres import fetch

    if not _PULSE_ASK.search(question or ""):
        return None
    named = areas_in(question)
    if not named:
        return None
    rows = fetch("select area, sales_12m, last_sale from propai_area_stats where upper(area) = upper(%s)", (named[0],))
    if not rows:
        return None
    row = rows[0]
    today = date.today()
    start = today - timedelta(days=365)
    shown = None
    for raw in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,6}\b", question):
        value = int(raw.replace(",", ""))
        if not 1900 <= value <= 2100:
            shown = value
            break
    lines = [
        f"The Market Overview card for **{row['area'].title()}** counts recorded Dubai Land Department sales:",
        "",
        f"- **Period:** {start:%-d %b %Y} to {today:%-d %b %Y} (a rolling 12 months ending today)",
        "- **Included:** every recorded sale in the area, off-plan and ready, all property types, with a sale price above zero",
        "- **Excluded:** deleted records, records flagged as outliers, and dates after today",
        f"- **Count today:** **{int(row['sales_12m']):,}** sales; latest recorded sale {row['last_sale']}",
    ]
    if shown is not None and shown != int(row["sales_12m"]):
        diff = int(row["sales_12m"]) - shown
        lines += ["", f"The {shown:,} you saw differs by {diff:+,}. The window moves every day: the oldest day drops "
                      "out and newly registered or corrected records come in, so the count changes from one day to the "
                      "next without any change in method."]
    elif shown is not None:
        lines += ["", f"That matches the {shown:,} on the card."]
    return _fast_answer(question, "\n".join(lines))


_ANALYTICAL_ASK = re.compile(
    r"\b(?:why|which|what|how|evidence|risk|reverse|undermine|net income|net yield|appreciation|liquidity|noise|"
    r"tenant demand|costs?|thesis|hold|stress|downside)\b", re.I)


async def _two_area_analysis_turn(question, conversation_memory, components):
    from types import SimpleNamespace

    from app.answers.advice_turn import advice_answer

    if len((question or "").split()) < 9 or not _ANALYTICAL_ASK.search(question or ""):
        return None  # "Compare Marina and JVC" keeps the plain comparison path with cards
    try:
        areas = await asyncio.to_thread(parse_area_comparison, question)
        table = await asyncio.to_thread(build_area_comparison, *areas) if areas else None
    except Exception as exc:
        print(f"two-area analysis unavailable: {exc}")
        return None
    if not table:
        return None
    figures = []
    for area in areas:
        try:
            m = await asyncio.to_thread(_market_numbers, area)
        except Exception:
            m = {}
        bits = [f"{k}={v}" for k, v in (m or {}).items() if v is not None and k in {
            "price_psf", "price_psf_change_pct", "sales_12m", "sales_change_pct", "gross_yield_pct", "growth_3y_pct",
            "median_price", "median_rent"}]
        if bits:
            figures.append(f"{area.title()} (last 12 months of DLD records): " + ", ".join(bits))
    turns = list(getattr(conversation_memory, "turns", None) or [])[-6:]
    turns.append({"user_question": "(ACOT recorded data for both areas)", "answer": "\n".join(figures) + "\n" + table})
    note = await advice_answer(
        question + "\nUse the ACOT figures above for both areas; say which figures are missing. Do not list projects.",
        SimpleNamespace(turns=turns), _llm_generate_from(conversation_memory, components))
    # The plain comparison closes with "ACOT has no yield…" and a follow-up offer; here the figures follow, so drop both.
    table = "\n".join(line for line in table.splitlines()
                      if not re.search(r"no yield or price-growth data|^would you like", line.strip(), re.I)).strip()
    answer = table + (f"\n\n{note}" if note else "")
    return _fast_answer(question, answer)


async def _prepare_acot_turn(question, components):
    """Prepare a turn; a broker's CRM/claim part carried from a mixed question rides along with the market answer."""
    components.pop("_carry_companion", None)
    components.pop("_resolved_action", None)  # set per turn by the understanding step
    prepared = await _prepare_acot_turn_inner(question, components)
    carry = components.pop("_carry_companion", None)
    if carry and not prepared.get("broker_companion"):
        prepared["broker_companion"] = carry
        prepared["companion_suffix"] = f"**For you:** {carry['answer']}" if carry.get("answer") else ""
    # A vacancy or viewing ask gets an honest line, whichever path answered it (ACOT QA R02).
    from app.answers.advice_turn import availability_note, document_note, scope_note

    doc_note = None if prepared.get("advice_kind") else document_note(question, prepared.get("documents"))
    if doc_note:
        from app.answers.advice_turn import documents_block
        from app.retrieval.propai_catalog import project_documents_for_question

        try:
            found = documents_block(project_documents_for_question(question))
        except Exception as exc:
            print(f"project documents unavailable: {exc}")
            found = None
        doc_note = found or doc_note  # the official files when on file; "not on file" only when there are none
    if doc_note and prepared.get("use_fast") and prepared.get("fast_answer"):
        # A brochure ask is answered first with "not on file", before any sale card (ACOT QA AI-013).
        prepared["fast_answer"] = f"{doc_note}\n\n{prepared['fast_answer']}"
        doc_note = None
    notes = [] if prepared.get("advice_kind") else [
        availability_note(question), doc_note,
        scope_note(question, prepared.get("fast_answer") if prepared.get("use_fast") else None),
    ]
    if any(notes):
        prepared["companion_suffix"] = "\n\n".join(p for p in (prepared.get("companion_suffix"), *notes) if p)
    return _ensure_prepared_turn(prepared)


async def _prepare_acot_turn_inner(
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
    question = repair_typed_question(question)
    question = _apply_pending_place_choice(question, conversation_memory)

    timing_ms = {}
    ask_started = time.perf_counter()

    # Greetings first — never answer "hi" with a sticky claim/CRM companion payload.
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
            "fast_answer": _small_talk_answer(
                small_talk,
                components.get("user_name"),
                components.get("audience") or "buyer",
            ),
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "small_talk": small_talk,
            "audience": components.get("audience") or "buyer",
        }

    community_developer = _community_developer_turn(question)
    if community_developer is not None:
        community_developer.update(
            timing_ms=timing_ms, ask_started=ask_started, rag_chain=rag_chain,
            conversation_memory=conversation_memory,
        )
        return community_developer

    # Another city / emirate is out of coverage: say so, never map it onto Dubai data.
    outside_reply = _outside_dubai_market_reply(question)
    if outside_reply:
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
            "fast_answer": outside_reply,
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "audience": components.get("audience") or "buyer",
        }

    # Nest agent Ask ACOT: CRM / claim / intro data in profile.companion (this question only).
    profile = components.get("profile") or {}
    companion = profile.get("companion") if isinstance(profile, dict) else None
    if (
        (components.get("audience") or "buyer") == "broker"
        and isinstance(companion, dict)
        and companion.get("kind")
    ):
        from app.answers.broker_companion import prepare_companion_turn

        prepared = await prepare_companion_turn(question, components, companion)
        prepared["timing_ms"] = timing_ms
        prepared["ask_started"] = ask_started
        broker_part = prepared.get("broker_companion") or {}
        if not broker_part.get("market") and not broker_part.get("alsoMarket"):
            return prepared
        components.setdefault("profile", {}).pop("companion", None)
        # Mixed question: run the market answer below, then add the broker's own part after it.
        if broker_part.get("alsoMarket") and not broker_part.get("market"):
            components["_carry_companion"] = broker_part

    # Calculations on the buyer's own numbers, laws, privacy/trust worries, "check your math" and "what have I
    # told you" are questions to answer, not searches: before every search path (ACOT QA R03/R04/R08-R10/R25).
    from app.answers.advice_turn import advice_answer, advice_kind

    kind = advice_kind(question)
    if kind:
        answer_text = await advice_answer(
            question, conversation_memory, _llm_generate_from(conversation_memory, components)
        )
        if answer_text:
            timing_ms["rewrite"] = 0
            advice = _fast_answer(question, answer_text)
            advice.update({
                "timing_ms": timing_ms, "ask_started": ask_started, "rag_chain": rag_chain,
                "conversation_memory": conversation_memory, "advice_kind": kind,
            })
            return advice

    # Two named areas plus an analytical ask ("Rank Arjan and JVC for net income… which costs could reverse it?"):
    # the area table from the database, then a short answer to the actual question over those figures (ACOT QA
    # AI-004, AI-005, AI-010 got a citywide ranking or only one of the two areas).
    from app.answers.advice_turn import advice_kind as _ak

    cohort = None if _ak(question) == "calc" else await asyncio.to_thread(_cohort_turn, question)
    if cohort is not None:
        cohort.update({"timing_ms": timing_ms, "ask_started": ask_started, "rag_chain": rag_chain,
                       "conversation_memory": conversation_memory})
        return cohort
    pulse = await asyncio.to_thread(_market_pulse_turn, question)
    if pulse is not None:
        pulse.update({"timing_ms": timing_ms, "ask_started": ask_started, "rag_chain": rag_chain,
                      "conversation_memory": conversation_memory})
        return pulse
    for record_turn in (_latest_sale_turn, _supply_turn):
        record = await asyncio.to_thread(record_turn, question)
        if record is not None:
            record.update({"timing_ms": timing_ms, "ask_started": ask_started, "rag_chain": rag_chain,
                           "conversation_memory": conversation_memory})
            return record
    two_areas = await _two_area_analysis_turn(question, conversation_memory, components)
    if two_areas is not None:
        two_areas.update({"timing_ms": timing_ms, "ask_started": ask_started, "rag_chain": rag_chain,
                          "conversation_memory": conversation_memory})
        return two_areas

    # A named project ("starting price of Serenz by Danube", "What is Sartoria") or amenities of the
    # cards on screen: answer that project from its own facts, never re-list the last browse. Before the
    # place-inheritance rewrite below, which turns "recent sale of Gallery Villas" into a Sports City browse.
    named_project = await _named_project_turn(question, conversation_memory, components)
    if named_project is not None:
        named_project["timing_ms"] = timing_ms
        named_project["ask_started"] = ask_started
        named_project["rag_chain"] = rag_chain
        named_project["conversation_memory"] = conversation_memory
        return named_project

    # After clarify: rewrite to a clean browse so SQL can answer (low tokens).
    prior = prior_anchor_question(conversation_memory, question)
    if not prior:
        prior = getattr(conversation_memory, "last_user_question", None)
    if prior and is_incomplete_home_browse(prior):
        beds = parse_bedrooms(prior)
        bed_label = f"{beds[0]} bedroom " if beds else ""
        home = "apartments" if re.search(
            r"\b(?:apartments?|flats?)\b", prior, re.I
        ) else "projects"
        if is_no_preference_reply(question):
            question = f"Show me {bed_label}{home}"
        else:
            place = extract_place(question)
            short = len((question or "").split()) <= 6
            if (
                place
                and place.lower() != "dubai"
                and short
                and not is_incomplete_home_browse(question)
            ):
                question = f"Show me {bed_label}{home} in {place}"

    # "show me two-bedroom apartments" after a JVC browse → keep that place on SQL.
    question = _inherit_place_for_home_browse(question, conversation_memory, prior)

    # Today / latest sale date: calendar + Prop-AI max(instance_date), no LLM.
    freshness = await asyncio.to_thread(_data_freshness_turn, question)
    if freshness is not None:
        freshness["timing_ms"] = timing_ms
        freshness["ask_started"] = ask_started
        freshness["rag_chain"] = rag_chain
        freshness["conversation_memory"] = conversation_memory
        return freshness

    # "Top 5 developers by sales value": a ranking, answered from the sales table. Before the interpreter step, which
    # read it as "the highest single sale". Only citywide; a named community goes the normal way.
    if _DEVELOPER_VALUE_RANK.search(question or ""):
        _place = extract_place(question)
        if not _place or str(_place).strip().lower() == "dubai":
            developer_early = await asyncio.to_thread(_developer_rank_turn, question)
            if developer_early is not None:
                developer_early["timing_ms"] = timing_ms
                developer_early["ask_started"] = ask_started
                developer_early["rag_chain"] = rag_chain
                developer_early["conversation_memory"] = conversation_memory
                return developer_early

    # "Show me that in a bar chart": draw what is on screen, before the interpreter step repeats the old answer.
    if _CHART_ASK.search(question or ""):
        chart_early = await asyncio.to_thread(_chart_turn, question, conversation_memory)
        if chart_early is not None:
            chart_early["timing_ms"] = timing_ms
            chart_early["ask_started"] = ask_started
            chart_early["rag_chain"] = rag_chain
            chart_early["conversation_memory"] = conversation_memory
            return chart_early

    # Projects ranked by rental yield: before the interpreter step, which sent it to "ACOT does not have rental yield".
    if _YIELD_RANK_ASK.search(question or ""):
        yield_early = await asyncio.to_thread(_yield_projects_turn, question, conversation_memory)
        if yield_early is not None:
            yield_early["timing_ms"] = timing_ms
            yield_early["ask_started"] = ask_started
            yield_early["rag_chain"] = rag_chain
            yield_early["conversation_memory"] = conversation_memory
            return yield_early

    # On-screen compare / best-of-shown before sticky followup/understood —
    # otherwise LLM resolve re-runs the prior afford with Business Bay + budget.
    compare_shown_early = await asyncio.to_thread(
        _compare_shown_turn, question, conversation_memory
    )
    if compare_shown_early is not None:
        compare_shown_early["timing_ms"] = timing_ms
        compare_shown_early["ask_started"] = ask_started
        compare_shown_early["rag_chain"] = rag_chain
        compare_shown_early["conversation_memory"] = conversation_memory
        return compare_shown_early
    shortlist_early = await asyncio.to_thread(
        _shortlist_recommend_turn, question, conversation_memory
    )
    if shortlist_early is not None:
        shortlist_early["timing_ms"] = timing_ms
        shortlist_early["ask_started"] = ask_started
        shortlist_early["rag_chain"] = rag_chain
        shortlist_early["conversation_memory"] = conversation_memory
        return shortlist_early

    # Production RAG pattern (NVIDIA AI-Q / OpenAI tool routing / Microsoft agent
    # routers): LLM understands the turn FIRST. Phrase regexes are only a cold
    # fallback when the LLM is unavailable or returns nothing usable.
    followup_turn = await _dialogue_followup_turn(
        question, conversation_memory, components
    )
    if followup_turn is not None:
        merged = dict(timing_ms)
        merged.update(followup_turn.get("timing_ms") or {})
        followup_turn["timing_ms"] = merged
        followup_turn["ask_started"] = ask_started
        followup_turn["rag_chain"] = rag_chain
        followup_turn["conversation_memory"] = conversation_memory
        return followup_turn

    understood = await _understood_turn(
        question, conversation_memory, components
    )
    if understood is not None:
        merged = dict(timing_ms)
        merged.update(understood.get("timing_ms") or {})
        understood["timing_ms"] = merged
        understood["ask_started"] = ask_started
        understood["rag_chain"] = rag_chain
        understood["conversation_memory"] = conversation_memory
        return understood

    # --- Offline / regex safety net only (LLM did not handle this turn) ---
    project_count = await asyncio.to_thread(
        _project_count_turn, question, conversation_memory
    )
    if project_count is not None:
        project_count["timing_ms"] = timing_ms
        project_count["ask_started"] = ask_started
        project_count["rag_chain"] = rag_chain
        project_count["conversation_memory"] = conversation_memory
        return project_count

    developer_rank = await asyncio.to_thread(_developer_rank_turn, question)
    if developer_rank is not None:
        developer_rank["timing_ms"] = timing_ms
        developer_rank["ask_started"] = ask_started
        developer_rank["rag_chain"] = rag_chain
        developer_rank["conversation_memory"] = conversation_memory
        return developer_rank

    if is_group_attribute_question(question):
        shown_attr = _shown_attribute_response(
            question, conversation_memory, timing_ms, ask_started, rag_chain
        )
        if shown_attr is not None:
            return shown_attr

    transaction_first = await asyncio.to_thread(
        _transaction_turn, question, conversation_memory
    )
    if transaction_first is not None:
        transaction_first["timing_ms"] = timing_ms
        transaction_first["ask_started"] = ask_started
        transaction_first["rag_chain"] = rag_chain
        transaction_first["conversation_memory"] = conversation_memory
        return transaction_first

    # Agents for on-screen projects must beat agent-first (which often ignores cards).
    broker_first = await asyncio.to_thread(
        _broker_turn, question, conversation_memory
    )
    if broker_first is not None:
        broker_first["timing_ms"] = timing_ms
        broker_first["ask_started"] = ask_started
        broker_first["rag_chain"] = rag_chain
        broker_first["conversation_memory"] = conversation_memory
        return broker_first

    # Clear Prop-AI place browses: SQL + template only (skip rewrite/plan/RAG).
    # Run before agent-first so min-price / ready filters stay on SQL.
    if not _agent_owns_turn(components) and _propai_sql_browse(question, conversation_memory):
        sql_browse = await asyncio.to_thread(
            _sql_project_browse_turn, question, conversation_memory
        )
        if sql_browse is not None:
            sql_browse["timing_ms"] = timing_ms
            sql_browse["ask_started"] = ask_started
            sql_browse["rag_chain"] = rag_chain
            sql_browse["conversation_memory"] = conversation_memory
            return sql_browse

    # Agent-first only when SQL browse should not handle this turn.
    if _agent_enabled(components) and (
        _agent_owns_turn(components) or not _propai_sql_browse(question, conversation_memory)
    ):
        agent_first = await _try_agent_turn(
            question, components, timing_ms, ask_started
        )
        if agent_first is not None:
            return agent_first

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

    community_list = await asyncio.to_thread(_community_list_turn, question)
    if community_list is not None:
        community_list["timing_ms"] = timing_ms
        community_list["ask_started"] = ask_started
        community_list["rag_chain"] = rag_chain
        community_list["conversation_memory"] = conversation_memory
        return community_list

    broker_turn = await asyncio.to_thread(
        _broker_turn, question, conversation_memory
    )
    if broker_turn is not None:
        broker_turn["timing_ms"] = timing_ms
        broker_turn["ask_started"] = ask_started
        broker_turn["rag_chain"] = rag_chain
        broker_turn["conversation_memory"] = conversation_memory
        return broker_turn

    wanted_place = await asyncio.to_thread(
        _ambiguous_place_question, question, conversation_memory
    )
    if wanted_place is False:
        coverage = _empty_structured_summary()
        return {
            "question": question,
            "standalone_question": question,
            "question_type": "general",
            "structured_summary": coverage,
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
            "fast_answer": (
                "I only cover Dubai real estate. "
                "Ask about an area such as Dubai Marina, Business Bay, or Palm Jumeirah."
            ),
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "small_talk": None,
        }
    if wanted_place:
        choice = _place_choice_turn(question, wanted_place, conversation_memory)
        choice["timing_ms"] = timing_ms
        choice["ask_started"] = ask_started
        choice["rag_chain"] = rag_chain
        choice["conversation_memory"] = conversation_memory
        return choice

    from app.answers.intent_route import classify_intent
    from app.answers.intent_turns import predict_turn, recommend_data_turn

    route = classify_intent(question, conversation_memory)
    if route.get("intent") == "predict":
        predicted = await asyncio.to_thread(
            predict_turn, question, conversation_memory, route
        )
        if predicted is not None:
            predicted["timing_ms"] = timing_ms
            predicted["ask_started"] = ask_started
            predicted["rag_chain"] = rag_chain
            predicted["conversation_memory"] = conversation_memory
            return predicted

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

    if route.get("intent") == "recommend":
        rec_turn = await asyncio.to_thread(
            recommend_data_turn, question, conversation_memory, route
        )
        if rec_turn is not None:
            rec_turn["timing_ms"] = timing_ms
            rec_turn["ask_started"] = ask_started
            rec_turn["rag_chain"] = rag_chain
            rec_turn["conversation_memory"] = conversation_memory
            return rec_turn

    # With the agent on, on-screen attribute follow-ups ("budget for each")
    # go through the agent so intent is understood from conversation context.
    prefer_agent_followup = (
        _agent_enabled(components)
        and bool(getattr(conversation_memory, "active_candidates", None) or [])
        and is_group_attribute_question(question)
    )
    if not prefer_agent_followup:
        shown_attr = _shown_attribute_response(
            question, conversation_memory, timing_ms, ask_started, rag_chain
        )
        if shown_attr is not None:
            return shown_attr

    agent_question = question
    # Sales paging is handled by dialogue_state + _transaction_turn. Do not
    # rewrite "show me more" into a fake project-name search string.
    if (
        route.get("kind") == "show_more"
        and _dialogue_action(conversation_memory) not in {"sales_count", "sales_list"}
    ):
        prior = prior_anchor_question(conversation_memory, question)
        if prior:
            agent_question = (
                f"{prior.rstrip('.?!')}. "
                "Show additional options for the same request, not the same list again."
            )
    if not prefer_agent_followup:
        project_detail = await asyncio.to_thread(
            _project_detail_turn, question, conversation_memory
        )
        if project_detail is not None:
            project_detail["timing_ms"] = timing_ms
            project_detail["ask_started"] = ask_started
            project_detail["rag_chain"] = rag_chain
            project_detail["conversation_memory"] = conversation_memory
            return project_detail
        place_detail = await asyncio.to_thread(
            _place_detail_turn, question, conversation_memory
        )
        if place_detail is not None:
            place_detail["timing_ms"] = timing_ms
            place_detail["ask_started"] = ask_started
            place_detail["rag_chain"] = rag_chain
            place_detail["conversation_memory"] = conversation_memory
            return place_detail

    # Already tried agent-first above when enabled; do not burn a second pass.
    if not _agent_enabled(components):
        agent_turn = await _try_agent_turn(
            question, components, timing_ms, ask_started
        )
        if agent_turn is not None:
            return agent_turn

    # Agent unavailable: fall back to structured on-screen attribute answers.
    if prefer_agent_followup:
        shown_attr = _shown_attribute_response(
            question, conversation_memory, timing_ms, ask_started, rag_chain
        )
        if shown_attr is not None:
            return shown_attr
        project_detail = await asyncio.to_thread(
            _project_detail_turn, question, conversation_memory
        )
        if project_detail is not None:
            project_detail["timing_ms"] = timing_ms
            project_detail["ask_started"] = ask_started
            project_detail["rag_chain"] = rag_chain
            project_detail["conversation_memory"] = conversation_memory
            return project_detail

    rewrite_started = time.perf_counter()
    question_for_search = prepare_user_question(agent_question) or agent_question
    if route.get("kind") == "show_more" and agent_question != question:
        standalone_question = agent_question
        timing_ms["rewrite"] = 0
    else:
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
        "budget_miss": bool(
            result.get("budget_miss", structured_data.get("budget_miss", False))
        ),
        "asked_max_price": result.get(
            "asked_max_price", structured_data.get("asked_max_price")
        ),
        "lowest_price": result.get(
            "lowest_price", structured_data.get("lowest_price")
        ),
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
            goal="invest",
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


def _ensure_prepared_turn(prepared):
    """Fill required finalize keys so a partial fast-turn never KeyErrors."""
    if not isinstance(prepared, dict):
        return prepared
    prepared.setdefault("documents", [])
    prepared.setdefault("investment_analysis", None)
    prepared.setdefault("structured_summary", _empty_structured_summary())
    prepared.setdefault(
        "retrieval_result",
        {
            "question_type": prepared.get("question_type") or "general",
            "structured_data": {
                "community": [],
                "projects": [],
                "sub_communities": [],
            },
            "documents": [],
        },
    )
    prepared.setdefault("timing_ms", {})
    prepared.setdefault("question", "")
    prepared.setdefault("standalone_question", prepared.get("question") or "")
    prepared.setdefault("question_type", "general")
    return prepared


_NO_ANSWER = ("I couldn't complete that answer right now because the AI service is busy. "
              "Please try again in a moment.")


def _finalize_acot_turn(prepared, answer, answer_started):
    prepared = _ensure_prepared_turn(prepared)
    timing_ms = dict(prepared.get("timing_ms") or {})
    timing_ms["answer"] = _elapsed_ms(answer_started)
    if prepared.get("ask_started") is not None:
        timing_ms["ask"] = _elapsed_ms(prepared["ask_started"])

    memory = prepared.get("conversation_memory")
    if memory is not None and hasattr(memory, "update"):
        memory.update(
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
        documents=prepared.get("documents") or [],
        investment_analysis=prepared.get("investment_analysis"),
    )
    response.setdefault("meta", {})["timing_ms"] = timing_ms
    try:
        from app.listings.tools import latest_sale_date
        response["meta"]["data_through"] = latest_sale_date()
    except Exception:
        pass
    if prepared.get("agent"):
        response["meta"]["agent"] = prepared["agent"]
    # Suggested next questions — same chips stream clients get via `followups`.
    followups = _followup_questions(prepared)
    if followups:
        response["followups"] = followups
    companion = prepared.get("broker_companion")
    if isinstance(companion, dict):
        response["companion"] = {
            "source": companion.get("source"),
            "leads": companion.get("leads") or [],
            "draft": companion.get("draft"),
            "suggestions": companion.get("suggestions") or followups or [],
            "claimCandidates": companion.get("claimCandidates") or [],
        }
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
    answer = answer if (answer or "").strip() else _NO_ANSWER
    if prepared.get("companion_suffix"):
        answer = f"{answer}\n\n{prepared['companion_suffix']}"

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

    # Larger chunks cut artificial stream delay on SQL/fast answers.
    if len(tokens) <= 40:
        group_size = 5
    elif len(tokens) <= 120:
        group_size = 10
    else:
        group_size = max(12, len(tokens) // 18)
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

_BROKER_FOLLOWUP = "Tell me the top agents for these projects"

_PROJECT_FOLLOWUPS = (
    ("brokers", _BROKER_FOLLOWUP),
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
    if _BROKER.search(question):
        topics.add("brokers")
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


def _broker_followup_chip(place=None, project_name=None):
    if project_name:
        return f"Tell me the top agents for {project_name}"
    if place:
        return f"Tell me the top agents for {place}"
    return _BROKER_FOLLOWUP


def _listing_followup_chips(place, listings):
    broker_chip = _broker_followup_chip(place=place)
    if len(listings) == 1:
        title = listings[0].get("title") or listings[0].get("name") or "this rental"
        return [
            broker_chip,
            f"What is the yearly rent of {title}?",
            f"How many bedrooms does {title} have?",
            f"What is the size of {title}?",
        ]

    where = f" in {place}" if place else ""
    return [
        broker_chip,
        f"Compare the first two rentals{where} and tell me which is best.",
        "Which of these has the lowest yearly rent?",
        "Which of these are 1-bedroom apartments?",
    ]


def _followup_questions(prepared):
    question = prepared.get("question") or ""

    companion = prepared.get("broker_companion") or {}
    if isinstance(companion, dict) and companion.get("suggestions"):
        return list(companion.get("suggestions") or [])[:3]

    if prepared.get("small_talk"):
        if (prepared.get("audience") or "buyer") == "broker":
            from app.answers.broker_companion import BROKER_STARTER_CHIPS

            return _unused_followups(prepared, BROKER_STARTER_CHIPS)
        return _unused_followups(prepared, _STARTER_CHIPS)

    # An answered calculation, law or privacy question: no agent or listing chips the buyer did not ask for.
    if prepared.get("advice_kind"):
        return []

    summary = prepared.get("structured_summary") or {}
    if summary.get("place_choice"):
        place = str(summary.get("understood_subject") or "").strip() or "Dubai"
        return [
            f"Show me properties in {place}",
            f"Show me agents in {place}",
        ]
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
    broker_done = summary.get("broker_followup") or "brokers" in _question_topics(question)
    if not projects and (summary.get("broker_followup") or summary.get("transaction_followup")):
        items = getattr(prepared.get("conversation_memory"), "active_candidates", None) or []
        projects = [
            item
            for item in items
            if isinstance(item, dict)
            and str(item.get("entity_type") or "project").lower() == "project"
            and item.get("name")
        ]
    place = _followup_place(prepared)
    ranked_areas = [
        row
        for row in (summary.get("community") or [])
        if isinstance(row, dict)
        and str(row.get("name") or "").strip()
        and (row.get("signals") or row.get("ranking") or row.get("focus") in {"ranking", "investment"})
    ]

    if listings or asks_for_listings(question):
        chips = _listing_followup_chips(place, listings)
        if broker_done:
            chips = [chip for chip in chips if not _BROKER.search(chip)]
            return _unused_followups(prepared, chips)
        return _unused_followups(prepared, chips, pinned=chips[0] if chips else None)

    if ranked_areas:
        first = str(ranked_areas[0].get("name") or "").strip()
        broker_chip = _broker_followup_chip(place=first)
        chips = [broker_chip]
        if len(projects) >= 2:
            chips.extend(chip for topic, chip in _PROJECT_FOLLOWUPS if topic != "brokers")
        elif len(projects) == 1:
            name = projects[0].get("name") or "this project"
            chips.extend([
                _TRANSACTION_FOLLOWUP,
                f"What is the starting price of {name}?",
                f"What is the handover date of {name}?",
            ])
        else:
            chips.extend([
                f"Show me properties in {first}",
                f"What are typical rents in {first}?",
            ])
        if broker_done:
            chips = [chip for chip in chips if chip != broker_chip]
            return _unused_followups(prepared, chips)
        return _unused_followups(prepared, chips, pinned=broker_chip)

    if len(projects) == 1:
        name = projects[0].get("name") or "this project"
        broker_chip = _broker_followup_chip(project_name=name)
        chips = [
            broker_chip,
            _TRANSACTION_FOLLOWUP,
            f"What is the starting price of {name}?",
            f"What is the handover date of {name}?",
            f"Who is the developer of {name}?",
            f"What amenities does {name} offer?",
            f"What is the bedroom range of {name}?",
        ]
        if broker_done:
            chips = [chip for chip in chips if chip != broker_chip]
            return _unused_followups(prepared, chips, pinned=_TRANSACTION_FOLLOWUP)
        return _unused_followups(prepared, chips, pinned=broker_chip)

    if len(projects) >= 2:
        communities = []
        for project in projects:
            community = str(project.get("community") or "").strip()
            if community and community not in communities:
                communities.append(community)
        broker_chip = _broker_followup_chip(
            place=communities[0] if len(communities) == 1 else place
        )
        chips = [broker_chip, *[chip for topic, chip in _PROJECT_FOLLOWUPS if topic != "brokers"]]
        if broker_done:
            chips = [chip for chip in chips if chip != broker_chip]
            return _unused_followups(prepared, chips, pinned=_TRANSACTION_FOLLOWUP)
        return _unused_followups(prepared, chips, pinned=broker_chip)

    if place and re.search(r"\bprojects?\b", question, flags=re.I):
        broker_chip = _broker_followup_chip(place=place)
        chips = [
            broker_chip,
            _TRANSACTION_FOLLOWUP,
            f"What are the starting prices of projects in {place}?",
            f"What are the handover dates of projects in {place}?",
            f"Compare the first and second projects in {place}.",
        ]
        if broker_done:
            chips = [chip for chip in chips if chip != broker_chip]
            return _unused_followups(prepared, chips, pinned=_TRANSACTION_FOLLOWUP)
        return _unused_followups(prepared, chips, pinned=broker_chip)

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
        # Voice UX stays on "Thinking…" until the first speech event. For fast
        # answers the spoken line is templated — push it before typewriter text.
        if speech and prepared.get("use_fast"):
            for event in spoken_so_far():
                yield event
        if prepared["use_fast"]:
            answer = prepared["fast_answer"] or _NO_ANSWER
            for piece in _iter_typewriter_pieces(answer):
                yield (
                    "delta",
                    {"type": "delta", "text": piece}
                )
                for event in spoken_so_far():
                    yield event
                await asyncio.sleep(0.001)
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
                # The model failed (rate limit, outage): say so instead of an empty bubble.
                answer = _NO_ANSWER
                yield ("delta", {"type": "delta", "text": answer})

        if prepared.get("companion_suffix"):
            tail = f"\n\n{prepared['companion_suffix']}"
            for piece in _iter_typewriter_pieces(tail):
                yield ("delta", {"type": "delta", "text": piece})
                await asyncio.sleep(0.001)
            answer = f"{answer}{tail}"

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
