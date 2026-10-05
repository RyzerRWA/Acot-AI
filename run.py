import os
import re
import io
import time
import json
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# SUPABASE STRUCTURED RETRIEVER
# =========================================================

from app.retrieval.supabase.structured_retriever import (
    SupabaseStructuredRetriever
)


# =========================================================
# SUPABASE PGVECTOR RETRIEVER
# =========================================================

from app.retrieval.supabase.pgvector_retriever import (
    SupabasePGVectorRetriever
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
    collapse_stretched_letters,
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

    structured_retriever = SupabaseStructuredRetriever()

    query_planner = QueryPlanner()

    entity_resolver = EntityResolver(
        structured_retriever
    )

    pgvector_retriever = SupabasePGVectorRetriever(
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

    return {
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
# PRINT STRUCTURED SUPABASE DATA
# =========================================================

def print_structured_data(structured_data):

    print("\n================================")
    print("SUPABASE POSTGRES DATA")
    print("================================")


    if not structured_data:

        print("No structured Supabase data.")

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
    print("SUPABASE PGVECTOR DATA")
    print("================================")


    if not documents:

        print("No Supabase PGVector document results.")

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

def _json_safe(value):
    """
    Convert backend/database values into JSON-safe Python values.

    This keeps the existing retrieval data untouched while making the
    final response safe for a REST API or frontend client.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    # Handles Decimal, datetime, UUID, and other database values.
    return str(value)


def _format_bedrooms(record):
    """
    Return the bedroom range exactly from the available project data.
    """

    minimum = record.get("bedroom_min")
    maximum = record.get("bedroom_max")

    if minimum is None and maximum is None:
        return None

    if minimum is None:
        return str(maximum)

    if maximum is None:
        return str(minimum)

    if minimum == maximum:
        return str(minimum)

    return f"{minimum}-{maximum}"


def _bedroom_bounds(project):
    minimum = project.get("bedroom_min")
    maximum = project.get("bedroom_max")
    bedrooms = project.get("bedrooms")

    if isinstance(bedrooms, dict):
        if minimum is None:
            minimum = bedrooms.get("min")
        if maximum is None:
            maximum = bedrooms.get("max")

    return minimum, maximum


def _format_bedroom_phrase(project):
    minimum, maximum = _bedroom_bounds(project)

    if minimum is None and maximum is None:
        return None

    if minimum == 0 and maximum not in (None, 0):
        noun = "bedroom" if maximum == 1 else "bedrooms"
        return f"Studios to {maximum} {noun}"

    if minimum == 0:
        return "Studios"

    if minimum is None:
        return f"{maximum} bedrooms"

    if maximum is None or minimum == maximum:
        noun = "bedroom" if minimum == 1 else "bedrooms"
        return f"{minimum} {noun}"

    return f"{minimum} to {maximum} bedrooms"


def _format_aed(value):
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.replace("AED", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            value = float(cleaned)
        except ValueError:
            return value

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number.is_integer():
        return f"AED {int(number):,}"

    return f"AED {number:,.0f}"


def _format_handover_month(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    return parsed.strftime("%B %Y")


def _format_project_card(record):
    """
    Convert a project database row into a frontend-friendly project card.

    Only fields that exist in the current database are exposed.
    No rental yield, fair value, deal score, ROI, or other missing
    investment metrics are invented here.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "developer": record.get("developer_name"),
        "developer_id": _json_safe(record.get("developer_id")),
        "city": record.get("city"),
        "community": record.get("community"),
        "sub_community": record.get("sub_community"),
        "price": _json_safe(record.get("price")),
        "price_label": (
            f"AED {record.get('price')}"
            if record.get("price") is not None
            else None
        ),
        "bedrooms": {
            "min": _json_safe(record.get("bedroom_min")),
            "max": _json_safe(record.get("bedroom_max")),
            "label": _format_bedrooms(record),
        },
        "size_sqft": {
            "min": _json_safe(record.get("size_min")),
            "max": _json_safe(record.get("size_max")),
        },
        "property_types": _json_safe(
            record.get("property_types") or []
        ),
        "status": record.get("status"),
        "project_status": record.get("project_status"),
        "pool_type": record.get("pool_type"),
        "handover_time": _json_safe(
            record.get("handover_time")
        ),
        "amenities": _json_safe(
            record.get("amenities") or []
        ),
        "photos": _json_safe(
            record.get("photos") or []
        ),
        "description": record.get("description"),
        "brochure_url": record.get("brochure_url"),
        "property_finder_url": record.get(
            "property_finder_url"
        ),
        "source": record.get("source"),
    }


def _format_community_card(record):
    """
    Convert a community database row into a frontend-friendly card.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "slug": record.get("slug"),
        "city": record.get("city"),
        "location": {
            "latitude": _json_safe(record.get("latitude")),
            "longitude": _json_safe(record.get("longitude")),
        },
        "inventory": {
            "sell_properties_count": _json_safe(
                record.get("sell_properties_count")
            ),
            "rent_properties_count": _json_safe(
                record.get("rent_properties_count")
            ),
            "projects_count": _json_safe(
                record.get("projects_count")
            ),
            "pool_projects_count": _json_safe(
                record.get("pool_projects_count")
            ),
            "total_count": _json_safe(
                record.get("total_count")
            ),
        },
        "assigned_agents": _json_safe(
            record.get("assigned_agents")
        ),
        "knowledge_text": record.get("knowledge_text"),
        "source": record.get("source"),
    }


def _format_sub_community_card(record):
    """
    Convert a sub-community database row into a frontend-friendly card.
    """

    return {
        "id": _json_safe(record.get("id")),
        "name": record.get("name"),
        "slug": record.get("slug"),
        "city": record.get("city"),
        "community": record.get("community"),
        "community_slug": record.get("community_slug"),
        "location": {
            "latitude": _json_safe(record.get("latitude")),
            "longitude": _json_safe(record.get("longitude")),
        },
        "photos": _json_safe(
            record.get("photos") or []
        ),
        "knowledge_text": record.get("knowledge_text"),
        "source": record.get("source"),
    }


def _format_document_card(record):
    """
    Convert a pgvector document result into a frontend-friendly source.
    """

    return {
        "id": _json_safe(record.get("id")),
        "document_id": _json_safe(
            record.get("document_id")
        ),
        "title": record.get("document_title"),
        "type": record.get("document_type"),
        "publisher": record.get("publisher"),
        "chunk_id": record.get("chunk_id"),
        "similarity": _json_safe(
            record.get("similarity")
        ),
        "url": record.get("document_url"),
        "content": record.get("content"),
    }


def _detect_response_type(
    question,
    question_type,
    structured_summary
):
    """
    Determine the frontend response component from the actual result.

    This is presentation metadata only. It does not affect retrieval.
    """

    question_lower = (question or "").lower()

    projects = structured_summary.get(
        "projects",
        []
    )
    communities = structured_summary.get(
        "community",
        []
    )
    sub_communities = structured_summary.get(
        "sub_communities",
        []
    )

    if question_type == "investment":
        return "investment_analysis"

    if any(
        phrase in question_lower
        for phrase in (
            "compare",
            "comparison",
            "difference between",
            "vs ",
            " versus "
        )
    ) and len(projects) >= 2:
        return "project_comparison"

    if question_type == "search" and projects:
        return "project_list"

    if sub_communities:
        return "sub_community_list"

    if communities:
        return "community_list"

    return "general"


def build_frontend_response(
    question,
    standalone_question,
    answer,
    question_type,
    structured_summary,
    documents,
    investment_analysis=None,
):
    """
    Build the single response object that a future frontend/API can consume.

    The AI answer remains available as `answer`.
    Structured database records are exposed as frontend-friendly `data`.
    """

    projects = structured_summary.get(
        "projects",
        []
    ) or []

    communities = structured_summary.get(
        "community",
        []
    ) or []

    sub_communities = structured_summary.get(
        "sub_communities",
        []
    ) or []

    formatted_projects = [
        _format_project_card(project)
        for project in projects
        if isinstance(project, dict)
    ]

    formatted_communities = [
        _format_community_card(community)
        for community in communities
        if isinstance(community, dict)
    ]

    formatted_sub_communities = [
        _format_sub_community_card(sub_community)
        for sub_community in sub_communities
        if isinstance(sub_community, dict)
    ]

    formatted_documents = [
        _format_document_card(document)
        for document in documents
        if isinstance(document, dict)
    ]

    response_type = _detect_response_type(
        question,
        question_type,
        structured_summary,
    )

    return {
        "status": "success",
        "response_type": response_type,
        "question": question,
        "standalone_question": standalone_question,
        "question_type": question_type,
        "answer": answer,
        "data": {
            "communities": formatted_communities,
            "projects": formatted_projects,
            "sub_communities": formatted_sub_communities,
            "documents": formatted_documents,
            "investment_analysis": _json_safe(
                investment_analysis
            ),
        },
        "meta": {
            "counts": {
                "communities": len(formatted_communities),
                "projects": len(formatted_projects),
                "sub_communities": len(
                    formatted_sub_communities
                ),
                "documents": len(formatted_documents),
            }
        },
    }


# =========================================================
# FAST STRUCTURED ANSWER
# =========================================================

def _is_fast_structured_question(
    question,
    question_type,
    structured_summary,
    documents,
):
    """
    Use the database template only for a browse/search that already has rows.

    Price, bedroom, handover, comparison, and other follow-ups stay on the
    model path so the reply is written instead of reprinted as a hit list.
    """

    del question

    if question_type != "search":
        return False

    if documents:
        return False

    if structured_summary.get("filter_no_match"):
        return True

    return bool(
        structured_summary.get("projects")
        or structured_summary.get("community")
        or structured_summary.get("sub_communities")
    )


def _build_fast_structured_answer(
    question,
    structured_summary,
):
    """
    Build a concise, grounded answer directly from PostgreSQL records.
    No model call is made for this path.
    """

    q = (question or "").lower()
    projects = structured_summary.get("projects", []) or []
    communities = structured_summary.get("community", []) or []
    sub_communities = structured_summary.get("sub_communities", []) or []

    # ---------------------------------------------------------
    # CONVERSATIONAL FILTER: ZERO MATCHES
    # ---------------------------------------------------------
    # Example:
    #   Q1: Show me projects in Dubai Marina
    #   Q2: Which of these have 2 bedroom options?
    #
    # `projects` is intentionally empty because no candidate matched the
    # filter. `filter_evidence_projects` contains the previous candidates
    # so we can explain the result using the real Supabase data.
    if structured_summary.get("filter_no_match"):
        evidence_projects = (
            structured_summary.get(
                "filter_evidence_projects",
                []
            )
            or []
        )

        bedroom_match = re.search(
            r"\b(\d+)\s*(?:-?\s*)bed(?:room)?s?\b",
            q,
        )

        if bedroom_match and evidence_projects:
            requested_bedrooms = int(
                bedroom_match.group(1)
            )

            lines = [
                f"No retrieved projects have "
                f"{requested_bedrooms}-bedroom options."
            ]

            for project in evidence_projects:
                name = project.get("name") or "Unknown project"
                bedroom_label = _format_bedrooms(project)

                if bedroom_label is not None:
                    lines.append(
                        f"{name} offers {bedroom_label} bedrooms."
                    )
                else:
                    lines.append(
                        f"{name} does not have bedroom information "
                        f"available in the current data."
                    )

            return "\n".join(lines)

        # Generic fallback for other conversational filters.
        if evidence_projects:
            names = [
                project.get("name") or "Unknown project"
                for project in evidence_projects
            ]
            return (
                "None of the previously retrieved projects matched "
                "the requested filter. Previous candidates: "
                + ", ".join(names)
                + "."
            )

    if projects:
        lines = []

        wants_bedrooms = any(
            term in q
            for term in ("bedroom", "bedrooms")
        )
        wants_handover = "handover" in q
        wants_developer = any(
            term in q
            for term in ("developer", "developers")
        )
        wants_status = "status" in q
        wants_price = any(
            term in q
            for term in ("price", "prices", "cost")
        )

        # Field-specific follow-up answers.
        if len(projects) == 1:
            project = projects[0]
            name = project.get("name") or "Unknown project"

            if wants_bedrooms:
                bedroom_label = _format_bedrooms(project)
                value = (
                    bedroom_label
                    if bedroom_label is not None
                    else "Not available"
                )
                return f"{name} offers {value} bedrooms."

            if wants_handover:
                value = project.get("handover_time")
                if value is None:
                    return f"Handover date for {name} is not available in the current data."
                return f"{name} handover date: {value}"

            if wants_developer:
                value = project.get("developer_name")
                value = (
                    str(value).strip()
                    if value is not None and str(value).strip()
                    else "Not available"
                )
                return f"{name} developer: {value}"

            if wants_status:
                status = project.get("status")
                project_status = project.get("project_status")

                if status and project_status:
                    return (
                        f"{name} status: {status}; "
                        f"project status: {project_status}"
                    )

                value = status or project_status or "Not available"
                return f"{name} status: {value}"

            if wants_price:
                value = project.get("price")
                value = (
                    f"AED {value}"
                    if value is not None
                    else "Not available"
                )
                return f"{name} starting price: {value}"

        # Multi-project field-specific answers.
        if wants_bedrooms:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                bedroom_label = _format_bedrooms(project)
                bedroom_text = (
                    bedroom_label
                    if bedroom_label is not None
                    else "Not available"
                )
                lines.append(
                    f"{index}. {name} | Bedrooms: {bedroom_text}"
                )
            return "\n".join(lines)

        if wants_handover:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                value = project.get("handover_time")
                value = value if value is not None else "Not available"
                lines.append(
                    f"{index}. {name} | Handover: {value}"
                )
            return "\n".join(lines)

        if wants_developer:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                developer = project.get("developer_name")
                developer_text = (
                    str(developer).strip()
                    if developer is not None and str(developer).strip()
                    else "Not available"
                )
                lines.append(
                    f"{index}. {name} | Developer: {developer_text}"
                )
            return "\n".join(lines)

        if wants_price:
            lines = [
                f"Found {len(projects)} project(s) matching your request:"
            ]
            for index, project in enumerate(projects, start=1):
                name = project.get("name") or "Unknown project"
                price = project.get("price")
                price_text = (
                    f"AED {price}" if price is not None else "Not available"
                )
                lines.append(
                    f"{index}. {name} | Starting price: {price_text}"
                )
            return "\n".join(lines)

        # Default browse answer.
        lines = [
            _browse_intro(
                projects,
                ambiguous_label=structured_summary.get("ambiguous_label"),
            ),
            "",
        ]

        for project in projects:
            name = project.get("name") or "Unknown project"
            developer = project.get("developer_name") or project.get("developer")
            developer_text = (
                str(developer).strip()
                if developer is not None and str(developer).strip()
                else ""
            )
            sentence = f"- **{name}**"
            sentence += f" by {developer_text}." if developer_text else "."

            details = []
            price_text = _format_aed(project.get("price"))
            if price_text:
                details.append(f"Starting price {price_text}.")

            bedroom_text = _format_bedroom_phrase(project)
            if bedroom_text:
                details.append(f"{bedroom_text}.")

            handover_text = _format_handover_month(project.get("handover_time"))
            if handover_text:
                details.append(f"Handover {handover_text}.")

            if details:
                sentence = f"{sentence} {' '.join(details)}"

            lines.append(sentence)

        return "\n".join(lines)

    if sub_communities:
        lines = [
            f"Found {len(sub_communities)} sub-community(s) matching your request:"
        ]
        for index, record in enumerate(sub_communities, start=1):
            lines.append(
                f"{index}. {record.get('name') or 'Unknown sub-community'}"
            )
        return "\n".join(lines)

    if communities:
        lines = [
            f"Found {len(communities)} community(s) matching your request:"
        ]
        for index, record in enumerate(communities, start=1):
            name = record.get("name") or "Unknown community"
            city = record.get("city")
            if city:
                lines.append(f"{index}. {name} | City: {city}")
            else:
                lines.append(f"{index}. {name}")
        return "\n".join(lines)

    return _missing_data_answer(structured_summary)


# =========================================================
# SMALL TALK
# =========================================================

_SMALL_TALK_KINDS = {
    "hi": "greeting",
    "hii": "greeting",
    "hello": "greeting",
    "hey": "greeting",
    "good morning": "greeting",
    "good evening": "greeting",
    "good afternoon": "greeting",
    "thanks": "thanks",
    "thank you": "thanks",
    "bye": "bye",
    "goodbye": "bye",
    "who are you": "identity",
    "what are you": "identity",
    "what can you do": "capabilities",
    "what do you do": "capabilities",
    "how can you help": "capabilities",
    "help": "capabilities",
}

_STARTER_CHIPS = (
    "Show me projects in Jumeirah Village Circle.",
    "What can you do?",
    "Who are you?",
)


def _normalize_chat_text(question):
    text = collapse_stretched_letters(question or "").strip().lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[!?.]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _unique_field(projects, key):
    values = []
    for project in projects:
        raw = project.get(key)
        if raw is None and key == "developer_name":
            raw = project.get("developer")
        text = str(raw).strip() if raw is not None else ""
        if text and text not in values:
            values.append(text)
    return values


def _price_range_clause(projects):
    prices = []
    for project in projects:
        value = project.get("price")
        if value is None:
            continue
        try:
            if isinstance(value, str):
                value = value.replace("AED", "").replace(",", "").strip()
            number = float(value)
        except (TypeError, ValueError):
            continue
        prices.append(number)

    if not prices:
        return ""

    low = min(prices)
    high = max(prices)
    if low == high:
        return f", with a starting price of {_format_aed(low)}"
    return (
        f", with starting prices from {_format_aed(low)} "
        f"to {_format_aed(high)}"
    )


def _browse_intro(projects, ambiguous_label=None):
    """Open a project list in one sentence."""

    count = len(projects)
    noun = "project" if count == 1 else "projects"
    all_active = all(
        str(project.get("status") or "").strip().upper() == "ACTIVE"
        for project in projects
    )
    active = "active " if all_active else ""

    if ambiguous_label and count > 1:
        return f"{ambiguous_label} matches {count} {noun}. Here they are:"

    communities = _unique_field(projects, "community")
    developers = _unique_field(projects, "developer_name")
    price_clause = _price_range_clause(projects)

    if count == 1:
        name = projects[0].get("name") or "This project"
        place = f" in {communities[0]}" if communities else ""
        kind = f"{active}project".strip()
        article = "an" if kind.startswith("active") else "a"
        return (
            f"{name} is {article} {kind}{place}. "
            "Here is what is on record:"
        )

    if len(communities) == 1:
        return (
            f"I found {count} {active}{noun} in {communities[0]}"
            f"{price_clause}. Here is a short look at each one:"
        )

    if len(developers) == 1:
        return (
            f"I found {count} {active}{developers[0]} {noun}"
            f"{price_clause}. Here is a short look at each one:"
        )

    return (
        f"I found {count} {active}{noun}{price_clause}. "
        "Here is a short look at each one:"
    )


def _missing_data_answer(structured_summary):
    subject = str(
        structured_summary.get("understood_subject") or ""
    ).strip()
    closest = str(structured_summary.get("closest_match") or "").strip()
    kind = structured_summary.get("understood_kind") or "place"

    if not subject:
        return "I don't have a match for that in the current data."

    if kind == "project":
        return (
            f"I don't have a project called {subject} "
            "in the current data."
        )

    if kind == "developer":
        return (
            f"I don't have {subject} projects in the current data."
        )

    if kind == "developer_community":
        return f"I don't have {subject} in the current data."

    if kind == "pattern":
        return (
            f"I don't have projects matching {subject} "
            "in the current data."
        )

    if closest:
        return (
            f"I don't have projects in {subject}. "
            f"The closest community I do have is {closest}."
        )

    return (
        f"I don't have projects in {subject} in the current data."
    )


def _classify_small_talk(question):
    """Return a small-talk kind when the whole message is a greeting or identity question."""

    return _SMALL_TALK_KINDS.get(_normalize_chat_text(question))


def _small_talk_answer(kind):
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

    if kind == "thanks":
        return (
            "You're welcome. Ask me about a Dubai community or project "
            "whenever you're ready."
        )

    if kind == "bye":
        return (
            "Goodbye. I'm here when you want to look at Dubai projects again."
        )

    return (
        "Hi, I'm ACOT, a Dubai real estate assistant. I can look up projects, "
        "starting prices, bedroom ranges, handover dates, and comparisons "
        "from the data I have. What would you like to know?"
    )


def _empty_structured_summary():
    return {
        "community": [],
        "projects": [],
        "sub_communities": [],
        "filter_no_match": False,
        "filter_evidence_projects": [],
    }


# =========================================================
# PROCESS QUESTION
# =========================================================

def _prepare_acot_turn(
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
            "fast_answer": _small_talk_answer(small_talk),
            "context": None,
            "timing_ms": timing_ms,
            "ask_started": ask_started,
            "small_talk": small_talk,
        }

    rewrite_started = time.perf_counter()
    question_for_search = prepare_user_question(question) or question
    standalone_question = conversation_memory.rewrite_question(
        question_for_search
    )
    timing_ms["rewrite"] = _elapsed_ms(rewrite_started)

    result = hybrid_retriever.retrieve(
        question=standalone_question,
        conversation_context=conversation_memory.context(),
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

    use_fast = _is_fast_structured_question(
        question=standalone_question,
        question_type=question_type,
        structured_summary=structured_summary,
        documents=documents,
    )
    if (
        not documents
        and not structured_summary.get("projects")
        and structured_summary.get("understood_subject")
        and not structured_summary.get("filter_no_match")
    ):
        use_fast = True

    if use_fast:
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
    return response


def ask_acot(
    question,
    components
):
    prepared = _prepare_acot_turn(
        question,
        components
    )

    answer_started = time.perf_counter()
    if prepared["use_fast"]:
        answer = prepared["fast_answer"]
    else:
        answer = prepared["rag_chain"].generate_answer(
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


_PROJECT_FOLLOWUPS = (
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


def _unused_followups(prepared, chips):
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
    return unused[:3]


def _followup_questions(prepared):
    question = prepared.get("question") or ""

    if prepared.get("small_talk"):
        return _unused_followups(prepared, _STARTER_CHIPS)

    projects = [
        project
        for project in (prepared["structured_summary"].get("projects") or [])
        if isinstance(project, dict)
    ]
    if len(projects) >= 2:
        return _unused_followups(
            prepared,
            [chip for _, chip in _PROJECT_FOLLOWUPS],
        )
    if len(projects) == 1:
        name = projects[0].get("name") or "this project"
        return _unused_followups(
            prepared,
            [
                f"What is the starting price of {name}?",
                f"What is the handover date of {name}?",
                f"Who is the developer of {name}?",
                f"What amenities does {name} offer?",
                f"What is the bedroom range of {name}?",
            ],
        )
    return _unused_followups(prepared, _STARTER_CHIPS)


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
    slim["data"] = data

    summary = slim.get("ai_summary")
    if isinstance(summary, dict) and isinstance(summary.get("data"), dict):
        summary = dict(summary)
        summary_data = dict(summary["data"])
        summary_data["projects"] = projects
        summary["data"] = summary_data
        slim["ai_summary"] = summary
    return slim


def stream_acot(
    question,
    components
):
    """Yield typed stream events: meta, delta, followups, then done.

    Database answers are split into small pieces so the client can show
    them like a typewriter. Model answers keep the provider's own tokens.
    Memory is updated only after the full answer is assembled.
    """

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        prepared = _prepare_acot_turn(
            question,
            components
        )

    yield (
        "meta",
        {
            "type": "meta",
            "sources": _stream_sources(prepared),
        }
    )

    answer_started = time.perf_counter()
    if prepared["use_fast"]:
        answer = prepared["fast_answer"] or ""
        for piece in _iter_typewriter_pieces(answer):
            yield (
                "delta",
                {"type": "delta", "text": piece}
            )
            time.sleep(0.02)
    else:
        parts = []

        for delta in prepared["rag_chain"].stream_answer(
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

    yield (
        "done",
        _finalize_acot_turn(
            prepared,
            answer,
            answer_started,
        )
    )


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
                response = ask_acot(
                    question,
                    components
                )

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