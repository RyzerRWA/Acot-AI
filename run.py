import os
import re
import io
import time
import json
from contextlib import redirect_stdout, redirect_stderr
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

    if projects:
        if any(
            phrase in question_lower
            for phrase in (
                "project",
                "projects",
                "property",
                "properties",
                "price",
                "prices",
                "bedroom",
                "handover",
                "amenit",
                "developer"
            )
        ):
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
    Identify simple database-only questions that do not need Gemini for
    the final answer. Retrieval has already verified the records.
    """

    if documents:
        return False

    if question_type in {
        "investment",
        "comparison",
        "ranking",
        "calculation",
        "prediction",
        "document",
    }:
        return False

    q = (question or "").lower().strip()

    # Complex/knowledge-heavy wording stays on the RAG path.
    blocked_terms = (
        "invest",
        "investment",
        "roi",
        "yield",
        "forecast",
        "predict",
        "compare",
        "comparison",
        "versus",
        " vs ",
        "best",
        "top ",
        "recommend",
        "brochure",
        "document",
        "building code",
        "amenities",
        "amenity",
        "security",
        "features",
        "interior",
        "interiors",
        "finishes",
        "floor plan",
        "floor plans",
        "layout",
        "layouts",
        "view",
        "views",
    )

    if any(term in q for term in blocked_terms):
        return False

    structured_terms = (
        "project",
        "projects",
        "property",
        "properties",
        "price",
        "prices",
        "cost",
        "bedroom",
        "bedrooms",
        "developer",
        "developers",
        "handover",
        "status",
        "size",
        "sqft",
        "square foot",
        "available",
    )

    if not any(term in q for term in structured_terms):
        return False

    # A conversational filter may produce zero matching projects while
    # still retaining the previous projects as filter evidence. Keep this
    # on the deterministic fast path so ACOT can explain the no-match
    # result without another LLM call.
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
    No Gemini call is made for this path.
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

        # Default project-list answer.
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

            price = project.get("price")
            price_text = (
                f"AED {price}" if price is not None else "Not available"
            )

            bedroom_label = _format_bedrooms(project)
            bedroom_text = (
                bedroom_label
                if bedroom_label is not None
                else "Not available"
            )

            lines.append(
                f"{index}. {name} | Developer: {developer_text} | "
                f"Starting price: {price_text} | Bedrooms: {bedroom_text}"
            )

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

    return "No matching structured records were found."


# =========================================================
# PROCESS QUESTION
# =========================================================

def ask_acot(
    question,
    components
):

    hybrid_retriever = components["hybrid_retriever"]
    context_builder = components["context_builder"]
    investment_analyzer = components["investment_analyzer"]
    rag_chain = components["rag_chain"]
    conversation_memory = components["conversation_memory"]

    # -----------------------------------------------------
    # CONVERSATIONAL MEMORY
    # -----------------------------------------------------

    standalone_question = conversation_memory.rewrite_question(
        question
    )

    result = hybrid_retriever.retrieve(
        question=standalone_question,
        conversation_context=conversation_memory.context(),
    )

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
    }

    investment_analysis = None

    if question_type == "investment":
        try:
            investment_analysis = (
                investment_analyzer.analyze(
                    structured_summary
                )
            )
        except Exception:
            investment_analysis = None

    # -----------------------------------------------------
    # FAST PATH: structured database answer
    # -----------------------------------------------------
    # Simple project/community/filter questions do not need the final
    # Gemini generation step. This reduces latency and token usage while
    # keeping the answer fully grounded in retrieved Supabase data.
    if _is_fast_structured_question(
        question=standalone_question,
        question_type=question_type,
        structured_summary=structured_summary,
        documents=documents,
    ):
        answer = _build_fast_structured_answer(
            question=standalone_question,
            structured_summary=structured_summary,
        )
    else:
        context = context_builder.build_context(
            structured_summary=structured_summary,
            investment_analysis=investment_analysis,
            document_results=documents,
            community_results=structured_data.get(
                "community",
                []
            )
        )

        answer = rag_chain.generate_answer(
            question=standalone_question,
            context=context,
            question_type=question_type
        )

    conversation_memory.update(
        user_question=question,
        standalone_question=standalone_question,
        answer=answer,
        retrieval_result=result,
    )

    return build_frontend_response(
        question=question,
        standalone_question=standalone_question,
        answer=answer,
        question_type=question_type,
        structured_summary=structured_summary,
        documents=documents,
        investment_analysis=investment_analysis,
    )


# =========================================================
# SEQUENTIAL ANSWER OUTPUT
# =========================================================

def print_answer_sequentially(answer, delay=0.004):
    """
    Display the final ACOT answer progressively instead of printing
    the entire response at once.

    The answer itself is still generated by the same RAG/Gemini pipeline.
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