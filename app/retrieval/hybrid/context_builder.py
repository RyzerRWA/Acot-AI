"""
ACOT Hybrid Context Builder

Builds the final grounded context from:

1. Supabase PostgreSQL structured data
2. Supabase pgvector document chunks

Expected input from HybridRetriever:

{
    "route": "...",
    "question_type": "...",
    "project_name": "...",
    "community_name": "...",
    "communities": [...],
    "projects": [...],
    "sub_communities": [...],
    "documents": [...]
}
"""


class HybridContextBuilder:

    def __init__(self):
        pass

    # ============================================================
    # MAIN METHOD
    # ============================================================

    def build_context(
        self,
        retrieval_result=None,
        structured_summary=None,
        investment_analysis=None,
        document_results=None,
        community_results=None,
    ):
        """
        Build one combined context.

        Supports both the new single-result interface and the older
        keyword-based interface used by run.py.
        """

        # Normalize old run.py arguments into the canonical retrieval result.
        if retrieval_result is None:
            structured_summary = structured_summary or {}
            retrieval_result = {
                "communities": structured_summary.get("community", []) or [],
                "projects": structured_summary.get("projects", []) or [],
                "sub_communities": structured_summary.get("sub_communities", []) or [],
                "documents": document_results or [],
            }

        if not retrieval_result:
            return "No relevant information was retrieved."

        context_parts = []

        # ========================================================
        # STRUCTURED DATA
        # ========================================================

        structured_context = self._build_structured_context(
            retrieval_result
        )

        if structured_context:
            context_parts.append(
                structured_context
            )

        # ========================================================
        # PGVECTOR DOCUMENT DATA
        # ========================================================

        document_context = self._build_document_context(
            retrieval_result
        )

        if document_context:
            context_parts.append(
                document_context
            )

        # ========================================================
        # FINAL CONTEXT
        # ========================================================

        if not context_parts:
            return "No relevant information was retrieved."

        return "\n\n".join(context_parts)

    # ============================================================
    # STRUCTURED CONTEXT
    # ============================================================

    def _build_structured_context(
        self,
        retrieval_result,
    ):

        communities = (
            retrieval_result.get(
                "communities",
                []
            )
            or []
        )

        projects = (
            retrieval_result.get(
                "projects",
                []
            )
            or []
        )

        sub_communities = (
            retrieval_result.get(
                "sub_communities",
                []
            )
            or []
        )

        # --------------------------------------------------------
        # Nothing structured
        # --------------------------------------------------------

        if not (
            communities
            or projects
            or sub_communities
        ):
            return ""

        context = """

SUPABASE STRUCTURED REAL ESTATE DATA

The following information was retrieved directly from the
Supabase PostgreSQL database.

Use this as the primary source for structured real-estate
information such as projects, communities, developers,
prices, bedrooms, property types, handover dates and URLs.

"""
        # ========================================================
        # PROJECTS
        # ========================================================

        if projects:

            context += """

PROJECTS

"""
            for index, project in enumerate(
                projects,
                start=1
            ):

                context += (
                    f"PROJECT {index}\n\n"
                )

                context += (
                    f"Project ID:\n"
                    f"{project.get('id', 'N/A')}\n\n"
                )

                context += (
                    f"Project Name:\n"
                    f"{project.get('name', 'N/A')}\n\n"
                )

                context += (
                    f"Description:\n"
                    f"{project.get('description', 'N/A')}\n\n"
                )

                context += (
                    f"Status:\n"
                    f"{project.get('status', 'N/A')}\n\n"
                )

                context += (
                    f"Project Status:\n"
                    f"{project.get('project_status', 'N/A')}\n\n"
                )

                context += (
                    f"Developer:\n"
                    f"{project.get('developer_name', 'N/A')}\n\n"
                )

                context += (
                    f"Developer ID:\n"
                    f"{project.get('developer_id', 'N/A')}\n\n"
                )

                context += (
                    f"City:\n"
                    f"{project.get('city', 'N/A')}\n\n"
                )

                context += (
                    f"Community:\n"
                    f"{project.get('community', 'N/A')}\n\n"
                )

                context += (
                    f"Sub-community:\n"
                    f"{project.get('sub_community', 'N/A')}\n\n"
                )

                context += (
                    f"Latitude:\n"
                    f"{project.get('latitude', 'N/A')}\n\n"
                )

                context += (
                    f"Longitude:\n"
                    f"{project.get('longitude', 'N/A')}\n\n"
                )

                # ------------------------------------------------
                # Price
                # ------------------------------------------------

                price = project.get(
                    "price"
                )

                if price is not None:

                    context += (
                        f"Price:\n"
                        f"AED {price}\n\n"
                    )

                else:

                    context += (
                        "Price:\n"
                        "N/A\n\n"
                    )

                # ------------------------------------------------
                # Size
                # ------------------------------------------------

                context += (
                    f"Minimum Size:\n"
                    f"{project.get('size_min', 'N/A')} sqft\n\n"
                )

                context += (
                    f"Maximum Size:\n"
                    f"{project.get('size_max', 'N/A')} sqft\n\n"
                )

                # ------------------------------------------------
                # Bedrooms
                # ------------------------------------------------

                context += (
                    f"Minimum Bedrooms:\n"
                    f"{project.get('bedroom_min', 'N/A')}\n\n"
                )

                context += (
                    f"Maximum Bedrooms:\n"
                    f"{project.get('bedroom_max', 'N/A')}\n\n"
                )

                # ------------------------------------------------
                # Property Types
                # ------------------------------------------------

                property_types = project.get(
                    "property_types"
                )

                if property_types:

                    if isinstance(
                        property_types,
                        list
                    ):

                        property_types_text = (
                            ", ".join(
                                str(x)
                                for x in property_types
                            )
                        )

                    else:

                        property_types_text = str(
                            property_types
                        )

                else:

                    property_types_text = "N/A"

                context += (
                    f"Property Types:\n"
                    f"{property_types_text}\n\n"
                )

                # ------------------------------------------------
                # Pool Type
                # ------------------------------------------------

                context += (
                    f"Pool Type:\n"
                    f"{project.get('pool_type', 'N/A')}\n\n"
                )

                # ------------------------------------------------
                # Handover
                # ------------------------------------------------

                context += (
                    f"Handover:\n"
                    f"{project.get('handover_time', 'N/A')}\n\n"
                )

                # ------------------------------------------------
                # Amenities codes
                # ------------------------------------------------

                amenities = project.get(
                    "amenities"
                )

                if amenities:

                    if isinstance(
                        amenities,
                        list
                    ):

                        amenities_text = (
                            ", ".join(
                                str(x)
                                for x in amenities
                            )
                        )

                    else:

                        amenities_text = str(
                            amenities
                        )

                else:

                    amenities_text = "N/A"

                context += (
                    f"Amenities:\n"
                    f"{amenities_text}\n\n"
                )

                # ------------------------------------------------
                # URLs
                # ------------------------------------------------

                context += (
                    f"Brochure URL:\n"
                    f"{project.get('brochure_url', 'N/A')}\n\n"
                )

                context += (
                    f"Property Finder URL:\n"
                    f"{project.get('property_finder_url', 'N/A')}\n\n"
                )

                context += (
                    f"Source:\n"
                    f"{project.get('source', 'N/A')}\n\n"
                )

                # ------------------------------------------------
                # Knowledge text
                # ------------------------------------------------

                knowledge_text = project.get(
                    "knowledge_text"
                )

                if knowledge_text:

                    context += (
                        "Knowledge:\n"
                        f"{knowledge_text}\n\n"
                    )

                context += (
                    "------------------------\n\n"
                )

        # ========================================================
        # COMMUNITIES
        # ========================================================

        if communities:

            context += """

COMMUNITIES

"""
            for index, community in enumerate(
                communities,
                start=1
            ):

                context += (
                    f"COMMUNITY {index}\n\n"
                )

                context += (
                    f"Community ID:\n"
                    f"{community.get('id', 'N/A')}\n\n"
                )

                context += (
                    f"Name:\n"
                    f"{community.get('name', 'N/A')}\n\n"
                )

                context += (
                    f"Slug:\n"
                    f"{community.get('slug', 'N/A')}\n\n"
                )

                context += (
                    f"City:\n"
                    f"{community.get('city', 'N/A')}\n\n"
                )

                context += (
                    f"Knowledge:\n"
                    f"{community.get('knowledge_text', 'N/A')}\n\n"
                )

                context += (
                    "------------------------\n\n"
                )

        # ========================================================
        # SUB COMMUNITIES
        # ========================================================

        if sub_communities:

            context += """

SUB-COMMUNITIES

"""
            for index, sub in enumerate(
                sub_communities,
                start=1
            ):

                context += (
                    f"SUB-COMMUNITY {index}\n\n"
                )

                context += (
                    f"Name:\n"
                    f"{sub.get('name', 'N/A')}\n\n"
                )

                context += (
                    f"Community:\n"
                    f"{sub.get('community', 'N/A')}\n\n"
                )

                context += (
                    f"City:\n"
                    f"{sub.get('city', 'N/A')}\n\n"
                )

                context += (
                    f"Knowledge:\n"
                    f"{sub.get('knowledge_text', 'N/A')}\n\n"
                )

                context += (
                    "------------------------\n\n"
                )

        return context

    # ============================================================
    # DOCUMENT CONTEXT
    # ============================================================

    def _build_document_context(
        self,
        retrieval_result,
    ):

        documents = (
            retrieval_result.get(
                "documents",
                []
            )
            or []
        )

        if not documents:
            return ""

        context = """

SUPABASE PGVECTOR DOCUMENT KNOWLEDGE

The following information was retrieved from document chunks
stored in Supabase pgvector.

Use this information for document-specific questions such as
amenities, security features, interiors, finishes, views,
floor plans, dimensions and facilities.

Do not invent information that is not present in these
retrieved document chunks.

"""

        seen_chunks = set()

        for index, item in enumerate(
            documents,
            start=1
        ):

            document_id = item.get(
                "document_id"
            )

            document_title = item.get(
                "document_title",
                "Unknown Document"
            )

            document_url = item.get(
                "document_url",
                ""
            )

            document_type = item.get(
                "document_type",
                "unknown"
            )

            publisher = item.get(
                "publisher",
                "Unknown Publisher"
            )

            chunk_id = item.get(
                "chunk_id"
            )

            content = item.get(
                "content",
                ""
            )

            similarity = item.get(
                "similarity"
            )

            unique_key = (
                document_id,
                chunk_id,
            )

            if unique_key in seen_chunks:
                continue

            seen_chunks.add(
                unique_key
            )

            context += (
                f"\nDOCUMENT CHUNK {index}\n\n"
            )

            context += (
                f"Document ID:\n"
                f"{document_id or 'N/A'}\n\n"
            )

            context += (
                f"Document Title:\n"
                f"{document_title}\n\n"
            )

            context += (
                f"Document Type:\n"
                f"{document_type}\n\n"
            )

            context += (
                f"Publisher:\n"
                f"{publisher}\n\n"
            )

            if similarity is not None:

                context += (
                    f"Similarity:\n"
                    f"{similarity:.4f}\n\n"
                )

            context += (
                f"Source URL:\n"
                f"{document_url}\n\n"
            )

            context += (
                "Content:\n"
                f"{content}\n\n"
            )

            context += (
                "------------------------\n"
            )

        return context