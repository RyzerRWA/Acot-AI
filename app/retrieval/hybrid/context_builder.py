"""
ACOT Hybrid Context Builder

Builds compact, grounded context for the RAG/LLM layer from:

1. Supabase PostgreSQL structured data
2. Supabase pgvector document chunks

Optimization goals:
- Preserve project/community/sub-community ordering.
- Keep the fields needed for factual answers.
- Remove unnecessary IDs, coordinates, URLs, and duplicate metadata from
  the Gemini prompt.
- Bound long description/knowledge/document text so prompt size stays small.
- Keep the existing build_context() interface compatible with run.py.
"""


class HybridContextBuilder:

    # Keep long database text bounded so simple requests do not send huge
    # descriptions/knowledge fields to Gemini.
    MAX_DESCRIPTION_CHARS = 2200
    MAX_KNOWLEDGE_CHARS = 1200
    MAX_DOCUMENT_CHARS = 3500

    def __init__(self):
        pass

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _compact_text(value, max_chars):
        """Return a compact single-line text value with a safe length."""
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "..."

    @staticmethod
    def _list_text(value):
        """Convert list-like database fields into compact readable text."""
        if not value:
            return None

        if isinstance(value, (list, tuple)):
            values = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(values) if values else None

        return str(value).strip() or None

    @staticmethod
    def _value(value, default="N/A"):
        return default if value is None or value == "" else value

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
        Build one compact combined context.

        Supports both the new single-result interface and the existing
        keyword-based interface used by run.py.
        """

        # Normalize existing run.py arguments into the canonical structure.
        if retrieval_result is None:
            structured_summary = structured_summary or {}

            retrieval_result = {
                "communities": structured_summary.get(
                    "community", []
                ) or [],
                "projects": structured_summary.get(
                    "projects", []
                ) or [],
                "sub_communities": structured_summary.get(
                    "sub_communities", []
                ) or [],
                "documents": document_results or [],
            }

        if not retrieval_result:
            return "No relevant information was retrieved."

        context_parts = []

        structured_context = self._build_structured_context(
            retrieval_result
        )

        if structured_context:
            context_parts.append(structured_context)

        document_context = self._build_document_context(
            retrieval_result
        )

        if document_context:
            context_parts.append(document_context)

        # Include investment analysis when a caller supplies it.
        # This keeps the builder compatible with future direct callers,
        # while run.py can continue to use its existing interface.
        if investment_analysis:
            investment_context = self._build_investment_context(
                investment_analysis
            )

            if investment_context:
                context_parts.append(investment_context)

        if not context_parts:
            return "No relevant information was retrieved."

        return "\n\n".join(context_parts)

    # ============================================================
    # STRUCTURED CONTEXT
    # ============================================================

    def _build_structured_context(self, retrieval_result):

        communities = (
            retrieval_result.get("communities", [])
            or retrieval_result.get("community", [])
            or []
        )

        projects = (
            retrieval_result.get("projects", [])
            or []
        )

        sub_communities = (
            retrieval_result.get("sub_communities", [])
            or []
        )

        if not (
            communities
            or projects
            or sub_communities
        ):
            return ""

        context_parts = [
            "SUPABASE STRUCTURED REAL ESTATE DATA",
            "",
            "Use this as the primary source for structured facts.",
            "Project order below is authoritative and must be preserved.",
            "",
        ]

        # ========================================================
        # PROJECTS
        # ========================================================

        if projects:
            context_parts.extend([
                "PROJECTS",
                "",
            ])

            for index, project in enumerate(projects, start=1):

                name = self._value(
                    project.get("name")
                )

                developer = self._value(
                    project.get("developer_name")
                )

                city = self._value(
                    project.get("city")
                )

                community = self._value(
                    project.get("community")
                )

                sub_community = self._value(
                    project.get("sub_community")
                )

                price = project.get("price")

                if price is None:
                    price_text = "N/A"
                else:
                    price_text = f"AED {price}"

                bedroom_min = project.get("bedroom_min")
                bedroom_max = project.get("bedroom_max")

                if (
                    bedroom_min is None
                    and bedroom_max is None
                ):
                    bedrooms_text = "N/A"
                elif bedroom_min is None:
                    bedrooms_text = str(bedroom_max)
                elif bedroom_max is None:
                    bedrooms_text = str(bedroom_min)
                elif bedroom_min == bedroom_max:
                    bedrooms_text = str(bedroom_min)
                else:
                    bedrooms_text = (
                        f"{bedroom_min}-{bedroom_max}"
                    )

                size_min = project.get("size_min")
                size_max = project.get("size_max")

                if size_min is None and size_max is None:
                    size_text = "N/A"
                elif size_min is None:
                    size_text = f"{size_max} sqft"
                elif size_max is None:
                    size_text = f"{size_min} sqft"
                else:
                    size_text = (
                        f"{size_min}-{size_max} sqft"
                    )

                property_types = self._list_text(
                    project.get("property_types")
                )

                amenities = self._list_text(
                    project.get("amenities")
                )

                description = self._compact_text(
                    project.get("description"),
                    self.MAX_DESCRIPTION_CHARS,
                )

                knowledge = self._compact_text(
                    project.get("knowledge_text"),
                    self.MAX_KNOWLEDGE_CHARS,
                )

                context_parts.extend([
                    f"PROJECT {index}",
                    f"Name: {name}",
                    f"Developer: {developer}",
                    f"City: {city}",
                    f"Community: {community}",
                    f"Sub-community: {sub_community}",
                    f"Price: {price_text}",
                    f"Bedrooms: {bedrooms_text}",
                    f"Size: {size_text}",
                    f"Property Types: {property_types or 'N/A'}",
                    f"Status: {self._value(project.get('status'))}",
                    (
                        "Project Status: "
                        f"{self._value(project.get('project_status'))}"
                    ),
                    (
                        "Handover: "
                        f"{self._value(project.get('handover_time'))}"
                    ),
                    f"Amenities: {amenities or 'N/A'}",
                ])

                if description:
                    context_parts.append(
                        f"Description: {description}"
                    )

                if knowledge:
                    context_parts.append(
                        f"Knowledge: {knowledge}"
                    )

                context_parts.extend([
                    "",
                    "------------------------",
                    "",
                ])

        # ========================================================
        # COMMUNITIES
        # ========================================================

        if communities:
            context_parts.extend([
                "COMMUNITIES",
                "",
            ])

            for index, community in enumerate(
                communities,
                start=1,
            ):

                knowledge = self._compact_text(
                    community.get("knowledge_text"),
                    self.MAX_KNOWLEDGE_CHARS,
                )

                context_parts.extend([
                    f"COMMUNITY {index}",
                    f"Name: {self._value(community.get('name'))}",
                    f"Slug: {self._value(community.get('slug'))}",
                    f"City: {self._value(community.get('city'))}",
                ])

                # Keep useful inventory information but omit IDs,
                # coordinates and other metadata from the LLM prompt.
                inventory_fields = (
                    "sell_properties_count",
                    "rent_properties_count",
                    "projects_count",
                    "pool_projects_count",
                    "total_count",
                )

                inventory = []

                for field in inventory_fields:
                    value = community.get(field)

                    if value is not None:
                        inventory.append(
                            f"{field}={value}"
                        )

                if inventory:
                    context_parts.append(
                        "Inventory: " + ", ".join(inventory)
                    )

                if knowledge:
                    context_parts.append(
                        f"Knowledge: {knowledge}"
                    )

                context_parts.extend([
                    "",
                    "------------------------",
                    "",
                ])

        # ========================================================
        # SUB-COMMUNITIES
        # ========================================================

        if sub_communities:
            context_parts.extend([
                "SUB-COMMUNITIES",
                "",
            ])

            for index, sub in enumerate(
                sub_communities,
                start=1,
            ):

                knowledge = self._compact_text(
                    sub.get("knowledge_text"),
                    self.MAX_KNOWLEDGE_CHARS,
                )

                context_parts.extend([
                    f"SUB-COMMUNITY {index}",
                    f"Name: {self._value(sub.get('name'))}",
                    (
                        "Community: "
                        f"{self._value(sub.get('community'))}"
                    ),
                    f"City: {self._value(sub.get('city'))}",
                ])

                if knowledge:
                    context_parts.append(
                        f"Knowledge: {knowledge}"
                    )

                context_parts.extend([
                    "",
                    "------------------------",
                    "",
                ])

        return "\n".join(context_parts).strip()

    # ============================================================
    # DOCUMENT CONTEXT
    # ============================================================

    def _build_document_context(self, retrieval_result):

        documents = (
            retrieval_result.get("documents", [])
            or []
        )

        if not documents:
            return ""

        context_parts = [
            "SUPABASE PGVECTOR DOCUMENT KNOWLEDGE",
            "",
            "Use these retrieved document chunks only for "
            "document-specific facts.",
            "",
        ]

        seen_chunks = set()
        output_index = 0

        for item in documents:

            document_id = item.get("document_id")
            chunk_id = item.get("chunk_id")

            unique_key = (
                document_id,
                chunk_id,
            )

            if unique_key in seen_chunks:
                continue

            seen_chunks.add(unique_key)
            output_index += 1

            title = self._value(
                item.get("document_title"),
                "Unknown Document",
            )

            document_type = self._value(
                item.get("document_type"),
                "unknown",
            )

            publisher = self._value(
                item.get("publisher"),
                "Unknown Publisher",
            )

            content = self._compact_text(
                item.get("content"),
                self.MAX_DOCUMENT_CHARS,
            )

            if not content:
                continue

            similarity = item.get("similarity")

            if similarity is None:
                similarity_text = "N/A"
            else:
                try:
                    similarity_text = f"{float(similarity):.4f}"
                except (TypeError, ValueError):
                    similarity_text = str(similarity)

            context_parts.extend([
                f"DOCUMENT CHUNK {output_index}",
                f"Title: {title}",
                f"Type: {document_type}",
                f"Publisher: {publisher}",
                f"Similarity: {similarity_text}",
                f"Content: {content}",
                "",
                "------------------------",
                "",
            ])

        if output_index == 0:
            return ""

        return "\n".join(context_parts).strip()

    # ============================================================
    # INVESTMENT CONTEXT
    # ============================================================

    def _build_investment_context(self, investment_analysis):

        if not isinstance(investment_analysis, dict):
            return ""

        fields = (
            ("Investment Verdict", "investment_verdict"),
            ("Investment Score", "investment_score"),
            ("Confidence", "confidence"),
            ("Summary", "summary"),
            ("Gross Estimated Rental Yield", "gross_rental_yield"),
            ("Records Found", "records_found"),
            ("Data Limitations", "limitations"),
        )

        lines = [
            "ACOT INVESTMENT ANALYSIS",
            "",
        ]

        added = False

        for label, key in fields:
            value = investment_analysis.get(key)

            if value is None or value == "":
                continue

            if isinstance(value, (list, tuple)):
                value = ", ".join(
                    str(item)
                    for item in value
                )

            value = self._compact_text(
                value,
                self.MAX_KNOWLEDGE_CHARS,
            )

            if value:
                lines.append(
                    f"{label}: {value}"
                )
                added = True

        if not added:
            return ""

        return "\n".join(lines)
