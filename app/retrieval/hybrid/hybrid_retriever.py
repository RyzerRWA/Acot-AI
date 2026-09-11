"""
ACOT Hybrid Retriever

Pipeline:

    User Question
          ↓
    Query Planner
          ↓
    Entity Resolver
          ↓
    Structured PostgreSQL Retrieval
          ↓
    Optional pgvector Document Retrieval
          ↓
    Consistent Retrieval Result

Structured data:
    communities
    projects
    sub_communities

Document data:
    document_chunks
"""

from typing import Any, Dict, List, Optional


class HybridRetriever:
    """
    Main retrieval layer for ACOT.

    Responsibilities:
        1. Plan the query
        2. Resolve real entities from Supabase
        3. Retrieve structured PostgreSQL data
        4. Retrieve pgvector documents only when required
        5. Apply supported user filters
        6. Return one consistent retrieval structure
    """

    def __init__(
        self,
        structured_retriever=None,
        document_retriever=None,
        pgvector_retriever=None,
        router=None,
        query_planner=None,
        entity_resolver=None,
    ):

        self.structured_retriever = (
            structured_retriever
        )

        # Backward compatibility
        self.document_retriever = (
            document_retriever
        )

        self.pgvector_retriever = (
            pgvector_retriever
            if pgvector_retriever is not None
            else document_retriever
        )

        # --------------------------------------------------------
        # Router
        # --------------------------------------------------------

        if router is not None:

            self.router = router

        else:

            from app.retrieval.hybrid.router import (
                HybridRouter
            )

            self.router = HybridRouter()

        # --------------------------------------------------------
        # Query Planner
        # --------------------------------------------------------

        if query_planner is not None:

            self.query_planner = query_planner

        else:

            from app.retrieval.hybrid.query_planner import (
                QueryPlanner
            )

            self.query_planner = QueryPlanner()

        # --------------------------------------------------------
        # Entity Resolver
        # --------------------------------------------------------

        if entity_resolver is not None:

            self.entity_resolver = (
                entity_resolver
            )

        else:

            from app.retrieval.hybrid.entity_resolver import (
                EntityResolver
            )

            if structured_retriever is None:

                self.entity_resolver = None

            else:

                self.entity_resolver = (
                    EntityResolver(
                        structured_retriever
                    )
                )

    # ============================================================
    # EMPTY RESULT
    # ============================================================

    @staticmethod
    def _empty_structured_result():

        return {
            "communities": [],
            "projects": [],
            "sub_communities": [],
        }

    # ============================================================
    # APPLY PROJECT FILTERS
    # ============================================================

    @staticmethod
    def _apply_project_filters(
        projects: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not projects:
            return []

        if not filters:
            return projects

        filtered = []

        max_price = filters.get(
            "max_price_aed"
        )

        min_price = filters.get(
            "min_price_aed"
        )

        requested_property_types = (
            filters.get(
                "property_types"
            )
            or []
        )

        # The query planner normally gives us either one exact bedroom
        # count (`bedrooms`) or an explicit range. Support both forms.
        requested_bedrooms = filters.get("bedrooms")

        min_bedrooms = filters.get(
            "min_bedrooms"
        )

        max_bedrooms = filters.get(
            "max_bedrooms"
        )

        if requested_bedrooms is not None:
            min_bedrooms = requested_bedrooms
            max_bedrooms = requested_bedrooms

        for project in projects:

            # ----------------------------------------------------
            # PRICE
            # ----------------------------------------------------

            price = project.get(
                "price"
            )

            if max_price is not None:

                if price is None:
                    continue

                try:

                    if float(price) > float(
                        max_price
                    ):
                        continue

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            if min_price is not None:

                if price is None:
                    continue

                try:

                    if float(price) < float(
                        min_price
                    ):
                        continue

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            # ----------------------------------------------------
            # PROPERTY TYPE
            # ----------------------------------------------------

            if requested_property_types:

                project_types = (
                    project.get(
                        "property_types"
                    )
                    or []
                )

                normalized_project_types = {
                    str(value).lower()
                    for value in project_types
                }

                requested_types = {
                    str(value).lower()
                    for value in requested_property_types
                }

                if not (
                    normalized_project_types
                    &
                    requested_types
                ):
                    continue

            # ----------------------------------------------------
            # BEDROOMS
            # ----------------------------------------------------

            bedroom_min = project.get(
                "bedroom_min"
            )

            bedroom_max = project.get(
                "bedroom_max"
            )

            if min_bedrooms is not None:

                if bedroom_max is None:
                    continue

                try:

                    if float(bedroom_max) < float(
                        min_bedrooms
                    ):
                        continue

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            if max_bedrooms is not None:

                if bedroom_min is None:
                    continue

                try:

                    if float(bedroom_min) > float(
                        max_bedrooms
                    ):
                        continue

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            filtered.append(
                project
            )

        return filtered

    # ============================================================
    # STRUCTURED RETRIEVAL BY PROJECT
    # ============================================================

    def _retrieve_by_project(
        self,
        project_name: str,
    ) -> Dict[str, Any]:

        if not self.structured_retriever:
            return self._empty_structured_result()

        print(
            f"Structured lookup by project: "
            f"{project_name}"
        )

        try:

            # Preferred current API
            projects = (
                self.structured_retriever
                .search_projects_by_name(
                    project_name
                )
                or []
            )

        except AttributeError:

            try:

                result = (
                    self.structured_retriever
                    .retrieve(
                        project_name=project_name
                    )
                    or {}
                )

                projects = (
                    result.get(
                        "projects",
                        []
                    )
                    or []
                )

            except Exception as exc:

                print(
                    "Project retrieval error:",
                    exc
                )

                return (
                    self._empty_structured_result()
                )

        if not projects:

            return (
                self._empty_structured_result()
            )

        communities = []

        sub_communities = []

        # --------------------------------------------------------
        # Resolve hierarchy from project record
        # --------------------------------------------------------

        project_community = (
            projects[0].get(
                "community"
            )
        )

        if project_community:

            try:

                communities = (
                    self.structured_retriever
                    .search_community(
                        project_community
                    )
                    or []
                )

            except Exception as exc:

                print(
                    "Community lookup error:",
                    exc
                )

            try:

                sub_communities = (
                    self.structured_retriever
                    .search_sub_communities(
                        project_community
                    )
                    or []
                )

            except Exception as exc:

                print(
                    "Sub-community lookup error:",
                    exc
                )

        return {
            "communities": communities,
            "projects": projects,
            "sub_communities":
                sub_communities,
        }

    # ============================================================
    # STRUCTURED RETRIEVAL BY COMMUNITY
    # ============================================================

    def _retrieve_by_community(
        self,
        community_name: str,
    ) -> Dict[str, Any]:

        if not self.structured_retriever:
            return self._empty_structured_result()

        print(
            f"Structured lookup by community: "
            f"{community_name}"
        )

        try:

            result = (
                self.structured_retriever
                .retrieve(
                    community_name=
                        community_name
                )
                or {}
            )

            # `SupabaseStructuredRetriever.retrieve()` returns the
            # community collection under `community`. Keep accepting
            # `communities` too so the retrieval layer stays compatible
            # with older callers.
            communities = (
                result.get("communities")
                or result.get("community")
                or []
            )

            return {
                "communities": communities,
                "projects": result.get("projects", []) or [],
                "sub_communities": (
                    result.get("sub_communities", [])
                    or []
                ),
            }

        except Exception as exc:

            print(
                "Community retrieval error:",
                exc
            )

            return (
                self._empty_structured_result()
            )

    # ============================================================
    # STRUCTURED RETRIEVAL BY CITY
    # ============================================================

    def _retrieve_by_city(
        self,
        city_name: str,
    ) -> Dict[str, Any]:

        if not self.structured_retriever:
            return self._empty_structured_result()

        print(
            f"Structured lookup by city: "
            f"{city_name}"
        )

        try:

            supabase = (
                self.structured_retriever
                .supabase
            )

            # ----------------------------------------------------
            # Communities in city
            # ----------------------------------------------------

            community_response = (
                supabase
                .table("communities")
                .select("*")
                .ilike(
                    "city",
                    city_name
                )
                .limit(1000)
                .execute()
            )

            communities = (
                community_response.data
                or []
            )

            # ----------------------------------------------------
            # Projects in city
            # ----------------------------------------------------

            project_response = (
                supabase
                .table("projects")
                .select("*")
                .ilike(
                    "city",
                    city_name
                )
                .limit(1000)
                .execute()
            )

            projects = (
                project_response.data
                or []
            )

            # ----------------------------------------------------
            # Sub-communities in city
            # ----------------------------------------------------

            subcommunity_response = (
                supabase
                .table("sub_communities")
                .select("*")
                .ilike(
                    "city",
                    city_name
                )
                .limit(1000)
                .execute()
            )

            sub_communities = (
                subcommunity_response.data
                or []
            )

            return {
                "communities":
                    communities,

                "projects":
                    projects,

                "sub_communities":
                    sub_communities,
            }

        except Exception as exc:

            print(
                "City retrieval error:",
                exc
            )

            return (
                self._empty_structured_result()
            )

    # ============================================================
    # QUERY HELPERS
    # ============================================================

    @staticmethod
    def _needs_structured_data_from_question(question: str) -> bool:
        """Return True when the wording clearly asks for database fields."""
        if not question:
            return False

        q = question.lower()
        structured_terms = (
            "price",
            "prices",
            "cost",
            "rent",
            "rental price",
            "bedroom",
            "bedrooms",
            "size",
            "area",
            "sqft",
            "square foot",
            "developer",
            "handover",
            "status",
            "projects",
            "project",
            "properties",
            "property",
        )
        return any(term in q for term in structured_terms)

    @staticmethod
    def _normalize_structured_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize the slightly different structured-retriever return shapes."""
        result = result or {}
        return {
            "communities": (
                result.get("communities")
                or result.get("community")
                or []
            ),
            "projects": result.get("projects", []) or [],
            "sub_communities": result.get("sub_communities", []) or [],
        }

    # ============================================================
    # STRUCTURED RETRIEVAL
    # ============================================================

    def retrieve_structured_data(
        self,
        question: str,
        resolved_entity=None,
        filters: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        if not self.structured_retriever:

            print(
                "WARNING: structured_retriever "
                "is None"
            )

            return (
                self._empty_structured_result()
            )

        try:

            # ----------------------------------------------------
            # PROJECT
            # ----------------------------------------------------

            if (
                resolved_entity
                and
                resolved_entity.entity_type
                == "project"
            ):

                result = (
                    self._retrieve_by_project(
                        resolved_entity.name
                    )
                )

            # ----------------------------------------------------
            # COMMUNITY
            # ----------------------------------------------------

            elif (
                resolved_entity
                and
                resolved_entity.entity_type
                == "community"
            ):

                result = (
                    self._retrieve_by_community(
                        resolved_entity.name
                    )
                )

            # ----------------------------------------------------
            # CITY
            # ----------------------------------------------------

            elif (
                resolved_entity
                and
                resolved_entity.entity_type
                == "city"
            ):

                result = (
                    self._retrieve_by_city(
                        resolved_entity.name
                    )
                )

            else:

                result = (
                    self._empty_structured_result()
                )

            # ----------------------------------------------------
            # NORMALIZE RESULT KEYS
            # ----------------------------------------------------

            result = self._normalize_structured_result(result)

            # ----------------------------------------------------
            # APPLY FILTERS
            # ----------------------------------------------------

            result["projects"] = (
                self._apply_project_filters(
                    result.get("projects", []),
                    filters,
                )
            )

            return result

        except Exception as exc:

            print(
                f"Structured retrieval error: "
                f"{exc}"
            )

            return (
                self._empty_structured_result()
            )

    # ============================================================
    # DOCUMENT RETRIEVAL
    # ============================================================

    def retrieve_documents(
        self,
        question: str,
    ) -> List[Dict[str, Any]]:

        if self.pgvector_retriever is None:

            print(
                "WARNING: pgvector_retriever "
                "is None"
            )

            return []

        try:

            documents = (
                self.pgvector_retriever
                .search_documents(
                    question=question,
                    match_threshold=0.30,
                    match_count=5,
                )
                or []
            )

            return documents

        except Exception as exc:

            print(
                f"Document retrieval error: "
                f"{exc}"
            )

            return []

    # ============================================================
    # MAIN RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        question: str,
        community_name: Optional[str] = None,
        conversation_context:
            Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        print()
        print("================================")
        print("HYBRID RETRIEVER")
        print("================================")

        # ========================================================
        # 1. QUERY PLANNING
        # ========================================================

        try:

            query_plan = (
                self.query_planner.plan(
                    question,
                    conversation_context=
                        conversation_context,
                )
            )

            print(
                "\nQuery Plan:"
            )

            print(
                query_plan.to_dict()
            )

        except Exception as exc:

            print(
                "Query planner error:",
                exc
            )

            query_plan = None

        # ========================================================
        # 2. ENTITY RESOLUTION
        # ========================================================

        resolved_entity = None

        if self.entity_resolver:

            try:

                resolved_entity = (
                    self.entity_resolver.resolve(
                        question
                    )
                )

                print(
                    "\nEntity Resolution:"
                )

                print(
                    resolved_entity.to_dict()
                )

            except Exception as exc:

                print(
                    "Entity resolver error:",
                    exc
                )

        # ========================================================
        # 3. FALLBACK COMMUNITY
        # ========================================================
        #
        # Backward compatibility only.
        #
        # New run.py will no longer use
        # hardcoded community detection.
        #

        if (
            resolved_entity is None
            and community_name
        ):

            class FallbackEntity:

                entity_type = "community"
                name = community_name

            resolved_entity = (
                FallbackEntity()
            )

        # ========================================================
        # 4. ROUTING
        # ========================================================

        try:

            route_result = (
                self.router.route(
                    question
                )
            )

            if isinstance(
                route_result,
                dict
            ):

                route = (
                    route_result.get(
                        "route",
                        "hybrid"
                    )
                )

                router_question_type = (
                    route_result.get(
                        "question_type",
                        "hybrid_query"
                    )
                )

            else:

                route = str(
                    route_result
                )

                router_question_type = (
                    "hybrid_query"
                    if route == "hybrid"
                    else route
                )

        except Exception as exc:

            print(
                "Router error:",
                exc
            )

            route = "hybrid"

            router_question_type = (
                "hybrid_query"
            )

        # ========================================================
        # 5. DETERMINE DATA REQUIREMENTS
        # ========================================================

        if query_plan:

            question_type = (
                query_plan.intent
            )

            needs_structured = (
                query_plan.needs_structured_data
                or route in {"structured", "hybrid"}
                or self._needs_structured_data_from_question(question)
            )

            needs_documents = (
                query_plan.needs_documents
            )

            filters = (
                query_plan.filters
                or {}
            )

        else:

            question_type = (
                router_question_type
            )

            needs_structured = (
                route in (
                    "structured",
                    "hybrid",
                )
            )

            needs_documents = (
                route in (
                    "documents",
                    "hybrid",
                )
                or self._document_terms(
                    question
                )
            )

            filters = {}

        # ========================================================
        # IMPORTANT DOCUMENT RULE
        # ========================================================
        #
        # Investment / ranking / comparison
        # queries should not automatically
        # retrieve brochures.
        #
        # This prevents:
        #
        # "Is Dubai Marina a good investment?"
        #
        # from retrieving:
        #
        # "Sky Edition brochure"
        #

        if query_plan:

            if (
                query_plan.intent
                in {
                    "investment",
                    "ranking",
                    "comparison",
                    "calculation",
                    "prediction",
                }
            ):

                # Keep analytical queries grounded in structured data.
                # Brochure retrieval is enabled only when the planner
                # explicitly requested documents.
                needs_documents = (
                    query_plan.needs_documents
                )

            # Questions such as "price and amenities" are often classified
            # as document queries because of the word "amenities". The
            # structured part must still be retrieved for price, bedrooms,
            # developer, handover, and other database-backed fields.
            if self._needs_structured_data_from_question(question):
                needs_structured = True

        # ========================================================
        # 6. PRINT PIPELINE
        # ========================================================

        print(
            f"\nRoute: {route}"
        )

        print(
            f"Question Type: "
            f"{question_type}"
        )

        if resolved_entity:

            print(
                "Resolved Entity:",
                resolved_entity.entity_type,
                resolved_entity.name,
            )

        else:

            print(
                "Resolved Entity: None"
            )

        print(
            "Needs Structured:",
            needs_structured
        )

        print(
            "Needs Documents:",
            needs_documents
        )

        print(
            "Filters:",
            filters
        )

        # ========================================================
        # 7. RETRIEVE STRUCTURED DATA
        # ========================================================

        communities = []

        projects = []

        sub_communities = []

        if needs_structured:

            print()
            print(
                "Retrieving structured "
                "PostgreSQL data..."
            )

            structured_result = (
                self.retrieve_structured_data(
                    question=question,
                    resolved_entity=
                        resolved_entity,
                    filters=filters,
                )
            )

            communities = (
                structured_result.get(
                    "communities",
                    []
                )
                or []
            )

            projects = (
                structured_result.get(
                    "projects",
                    []
                )
                or []
            )

            sub_communities = (
                structured_result.get(
                    "sub_communities",
                    []
                )
                or []
            )

        # ========================================================
        # 8. RETRIEVE DOCUMENTS
        # ========================================================

        documents = []

        if needs_documents:

            print()
            print(
                "Retrieving Supabase "
                "PGVector documents..."
            )

            documents = (
                self.retrieve_documents(
                    question
                )
            )

        # ========================================================
        # 9. ENTITY INFORMATION
        # ========================================================

        resolved_entity_dict = None

        if resolved_entity:

            try:

                resolved_entity_dict = (
                    resolved_entity.to_dict()
                )

            except Exception:

                resolved_entity_dict = {
                    "entity_type":
                        getattr(
                            resolved_entity,
                            "entity_type",
                            None
                        ),

                    "name":
                        getattr(
                            resolved_entity,
                            "name",
                            None
                        ),
                }

        # ========================================================
        # 10. FINAL SUMMARY
        # ========================================================

        print()
        print("================================")
        print("RETRIEVAL SUMMARY")
        print("================================")

        print(
            f"Communities found: "
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

        print(
            f"Documents found: "
            f"{len(documents)}"
        )

        # ========================================================
        # 11. STRUCTURED DATA
        # ========================================================

        structured_data = {

            "community":
                communities,

            "projects":
                projects,

            "sub_communities":
                sub_communities,
        }

        # ========================================================
        # 12. RETURN
        # ========================================================

        return {

            "route":
                route,

            "question_type":
                question_type,

            "router_question_type":
                router_question_type,

            "query_plan":
                (
                    query_plan.to_dict()
                    if query_plan
                    else None
                ),

            "resolved_entity":
                resolved_entity_dict,

            "project_name":
                (
                    resolved_entity.name
                    if (
                        resolved_entity
                        and
                        resolved_entity.entity_type
                        == "project"
                    )
                    else None
                ),

            "community_name":
                (
                    resolved_entity.name
                    if (
                        resolved_entity
                        and
                        resolved_entity.entity_type
                        == "community"
                    )
                    else (
                        resolved_entity.community.get(
                            "name"
                        )
                        if (
                            resolved_entity
                            and
                            getattr(
                                resolved_entity,
                                "community",
                                None
                            )
                        )
                        else None
                    )
                ),

            "city_name":
                (
                    resolved_entity.name
                    if (
                        resolved_entity
                        and
                        resolved_entity.entity_type
                        == "city"
                    )
                    else (
                        resolved_entity.location.get(
                            "city"
                        )
                        if (
                            resolved_entity
                            and
                            getattr(
                                resolved_entity,
                                "location",
                                None
                            )
                        )
                        else None
                    )
                ),

            "filters":
                filters,

            # Canonical
            "communities":
                communities,

            "projects":
                projects,

            "sub_communities":
                sub_communities,

            "documents":
                documents,

            # Compatibility
            "structured_data":
                structured_data,
        }

    # ============================================================
    # SIMPLE DOCUMENT TERM FALLBACK
    # ============================================================

    @staticmethod
    def _document_terms(
        question: str
    ) -> bool:

        if not question:
            return False

        q = question.lower()

        document_terms = [

            "brochure",
            "document",
            "documents",

            "building code",
            "building codes",

            "requirement",
            "requirements",

            "regulation",
            "regulations",

            "law",
            "laws",

            "safety",

            "security",
            "security features",

            "guideline",
            "guidelines",

            "amenities",
            "amenity",

            "facilities",
            "facility",

            "features",

            "interior",
            "interiors",

            "finish",
            "finishes",

            "kitchen",
            "appliances",

            "views",
            "view",

            "floor plan",
            "floor plans",

            "layout",
            "layouts",

            "dimensions",

            "clubhouse",
            "pool",
            "gym",
            "cinema",
            "lobby",

            "surveillance",

            "balcony",
            "bathroom",

            "smart home",
            "smart wc",

            "concierge",
            "valet",

            "sauna",
            "steam",
        ]

        return any(
            term in q
            for term in document_terms
        )