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

import re
from typing import Any, Dict, List, Optional


class HybridRetriever:

    # ============================================================
    # ENTITY ALIASES
    # ============================================================
    # Canonical names used by the structured database.
    ENTITY_ALIASES = {
        "community": {
            "jvc": "Jumeirah Village Circle",
            "jumeirah village": "Jumeirah Village Circle",
            "jumeirah village circle": "Jumeirah Village Circle",
            "dubai marina": "Dubai Marina",
        },
        "project": {},
        "property": {},
        "developer": {},
        "sub_community": {},
        "city": {},
    }

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
    # ENTITY NORMALIZATION
    # ============================================================

    @classmethod
    def _normalize_entity_name(
        cls,
        entity_type: str,
        entity_name: str,
    ) -> str:
        """Normalize a user-facing alias to a canonical database name."""
        entity_type = str(entity_type or "").strip().lower()
        entity_name = str(entity_name or "").strip()

        if not entity_name:
            return ""

        aliases = cls.ENTITY_ALIASES.get(entity_type, {})
        return aliases.get(entity_name.lower(), entity_name)

    def _extract_comparison_candidates(
        self,
        question: str,
        query_plan=None,
    ) -> List[str]:
        """Return explicit entities participating in a comparison.

        Prefer planner candidates, then supplement them from the natural
        language comparison itself. This is intentionally conservative: it
        only extracts the entity slots around common comparison operators.
        """
        candidates = []

        planned = (
            getattr(query_plan, "entity_candidates", None) or []
            if query_plan is not None
            else []
        )
        for value in planned:
            value = str(value or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        q = (question or "").strip()
        if not q:
            return candidates

        # Examples handled:
        #   Compare A and B
        #   Compare A vs B
        #   Compare A versus B
        #   Compare A & B
        match = re.search(
            r"^\s*compare\s+(.+?)\s+(?:and|vs\.?|versus|&)\s+(.+?)\s*$",
            q,
            flags=re.IGNORECASE,
        )
        if match:
            extracted = [match.group(1).strip(), match.group(2).strip()]
            for value in extracted:
                if value and value not in candidates:
                    candidates.append(value)

        return candidates

    def _resolve_one_comparison_entity(
        self,
        candidate: str,
        planned_entity_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve one comparison candidate independently.

        Unlike the old implementation, a comparison is allowed to contain
        different entity types, e.g. a project + a community.
        """
        requested_name = str(candidate or "").strip()
        if not requested_name:
            return None

        # ------------------------------------------------------------
        # 1. Explicit aliases are authoritative.
        # ------------------------------------------------------------
        candidate_lower = requested_name.lower()
        for entity_type, aliases in self.ENTITY_ALIASES.items():
            canonical = aliases.get(candidate_lower)
            if canonical:
                return {
                    "entity_type": entity_type,
                    "requested_name": requested_name,
                    "resolved_name": canonical,
                }

        # ------------------------------------------------------------
        # 2. Explicit planner type is useful only for this candidate.
        #    It must NOT force the type of every comparison candidate.
        # ------------------------------------------------------------
        valid_types = {
            "community",
            "project",
            "property",
            "developer",
            "sub_community",
            "city",
        }
        planned_type = str(planned_entity_type or "").strip().lower()

        # ------------------------------------------------------------
        # 3. Resolve against the actual structured database.
        #    Exact entity existence beats LLM classification.
        # ------------------------------------------------------------
        if self.structured_retriever is not None:
            if planned_type in {"project", "property"}:
                try:
                    matches = (
                        self.structured_retriever.search_projects_by_name(
                            requested_name
                        ) or []
                    )
                    if matches:
                        return {
                            "entity_type": "project" if planned_type == "project" else "property",
                            "requested_name": requested_name,
                            "resolved_name": str(
                                matches[0].get("name") or requested_name
                            ).strip(),
                        }
                except Exception:
                    pass

            if planned_type == "community":
                try:
                    matches = (
                        self.structured_retriever.search_community(
                            requested_name
                        ) or []
                    )
                    if matches:
                        return {
                            "entity_type": "community",
                            "requested_name": requested_name,
                            "resolved_name": str(
                                matches[0].get("name") or requested_name
                            ).strip(),
                        }
                except Exception:
                    pass

            # Generic entity probing. Order matters because project names are
            # often also described using location words.
            try:
                project_matches = (
                    self.structured_retriever.search_projects_by_name(
                        requested_name
                    ) or []
                )
                if project_matches:
                    return {
                        "entity_type": "project",
                        "requested_name": requested_name,
                        "resolved_name": str(
                            project_matches[0].get("name") or requested_name
                        ).strip(),
                    }
            except Exception:
                pass

            try:
                community_matches = (
                    self.structured_retriever.search_community(
                        requested_name
                    ) or []
                )
                if community_matches:
                    return {
                        "entity_type": "community",
                        "requested_name": requested_name,
                        "resolved_name": str(
                            community_matches[0].get("name") or requested_name
                        ).strip(),
                    }
            except Exception:
                pass

        # ------------------------------------------------------------
        # 4. If the planner supplied a type but DB lookup could not verify
        #    it, keep the candidate as a typed entity only when the type is
        #    unambiguous. Retrieval will then determine whether data exists.
        # ------------------------------------------------------------
        if planned_type in valid_types:
            return {
                "entity_type": planned_type,
                "requested_name": requested_name,
                "resolved_name": self._normalize_entity_name(
                    planned_type,
                    requested_name,
                ),
            }

        return None

    def _resolve_comparison_entities(
        self,
        question: str,
        query_plan=None,
    ) -> List[Dict[str, Any]]:
        """Resolve every explicit comparison candidate independently."""
        if query_plan is not None and not getattr(
            query_plan, "needs_comparison", False
        ):
            return []

        candidates = self._extract_comparison_candidates(
            question=question,
            query_plan=query_plan,
        )

        if len(candidates) < 2:
            return []

        planned_type = str(
            getattr(query_plan, "entity_type", "") or ""
        ).strip().lower() if query_plan is not None else ""

        resolved = []
        seen = set()

        for candidate in candidates:
            entity = self._resolve_one_comparison_entity(
                candidate=candidate,
                planned_entity_type=planned_type,
            )
            if not entity:
                print(
                    "Comparison entity could not be resolved:",
                    candidate,
                )
                continue

            entity_type = entity["entity_type"]
            resolved_name = entity["resolved_name"]
            key = (entity_type, str(resolved_name).lower())
            if key in seen:
                continue

            seen.add(key)
            resolved.append(entity)

        return resolved

    # ============================================================
    # GENERIC COMPARISON RETRIEVAL
    # ============================================================

    def _retrieve_single_entity_for_comparison(
        self,
        entity_type: str,
        entity_name: str,
    ) -> Dict[str, Any]:
        """Dispatch one comparison entity to the appropriate retriever."""
        entity_type = str(entity_type or "").strip().lower()
        entity_name = str(entity_name or "").strip()

        if not entity_name:
            return self._empty_structured_result()

        if entity_type == "community":
            return self._retrieve_by_community(entity_name)

        if entity_type == "project":
            return self._retrieve_by_project(entity_name)

        # ACOT currently has no separate property/listing table;
        # project records are the structured source for property-style data.
        if entity_type == "property":
            return self._retrieve_by_project(entity_name)

        if entity_type == "city":
            return self._retrieve_by_city(entity_name)

        if entity_type == "developer":
            return self._retrieve_by_developer(entity_name)

        if entity_type == "sub_community":
            return self._retrieve_by_sub_community(entity_name)

        return self._empty_structured_result()

    def _retrieve_comparison_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Retrieve multiple explicit entities independently and preserve ownership."""
        comparison_entities = []
        merged_communities = []
        merged_projects = []
        merged_sub_communities = []

        seen_communities = set()
        seen_projects = set()
        seen_sub_communities = set()

        for entity in entities:
            entity_type = str(entity.get("entity_type") or "").strip().lower()
            requested_name = str(entity.get("requested_name") or "").strip()
            resolved_name = str(entity.get("resolved_name") or requested_name).strip()

            if not resolved_name:
                continue

            print(
                "\nComparison entity:",
                entity_type,
                "| requested:", requested_name,
                "| resolved:", resolved_name,
            )

            data = self._retrieve_single_entity_for_comparison(
                entity_type=entity_type,
                entity_name=resolved_name,
            )
            data = self._normalize_structured_result(data)

            comparison_entities.append({
                "entity_type": entity_type,
                "requested_name": requested_name,
                "resolved_name": resolved_name,
                "data": data,
            })

            for community in data.get("communities", []) or []:
                name = str(community.get("name") or "").strip()
                key = name.lower()
                if name and key not in seen_communities:
                    merged_communities.append(community)
                    seen_communities.add(key)

            for project in data.get("projects", []) or []:
                project_id = project.get("id")
                name = str(project.get("name") or "").strip()
                key = str(project_id) if project_id is not None else name.lower()
                if key and key not in seen_projects:
                    merged_projects.append(project)
                    seen_projects.add(key)

            for subcommunity in data.get("sub_communities", []) or []:
                sub_id = subcommunity.get("id")
                name = str(subcommunity.get("name") or "").strip()
                key = str(sub_id) if sub_id is not None else name.lower()
                if key and key not in seen_sub_communities:
                    merged_sub_communities.append(subcommunity)
                    seen_sub_communities.add(key)

        return {
            "comparison": True,
            "comparison_entities": comparison_entities,
            "communities": merged_communities,
            "projects": merged_projects,
            "sub_communities": merged_sub_communities,
        }

    # ============================================================
    # DEVELOPER RETRIEVAL
    # ============================================================

    def _retrieve_by_developer(
        self,
        developer_name: str,
    ) -> Dict[str, Any]:
        """Retrieve projects by developer name from the current schema."""
        if not self.structured_retriever:
            return self._empty_structured_result()

        try:
            supabase = self.structured_retriever.supabase
            response = (
                supabase
                .table("projects")
                .select("*")
                .ilike("developer_name", f"%{developer_name}%")
                .limit(1000)
                .execute()
            )

            return {
                "communities": [],
                "projects": response.data or [],
                "sub_communities": [],
            }
        except Exception as exc:
            print("Developer retrieval error:", exc)
            return self._empty_structured_result()

    # ============================================================
    # SUB-COMMUNITY RETRIEVAL
    # ============================================================

    def _retrieve_by_sub_community(
        self,
        sub_community_name: str,
    ) -> Dict[str, Any]:
        """Retrieve sub-community records from the current schema."""
        if not self.structured_retriever:
            return self._empty_structured_result()

        try:
            supabase = self.structured_retriever.supabase
            response = (
                supabase
                .table("sub_communities")
                .select("*")
                .ilike("name", f"%{sub_community_name}%")
                .limit(1000)
                .execute()
            )

            return {
                "communities": [],
                "projects": [],
                "sub_communities": response.data or [],
            }
        except Exception as exc:
            print("Sub-community retrieval error:", exc)
            return self._empty_structured_result()

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

            # Support normalized project records where bedroom information
            # is stored as:
            # {
            #     "bedrooms": {
            #         "min": 0,
            #         "max": 3
            #     }
            # }
            if bedroom_min is None or bedroom_max is None:
                bedrooms = project.get("bedrooms")

                if isinstance(bedrooms, dict):
                    if bedroom_min is None:
                        bedroom_min = bedrooms.get("min")

                    if bedroom_max is None:
                        bedroom_max = bedrooms.get("max")

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
    # STRUCTURED RETRIEVAL BY MULTIPLE PROJECTS
    # ============================================================

    def _retrieve_by_projects(
        self,
        project_names: List[str],
    ) -> Dict[str, Any]:
        """
        Retrieve multiple explicit projects while preserving the order
        supplied by conversational memory.
        """
        merged_projects = []
        seen_names = set()
        communities = []
        sub_communities = []
        seen_community_names = set()
        seen_subcommunity_names = set()

        for project_name in project_names:
            if not project_name:
                continue

            result = self._retrieve_by_project(
                str(project_name)
            )

            for project in result.get("projects", []) or []:
                name = str(project.get("name") or "").strip()
                key = name.lower()

                if name and key not in seen_names:
                    merged_projects.append(project)
                    seen_names.add(key)

            for community in result.get("communities", []) or []:
                name = str(community.get("name") or "").strip()
                key = name.lower()

                if name and key not in seen_community_names:
                    communities.append(community)
                    seen_community_names.add(key)

            for subcommunity in result.get("sub_communities", []) or []:
                name = str(subcommunity.get("name") or "").strip()
                key = name.lower()

                if name and key not in seen_subcommunity_names:
                    sub_communities.append(subcommunity)
                    seen_subcommunity_names.add(key)

        return {
            "communities": communities,
            "projects": merged_projects,
            "sub_communities": sub_communities,
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
            "comparison": bool(result.get("comparison", False)),
            "comparison_entities": (
                result.get("comparison_entities", []) or []
            ),
            # Evidence retained when a conversational filter produces
            # zero matches. This lets ACOT explain why nothing matched
            # instead of incorrectly saying that no project data exists.
            "filter_no_match": bool(
                result.get("filter_no_match", False)
            ),
            "filter_evidence_projects": (
                result.get("filter_evidence_projects", []) or []
            ),
        }

    @staticmethod
    def _requested_data_scope(
        question: str,
        query_plan=None,
    ) -> Optional[List[str]]:
        """
        Work out which structured entity the user actually asked for.

        The retriever can return related data internally, but the final
        result should stay focused on what the user asked for.

        Examples:
            "show me communities" -> ["communities"]
            "what projects are available there" -> ["projects"]

        If the question clearly asks for more than one entity, keep both.
        If the wording is too general, return None and keep the existing
        result.
        """

        q = (question or "").lower()

        # These phrases normally mean the user wants project/listing data,
        # not the community record itself.
        asks_projects = bool(
            re.search(r"\bprojects?\b", q)
        )

        asks_communities = bool(
            re.search(r"\bcommunities?\b", q)
        )

        asks_sub_communities = bool(
            re.search(
                r"\bsub[- ]communities?\b",
                q,
            )
        )

        asks_properties = bool(
            re.search(r"\bproperties?\b", q)
        )

        scopes: List[str] = []

        if asks_communities:
            scopes.append("communities")

        if asks_sub_communities:
            scopes.append("sub_communities")

        if asks_projects:
            scopes.append("projects")

        # There is no property/listing table in the current ACOT schema.
        # Keep project data for property-style questions because projects
        # are the closest available structured source.
        if asks_properties and not scopes:
            scopes.append("projects")

        if scopes:
            return scopes

        # QueryPlan is only a fallback. We do not depend on it when the
        # user's wording already tells us the requested entity.
        if query_plan is not None:
            entity_type = str(
                getattr(query_plan, "entity_type", "") or ""
            ).lower().strip()

            mapping = {
                "community": ["communities"],
                "communities": ["communities"],
                "sub_community": ["sub_communities"],
                "sub_communities": ["sub_communities"],
                "project": ["projects"],
                "projects": ["projects"],
                "property": ["projects"],
                "properties": ["projects"],
            }

            return mapping.get(entity_type)

        return None

    @staticmethod
    def _apply_data_scope(
        structured_result: Dict[str, Any],
        question: str,
        query_plan=None,
    ) -> Dict[str, Any]:
        """
        Keep only the structured collections needed for the question.

        Retrieval methods may fetch parent/related records to resolve
        hierarchy. Those records should not automatically become part
        of the answer context.
        """

        scope = HybridRetriever._requested_data_scope(
            question,
            query_plan=query_plan,
        )

        if not scope:
            return structured_result

        allowed = set(scope)

        # Preserve retrieval metadata even when data-scope filtering
        # removes related collections. In particular, conversational
        # zero-match filters need their original candidate projects as
        # evidence for the final answer.
        return {
            "communities": (
                structured_result.get("communities", [])
                if "communities" in allowed
                else []
            ),
            "projects": (
                structured_result.get("projects", [])
                if "projects" in allowed
                else []
            ),
            "sub_communities": (
                structured_result.get("sub_communities", [])
                if "sub_communities" in allowed
                else []
            ),
            "comparison": bool(
                structured_result.get("comparison", False)
            ),
            "comparison_entities": (
                structured_result.get("comparison_entities", []) or []
            ),
            "filter_no_match": bool(
                structured_result.get("filter_no_match", False)
            ),
            "filter_evidence_projects": (
                structured_result.get(
                    "filter_evidence_projects",
                    []
                )
                or []
            ),
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
        conversation_context: Optional[Dict[str, Any]] = None,
        query_plan=None,
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
            # MULTI-PROJECT COMPARISON / INVESTMENT FOLLOW-UP
            # ----------------------------------------------------
            active_candidates = (
                (conversation_context or {}).get(
                    "active_candidates"
                )
                or []
            )

            candidate_names = [
                str(item.get("name")).strip()
                for item in active_candidates
                if isinstance(item, dict)
                and item.get("name")
            ]

            question_lower = (question or "").lower()

            referenced_candidates = [
                name
                for name in candidate_names
                if name.lower() in question_lower
            ]

            # --------------------------------------------------------
            # Detect references to the previous candidate list
            # --------------------------------------------------------

            # ConversationMemory stores the verified candidate list from
            # the previous turn. A follow-up can refer to that list without
            # repeating any project name.
            #
            # Examples:
            #   "Which of these have 2 bedroom options?"
            #   "What are their starting prices?"
            #   "Which ones have the lowest price?"
            #   "Show me those projects"
            #
            # In these cases the retriever must use ALL previous candidates
            # in the exact order stored by ConversationMemory.
            candidate_list_reference = any(
    phrase in question_lower
    for phrase in (
        "these projects",
        "those projects",
        "these properties",
        "those properties",
        "these project",
        "those project",
        "these property",
        "those property",

        # IMPORTANT: these survive ConversationMemory rewriting
        "which of the projects",
        "which of the properties",
        "which projects",
        "which properties",
        "what are the projects",
        "what are the properties",

        "which of these",
        "which ones",
        "which one",
        "their",
        "them",
    )
)

            # Explicit project names in the current question.
            explicit_multi_project_reference = (
                len(referenced_candidates) >= 2
            )

            # Normal comparison / ranking / investment questions.
            comparison_reference = (
                (
                    query_plan
                    and (
                        getattr(
                            query_plan,
                            "needs_comparison",
                            False,
                        )
                        or getattr(
                            query_plan,
                            "needs_ranking",
                            False,
                        )
                    )
                )
                or any(
                    term in question_lower
                    for term in (
                        "compare",
                        "comparison",
                        "versus",
                        " vs ",
                        "better",
                        "invest",
                        "investment",
                        "investing",
                    )
                )
            )

            wants_multi_project = (
                bool(candidate_names)
                and (
                    candidate_list_reference
                    or explicit_multi_project_reference
                    or comparison_reference
                )
            )

            # ----------------------------------------------------
            # EXPLICIT MULTI-ENTITY COMPARISON
            # ----------------------------------------------------
            comparison_entities = self._resolve_comparison_entities(
                question=question,
                query_plan=query_plan,
            )

            explicit_multi_entity_comparison = (
                len(comparison_entities) >= 2
            )

            if explicit_multi_entity_comparison:

                print("\n================================")
                print("MULTI-ENTITY COMPARISON")
                print("================================")
                print("Entities:", comparison_entities)

                result = self._retrieve_comparison_entities(
                    comparison_entities
                )

            elif wants_multi_project:

                # ----------------------------------------------------
                # Previous candidate-list follow-up
                # ----------------------------------------------------
                #
                # If the user says "these", "which ones", "their", etc.,
                # retrieve the complete previous candidate list.
                #
                # Do NOT depend on the rewritten question containing the
                # project names. ConversationMemory may rewrite:
                #
                #   "Which of these have 2 bedroom options?"
                #
                # into:
                #
                #   "Which of the projects in Jumeirah Village Circle
                #    have 2 bedroom options?"
                #
                # The rewritten question intentionally does not contain
                # Serenz/Samana Waves/etc., so the active candidate list
                # must be used directly.
                if candidate_list_reference:
                    ordered_names = list(candidate_names)

                # ----------------------------------------------------
                # Explicit multi-project reference
                # ----------------------------------------------------
                #
                # Example:
                #   "Compare Serenz and Azizi Ruby"
                #
                # Preserve the original candidate order rather than
                # changing the order based on price/name/etc.
                elif explicit_multi_project_reference:
                    ordered_names = [
                        name
                        for name in candidate_names
                        if name in referenced_candidates
                    ]

                # ----------------------------------------------------
                # Comparison/ranking/investment fallback
                # ----------------------------------------------------
                #
                # If the planner identifies a comparison/ranking query
                # and there is an active candidate list, use that list.
                else:
                    ordered_names = list(candidate_names)

                result = self._retrieve_by_projects(
                    ordered_names
                )

            # ----------------------------------------------------
            # PROJECT
            # ----------------------------------------------------

            elif (
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

            original_projects = list(
                result.get("projects", []) or []
            )

            filtered_projects = self._apply_project_filters(
                original_projects,
                filters,
            )

            # For a conversational candidate-list follow-up, keep the
            # original projects as evidence when the requested filter
            # matches none of them. The actual `projects` collection stays
            # filtered (empty), so downstream data is never presented as a
            # false match.
            filter_no_match = (
                bool(original_projects)
                and not filtered_projects
                and bool(filters)
                and wants_multi_project
                and not explicit_multi_entity_comparison
            )

            result["projects"] = filtered_projects
            result["filter_no_match"] = filter_no_match
            result["filter_evidence_projects"] = (
                original_projects if filter_no_match else []
            )

            # A community lookup may internally fetch its projects and
            # sub-communities so the hierarchy is available. Do not pass
            # all of that related data to the answer layer when the user
            # asked for only one entity type.
            result = self._apply_data_scope(
                result,
                question,
                query_plan=query_plan,
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

            # Fast-path for structured-only follow-up questions.
            # If the user asks for database fields such as price, bedrooms,
            # projects, developers, handover, etc., do not call pgvector
            # unless the question explicitly asks for document/brochure data.
            # This prevents unnecessary Supabase document RPC calls for
            # questions such as: "Which of these have 2 bedroom options?"
            if (
                self._needs_structured_data_from_question(question)
                and not self._document_terms(question)
                and str(query_plan.intent or "").lower()
                not in {"document"}
            ):
                needs_documents = False

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
                    conversation_context=
                        conversation_context,
                    query_plan=query_plan,
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

            "filter_no_match":
                bool(
                    structured_result.get(
                        "filter_no_match",
                        False
                    )
                ) if needs_structured else False,

            "filter_evidence_projects":
                (
                    structured_result.get(
                        "filter_evidence_projects",
                        []
                    )
                    or []
                ) if needs_structured else [],
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

            # Filter diagnostics/evidence for conversational follow-ups.
            "filter_no_match":
                structured_data.get(
                    "filter_no_match",
                    False
                ),

            "filter_evidence_projects":
                structured_data.get(
                    "filter_evidence_projects",
                    []
                ),

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