# app/intelligence/ai_analysis_engine.py

import json
import re
from typing import Any, Dict, List, Optional


class AIAnalysisEngine:
    """
    ACOT AI Analysis Engine.

    Responsibilities:
    - Analyze retrieved ACOT evidence
    - Produce data description
    - Identify positive factors
    - Identify considerations / limitations
    - Generate an AI solution
    - Generate an evidence-based ACOT score

    IMPORTANT:
    The engine does not invent missing database values.
    """

    def __init__(self, llm=None):
        self.llm = llm

    # =========================================================
    # PUBLIC METHOD
    # =========================================================

    async def analyze(
        self,
        question: str,
        structured_summary: Optional[Dict[str, Any]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        question_type: str = "analysis",
    ) -> Dict[str, Any]:

        structured_summary = structured_summary or {}
        documents = documents or []

        data = self._prepare_data(
            structured_summary
        )

        score = self._calculate_acot_score(
            question=question,
            data=data,
            documents=documents,
        )

        data_description = self._build_data_description(
            data=data
        )

        recommendation = self._build_recommendation(
            question=question,
            data=data,
            score=score,
            documents=documents,
        )

        # -----------------------------------------------------
        # LLM explanation
        # -----------------------------------------------------

        # Investment/search already have deterministic recommendations.
        # Only spend an extra LLM call for comparison/ranking narrative.
        needs_llm = question_type in {
            "comparison",
            "ranking",
            "analysis",
        } and bool(
            data.get("projects")
            or data.get("communities")
            or data.get("sub_communities")
        )

        llm_analysis = None
        if needs_llm:
            llm_analysis = await self._generate_llm_analysis(
                question=question,
                data=data,
                documents=documents,
                score=score,
            )

        if llm_analysis:

            data_description = self._merge_llm_data_description(
                data_description,
                llm_analysis,
            )

            recommendation = self._merge_llm_recommendation(
                recommendation,
                llm_analysis,
            )

        return {
            "data": data,

            "data_description": data_description,

            "acot_recommendation": recommendation,

            "acot_score": score,
        }

    # =========================================================
    # PREPARE DATA
    # =========================================================

    @staticmethod
    def _slim_project(project: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only fields needed for analysis prompts (cuts token burn)."""

        bedrooms = project.get("bedrooms")
        if not isinstance(bedrooms, dict):
            bedrooms = {
                "min": project.get("bedroom_min"),
                "max": project.get("bedroom_max"),
            }
        amenities = project.get("amenities") or []
        if isinstance(amenities, list):
            amenities = [
                str(item).strip()
                for item in amenities[:8]
                if str(item).strip()
            ]
        else:
            amenities = []

        return {
            "name": project.get("name"),
            "developer": (
                project.get("developer")
                or project.get("developer_name")
            ),
            "community": project.get("community"),
            "city": project.get("city"),
            "price": project.get("price"),
            "bedrooms": bedrooms,
            "status": project.get("status"),
            "project_status": project.get("project_status"),
            "handover_time": project.get("handover_time"),
            "property_types": (project.get("property_types") or [])[:6],
            "amenities": amenities,
        }

    @staticmethod
    def _slim_named(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": record.get("name"),
            "city": record.get("city"),
            "community": record.get("community"),
        }

    def _prepare_data(
        self,
        structured_summary: Dict[str, Any],
    ) -> Dict[str, Any]:

        communities = (
            structured_summary.get(
                "community",
                structured_summary.get(
                    "communities",
                    [],
                ),
            )
            or []
        )

        projects = (
            structured_summary.get(
                "projects",
                [],
            )
            or []
        )

        sub_communities = (
            structured_summary.get(
                "sub_communities",
                [],
            )
            or []
        )

        return {
            "communities": [
                self._slim_named(item)
                for item in communities[:8]
                if isinstance(item, dict)
            ],
            "projects": [
                self._slim_project(item)
                for item in projects[:8]
                if isinstance(item, dict)
            ],
            "sub_communities": [
                self._slim_named(item)
                for item in sub_communities[:8]
                if isinstance(item, dict)
            ],
        }

    # =========================================================
    # DATA DESCRIPTION
    # =========================================================

    def _build_data_description(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:

        communities = data.get(
            "communities",
            [],
        )

        projects = data.get(
            "projects",
            [],
        )

        sub_communities = data.get(
            "sub_communities",
            [],
        )

        key_facts = []

        for community in communities:

            name = community.get("name")

            if name:
                key_facts.append(
                    f"Community: {name}"
                )

            city = community.get("city")

            if city:
                key_facts.append(
                    f"City: {city}"
                )

            inventory = community.get(
                "inventory",
                {},
            )

            if not inventory:

                inventory = {
                    "projects_count":
                        community.get(
                            "projects_count"
                        ),
                    "pool_projects_count":
                        community.get(
                            "pool_projects_count"
                        ),
                    "total_count":
                        community.get(
                            "total_count"
                        ),
                }

            if inventory.get(
                "projects_count"
            ) is not None:

                key_facts.append(
                    "Projects: "
                    + str(
                        inventory.get(
                            "projects_count"
                        )
                    )
                )

        for project in projects:

            name = project.get("name")

            if name:
                key_facts.append(
                    f"Project: {name}"
                )

            developer = (
                project.get(
                    "developer"
                )
                or project.get(
                    "developer_name"
                )
            )

            if developer:
                key_facts.append(
                    f"Developer: {developer}"
                )

            price = project.get("price")

            if price is not None:
                key_facts.append(
                    "Starting price: "
                    f"AED {price}"
                )

            bedrooms = project.get(
                "bedrooms"
            )

            if isinstance(
                bedrooms,
                dict,
            ):

                bedroom_label = (
                    bedrooms.get(
                        "label"
                    )
                )

                if bedroom_label:
                    key_facts.append(
                        "Bedrooms: "
                        f"{bedroom_label}"
                    )

        summary = (
            f"Retrieved "
            f"{len(communities)} "
            f"community record(s), "
            f"{len(projects)} "
            f"project record(s), and "
            f"{len(sub_communities)} "
            f"sub-community record(s)."
        )

        return {
            "summary": summary,
            "key_facts": key_facts,
            "comparison": self._build_comparison(
                communities,
                projects,
            ),
        }

    # =========================================================
    # COMPARISON DATA
    # =========================================================

    def _build_comparison(
        self,
        communities: List[Dict[str, Any]],
        projects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        comparison = []

        # -----------------------------------------------------
        # Community comparison
        # -----------------------------------------------------

        for community in communities:

            inventory = community.get(
                "inventory",
                {},
            )

            if not inventory:

                inventory = {
                    "projects_count":
                        community.get(
                            "projects_count"
                        ),
                    "pool_projects_count":
                        community.get(
                            "pool_projects_count"
                        ),
                    "total_count":
                        community.get(
                            "total_count"
                        ),
                }

            comparison.append(
                {
                    "entity_type": "community",
                    "name": community.get(
                        "name"
                    ),
                    "city": community.get(
                        "city"
                    ),
                    "projects_count":
                        inventory.get(
                            "projects_count"
                        ),
                    "pool_projects_count":
                        inventory.get(
                            "pool_projects_count"
                        ),
                    "total_inventory":
                        inventory.get(
                            "total_count"
                        ),
                }
            )

        # -----------------------------------------------------
        # Project comparison
        # -----------------------------------------------------

        for project in projects:

            bedrooms = project.get(
                "bedrooms",
                {},
            )

            if not isinstance(
                bedrooms,
                dict,
            ):
                bedrooms = {}

            comparison.append(
                {
                    "entity_type": "project",
                    "name": project.get(
                        "name"
                    ),
                    "developer": (
                        project.get(
                            "developer"
                        )
                        or project.get(
                            "developer_name"
                        )
                    ),
                    "community": project.get(
                        "community"
                    ),
                    "price": project.get(
                        "price"
                    ),
                    "bedrooms":
                        bedrooms.get(
                            "label"
                        ),
                    "property_types":
                        project.get(
                            "property_types",
                            [],
                        ),
                    "handover_time":
                        project.get(
                            "handover_time"
                        ),
                }
            )

        return comparison

    # =========================================================
    # ACOT SCORE
    # =========================================================

    def _calculate_acot_score(
        self,
        question: str,
        data: Dict[str, Any],
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        communities = data.get(
            "communities",
            []
        )

        projects = data.get(
            "projects",
            []
        )

        # -----------------------------------------------------
        # Evidence completeness
        # -----------------------------------------------------

        evidence_fields = 0
        available_fields = 0

        for community in communities:

            fields = (
                "name",
                "city",
                "knowledge_text",
                "source",
            )

            for field in fields:

                evidence_fields += 1

                if community.get(field):
                    available_fields += 1

        for project in projects:

            fields = (
                "name",
                "community",
                "price",
                "bedroom_min",
                "bedroom_max",
                "developer_name",
                "source",
            )

            for field in fields:

                evidence_fields += 1

                if project.get(field) is not None:
                    available_fields += 1

        if evidence_fields:

            evidence_completeness = round(
                (
                    available_fields
                    / evidence_fields
                ) * 100
            )

        else:

            evidence_completeness = 0

        # -----------------------------------------------------
        # Entity coverage
        # -----------------------------------------------------

        entity_count = (
            len(communities)
            + len(projects)
        )

        if entity_count >= 2:
            entity_coverage = 100

        elif entity_count == 1:
            entity_coverage = 60

        else:
            entity_coverage = 0

        # -----------------------------------------------------
        # Query fit
        # -----------------------------------------------------

        q = (
            question or ""
        ).lower()

        query_fit = 0

        if projects and any(
            word in q
            for word in (
                "project",
                "projects",
                "property",
                "properties",
                "price",
                "bedroom",
                "developer",
            )
        ):
            query_fit = 100

        elif communities and any(
            word in q
            for word in (
                "community",
                "communities",
                "compare",
            )
        ):
            query_fit = 100

        elif entity_count:
            query_fit = 70

        # -----------------------------------------------------
        # Source diversity
        # -----------------------------------------------------

        sources = set()

        for community in communities:

            source = community.get(
                "source"
            )

            if source:
                sources.add(
                    str(source)
                )

        for project in projects:

            source = project.get(
                "source"
            )

            if source:
                sources.add(
                    str(source)
                )

        for document in documents:

            source = (
                document.get(
                    "document_url"
                )
                or document.get(
                    "url"
                )
            )

            if source:
                sources.add(
                    str(source)
                )

        if len(sources) >= 2:
            source_diversity = 100

        elif len(sources) == 1:
            source_diversity = 60

        else:
            source_diversity = 0

        # -----------------------------------------------------
        # Weighted score
        # -----------------------------------------------------

        score = round(
            (
                evidence_completeness * 0.40
                + entity_coverage * 0.30
                + query_fit * 0.20
                + source_diversity * 0.10
            )
        )

        # Always return a numeric ACOT score.
        # The score measures evidence coverage/quality,
        # not investment quality or market performance.
        score = max(5, min(score, 100))

        return score

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    def _build_recommendation(
        self,
        question: str,
        data: Dict[str, Any],
        score: Dict[str, Any],
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        documents = documents or []

        projects = data.get(
            "projects",
            []
        )

        communities = data.get(
            "communities",
            []
        )

        positive_factors = []
        considerations = []

        # -----------------------------------------------------
        # Positive project factors
        # -----------------------------------------------------

        for project in projects:

            if project.get("name"):
                positive_factors.append(
                    f"Retrieved project: "
                    f"{project.get('name')}"
                )

            if project.get(
                "developer"
            ) or project.get(
                "developer_name"
            ):

                developer = (
                    project.get(
                        "developer"
                    )
                    or project.get(
                        "developer_name"
                    )
                )

                positive_factors.append(
                    f"Developer information "
                    f"is available: {developer}"
                )

            if project.get(
                "price"
            ) is not None:

                positive_factors.append(
                    "Starting price is "
                    "available in the "
                    "structured data."
                )

            bedrooms = project.get(
                "bedrooms"
            )

            if isinstance(
                bedrooms,
                dict,
            ) and bedrooms.get(
                "label"
            ):

                positive_factors.append(
                    "Bedroom configuration: "
                    f"{bedrooms.get('label')}"
                )

            if project.get(
                "handover_time"
            ):

                positive_factors.append(
                    "Handover information "
                    "is available."
                )

        # -----------------------------------------------------
        # Community factors
        # -----------------------------------------------------

        for community in communities:

            if community.get(
                "name"
            ):

                positive_factors.append(
                    f"Community evidence is "
                    f"available for "
                    f"{community.get('name')}."
                )

        # -----------------------------------------------------
        # Missing evidence
        # -----------------------------------------------------

        if not projects and not communities:

            considerations.append(
                "No structured ACOT evidence "
                "was retrieved for this analysis."
            )

        if not documents:

            considerations.append(
                "No supporting document "
                "evidence was retrieved."
            )

        if not positive_factors:

            positive_factors.append(
                "No specific positive factor "
                "can be established from the "
                "available evidence."
            )

        # -----------------------------------------------------
        # Confidence + deterministic AI solution fallback
        # -----------------------------------------------------

        if score >= 80:
            confidence = "High"
            recommendation_summary = (
                "Strong relevant evidence was retrieved for this analysis."
            )
        elif score >= 60:
            confidence = "Medium"
            recommendation_summary = (
                "A good amount of relevant evidence was retrieved, "
                "although some information may still be incomplete."
            )
        elif score >= 40:
            confidence = "Low"
            recommendation_summary = (
                "Some relevant evidence was retrieved, but the "
                "available information is incomplete."
            )
        else:
            confidence = "Low"
            recommendation_summary = (
                "Very limited relevant evidence was retrieved. "
                "The result should be treated as preliminary."
            )

        if projects:
            project_names = [p.get("name") for p in projects if p.get("name")]
            if len(project_names) == 1:
                ai_solution = (
                    f"The retrieved evidence contains the project {project_names[0]}. "
                    "Use the available project attributes such as price, bedrooms, "
                    "developer, property type, status, and handover to address the "
                    "user's request. Missing attributes should be verified from "
                    "additional ACOT evidence."
                )
            else:
                ai_solution = (
                    f"The retrieved evidence contains {len(project_names)} projects. "
                    "Use the available project attributes to address the user's "
                    "request and compare only fields supported by the retrieved evidence."
                )
        elif communities:
            ai_solution = (
                f"The retrieved evidence contains {len(communities)} community record(s). "
                "Use the available community inventory and location information to "
                "address the request. Retrieve project-level or document evidence "
                "when additional detail is required."
            )
        else:
            ai_solution = (
                "Very limited relevant ACOT evidence was found. Refine the entity or "
                "query and retrieve additional structured records or supporting "
                "documents before making a detailed assessment."
            )

        return {
            "summary": recommendation_summary,
            "positive_factors": positive_factors,
            "considerations": considerations,
            "ai_solution": ai_solution,
            "confidence": confidence,
        }

    # =========================================================
    # LLM ANALYSIS
    # =========================================================

    # Document chunks are stored up to ~1200 chars. The analysis prompt
    # keeps that cap (so it sees the same substantive facts the answer
    # chain sees) but drops stored ids/urls and never overruns the cap,
    # which otherwise inflated this secondary call's input tokens.
    MAX_ANALYSIS_DOC_CHARS = 1200

    @classmethod
    def _slim_documents(
        cls,
        documents: List[Any],
    ) -> List[Dict[str, Any]]:
        """Keep only the analysis-relevant fields of document cards."""

        slim: List[Dict[str, Any]] = []

        for item in documents or []:
            if not isinstance(item, dict):
                continue

            content = str(
                item.get("content")
                or ""
            ).strip()

            if not content:
                continue

            if len(content) > cls.MAX_ANALYSIS_DOC_CHARS:
                content = (
                    content[: cls.MAX_ANALYSIS_DOC_CHARS]
                    + "..."
                )

            slim.append(
                {
                    "title": (
                        item.get("document_title")
                        or item.get("title")
                    ),
                    "type": (
                        item.get("document_type")
                        or item.get("type")
                    ),
                    "publisher": item.get("publisher"),
                    "similarity": item.get("similarity"),
                    "content": content,
                }
            )

        return slim

    async def _generate_llm_analysis(
        self,
        question: str,
        data: Dict[str, Any],
        documents: List[Dict[str, Any]],
        score: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        if self.llm is None:
            return None

        structured_json = json.dumps(
            data,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        documents_json = json.dumps(
            self._slim_documents(documents),
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

        print(
            "ACOT LLM INPUT SIZE: "
            f"question_chars={len(question):,}, "
            f"structured_chars={len(structured_json):,}, "
            f"documents_chars={len(documents_json):,}"
        )

        prompt = f"""
You are the ACOT Real Estate AI Analysis Engine.

Analyze the user's question using ONLY the supplied ACOT evidence.

USER QUESTION:
{question}

STRUCTURED ACOT DATA:
{structured_json}

DOCUMENT EVIDENCE:
{documents_json}

ACOT SCORE:
{score}

RULES:
1. Never invent property, community, price, rental, ROI,
   developer, or market data.
2. If information is missing, explicitly say it is missing.
3. Distinguish retrieved facts from AI interpretation.
4. Provide useful positive factors supported by evidence.
5. Do not claim that something is a good investment merely
   because it exists in the data.
6. Do not create or modify the ACOT score.
7. The supplied ACOT score is an evidence-coverage score.
8. Keep the analysis concise and structured.
9. The ai_solution must directly address the user's question using only the supplied evidence.
10. If evidence is missing, explain what can and cannot be concluded and state what additional evidence is needed.
11. Identify the entity type of every retrieved entity.
12. A project and a community are different entities.
13. If the question compares a project with a community,
    explicitly state that they are different entity levels.
14. If one compared entity is a community, summarize the
    relevant projects retrieved for that community.
15. Do not omit retrieved entities when they are relevant
    to the comparison.
16. Do not treat a community as if it were a single project.
17. Do not make unsupported judgments such as "better",
    "best", "more attractive", or "better investment".
18. Base every comparison point on the supplied evidence.

Return ONLY valid JSON:

{{
  "summary": "...",
  "key_facts": [],
  "positive_factors": [],
  "considerations": [],
  "ai_solution": "...",
  "confidence": "High|Medium|Low"
}}
"""

        try:

            response = await self.llm.generate(
                prompt,
                max_output_tokens=500,
            )

            if not response:
                return None

            text = str(
                response
            ).strip()

            # Remove markdown JSON fences.
            text = re.sub(
                r"^```json\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except Exception as exc:

            print(
                "AI analysis generation error:",
                exc,
            )

        return None

    # =========================================================
    # MERGE LLM DATA DESCRIPTION
    # =========================================================

    def _merge_llm_data_description(
        self,
        base: Dict[str, Any],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = dict(
            base
        )

        if llm_result.get(
            "summary"
        ):

            result["summary"] = (
                llm_result.get(
                    "summary"
                )
            )

        if llm_result.get(
            "key_facts"
        ):

            result["key_facts"] = (
                llm_result.get(
                    "key_facts"
                )
            )

        return result

    # =========================================================
    # MERGE LLM RECOMMENDATION
    # =========================================================

    def _merge_llm_recommendation(
        self,
        base: Dict[str, Any],
        llm_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = dict(
            base
        )

        if llm_result.get(
            "positive_factors"
        ):

            result["positive_factors"] = (
                llm_result.get(
                    "positive_factors"
                )
            )

        if llm_result.get(
            "considerations"
        ):

            result["considerations"] = (
                llm_result.get(
                    "considerations"
                )
            )

        if llm_result.get(
            "ai_solution"
        ):

            result["ai_solution"] = (
                llm_result.get(
                    "ai_solution"
                )
            )

        if llm_result.get(
            "confidence"
        ):

            result["confidence"] = (
                llm_result.get(
                    "confidence"
                )
            )

        return result