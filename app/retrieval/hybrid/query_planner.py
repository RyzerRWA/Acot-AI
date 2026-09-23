"""
ACOT Query Planner
------------------

First stage of the ACOT intelligence pipeline.

Purpose:
    Convert a natural-language user question into a structured query plan
    BEFORE retrieval or LLM answer generation.

The planner uses an LLM for semantic understanding when available,
with a deterministic rule-based fallback so the application remains usable
when the LLM is unavailable.

The planner identifies:
    - user intent
    - requested operation(s)
    - likely entity type
    - comparison/ranking requirements
    - filters such as budget/bedrooms
    - whether documents, structured data, analytics, or prediction are needed

Entity resolution is deliberately left to the retrieval layer.  This keeps
the planner independent from the current database contents.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryPlan:
    # Main intent, e.g. search, information, investment, comparison, prediction.
    intent: str = "general"

    # Ordered operations the intelligence layer should perform.
    operations: List[str] = field(default_factory=list)

    # What the user is primarily asking about.
    entity_type: Optional[str] = None

    # Candidate entity phrases. These are NOT treated as verified entities.
    entity_candidates: List[str] = field(default_factory=list)

    # User constraints extracted from the question.
    filters: Dict[str, Any] = field(default_factory=dict)

    # Data sources / capabilities required.
    needs_structured_data: bool = False
    needs_documents: bool = False
    needs_analytics: bool = False
    needs_prediction: bool = False

    # Whether ranking/comparison is explicitly requested.
    needs_ranking: bool = False
    needs_comparison: bool = False

    # Whether this question depends on previous conversation turns.
    needs_conversation_context: bool = False

    # Human-readable reason for debugging/observability.
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryPlanner:
    """
    Natural-language query planner for ACOT.

    This is the semantic planning layer, not the final answer generator.
    The planner decides what the user wants and which capabilities are needed;
    retrieval and the final answer chain still provide the actual evidence and response.
    """

    VALID_INTENTS = {
        "investment",
        "prediction",
        "comparison",
        "ranking",
        "calculation",
        "document",
        "search",
        "information",
        "general",
    }

    VALID_ENTITY_TYPES = {
        "city",
        "community",
        "sub_community",
        "project",
        "property",
        "developer",
        None,
    }

    def __init__(self, llm_client=None, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm_client = llm_client
        self._llm_error = None

        if self.use_llm and self._llm_client is None:
            try:
                from app.llm.client import GeminiClient
                self._llm_client = GeminiClient()
            except Exception as exc:
                self._llm_client = None
                self._llm_error = str(exc)

    INVESTMENT_TERMS = (
        "invest",
        "investment",
        "investing",
        "roi",
        "return",
        "rental yield",
        "yield",
        "capital appreciation",
        "appreciation",
        "profit",
        "profitable",
        "worth buying",
        "good buy",
        "best investment",
    )

    PREDICTION_TERMS = (
        "predict",
        "prediction",
        "forecast",
        "future",
        "next year",
        "next month",
        "will prices",
        "will price",
        "expected price",
        "expected rent",
        "outlook",
    )

    COMPARISON_TERMS = (
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference between",
        "which is better",
        "better between",
        "choose between",
    )

    RANKING_TERMS = (
        "best",
        "top",
        "highest",
        "lowest",
        "cheapest",
        "most expensive",
        "recommend",
        "recommended",
        "best options",
        "top options",
    )

    SEARCH_TERMS = (
        "show me",
        "find",
        "list",
        "search",
        "available",
        "options",
        "properties",
        "projects",
    )

    DOCUMENT_TERMS = (
        "brochure",
        "document",
        "documents",
        "building code",
        "regulation",
        "regulations",
        "law",
        "laws",
        "requirements",
        "guideline",
        "guidelines",
        "security",
        "security features",
        "amenity",
        "amenities",
        "facilities",
        "interior",
        "interiors",
        "finishes",
        "kitchen",
        "appliances",
        "views",
        "floor plan",
        "floor plans",
        "layout",
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
    )

    STRUCTURED_TERMS = (
        "price",
        "prices",
        "cost",
        "rent",
        "rental",
        "bedroom",
        "bedrooms",
        "size",
        "area",
        "property",
        "properties",
        "project",
        "projects",
        "community",
        "communities",
        "developer",
        "developers",
        "handover",
        "status",
        "location",
        "district",
        "sub-community",
        "subcommunity",
    )

    CALCULATION_TERMS = (
        "calculate",
        "calculation",
        "how much",
        "how many",
        "percentage",
        "percent",
        "per sqft",
        "per square foot",
        "monthly payment",
        "mortgage",
        "afford",
        "budget",
        "yield",
        "roi",
    )

    INFORMATION_PATTERNS = (
        r"\btell me about\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwho is\b",
        r"\bgive me details\b",
        r"\bdetails about\b",
        r"\binformation about\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bhow does\b",
        r"\bwhat does\b",
    )

    FOLLOW_UP_PATTERNS = (
        r"^\s*(what about|how about|and|also|then what|what if)\b",
        r"^\s*(compare them|compare those|compare these|the first two|the second one|that one)\b",
        r"^\s*compare\s+\d+\s*(?:and|&|,)\s*\d+\b",
        r"\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
        r"^\s*(why|how much|what price|what rent|what about)\b",
    )

    def _rule_plan(
        self,
        question: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        q = (question or "").strip()
        q_lower = q.lower()

        if not q:
            return QueryPlan(
                intent="invalid",
                operations=["clarify"],
                reasoning=["The user question is empty."],
            )

        plan = QueryPlan()

        # ------------------------------------------------------------
        # Conversation dependency
        # ------------------------------------------------------------
        if conversation_context and self._looks_like_follow_up(q_lower):
            plan.needs_conversation_context = True
            plan.reasoning.append(
                "The question appears to depend on a previous conversation turn."
            )

        # ------------------------------------------------------------
        # Primary intent
        # ------------------------------------------------------------
        if self._contains_any(q_lower, self.PREDICTION_TERMS):
            plan.intent = "prediction"
            plan.operations.append("predict")
            plan.needs_prediction = True
            plan.needs_structured_data = True
            plan.needs_analytics = True
            plan.reasoning.append(
                "Prediction language was detected; historical/feature data and a prediction capability are required."
            )

        elif self._contains_any(q_lower, self.INVESTMENT_TERMS):
            plan.intent = "investment"
            plan.operations.append("investment_analysis")
            plan.needs_structured_data = True
            plan.needs_analytics = True
            plan.reasoning.append(
                "Investment language was detected; the answer must be based on investment evidence and analysis."
            )

        elif self._contains_any(q_lower, self.COMPARISON_TERMS):
            plan.intent = "comparison"
            plan.operations.append("compare")
            plan.needs_comparison = True
            plan.needs_structured_data = True
            plan.reasoning.append(
                "Comparison language was detected; multiple candidates and comparable attributes are required."
            )

        elif self._contains_any(q_lower, self.RANKING_TERMS):
            plan.intent = "ranking"
            plan.operations.append("rank")
            plan.needs_ranking = True
            plan.needs_structured_data = True
            plan.reasoning.append(
                "Ranking/recommendation language was detected; multiple candidates are required before claiming a best option."
            )

        elif self._contains_any(q_lower, self.CALCULATION_TERMS):
            plan.intent = "calculation"
            plan.operations.append("calculate")
            plan.needs_structured_data = True
            plan.needs_analytics = True
            plan.reasoning.append(
                "Calculation language was detected; values must come from retrieved data before computing."
            )

        elif self._contains_any(q_lower, self.DOCUMENT_TERMS):
            plan.intent = "document"
            plan.operations.append("retrieve_documents")
            plan.needs_documents = True
            plan.reasoning.append(
                "The question asks for document/brochure/feature knowledge; source documents are required."
            )

        elif self._contains_any(q_lower, self.SEARCH_TERMS):
            plan.intent = "search"
            plan.operations.append("search")
            plan.needs_structured_data = True
            plan.reasoning.append(
                "The question requests real-estate records/options."
            )

        elif any(re.search(pattern, q_lower) for pattern in self.INFORMATION_PATTERNS):
            plan.intent = "information"
            plan.operations.append("retrieve_information")
            plan.needs_structured_data = True
            plan.needs_documents = True
            plan.reasoning.append(
                "The question requests general information; both structured facts and relevant documents may be useful."
            )

        else:
            # General real-estate questions should still enter the intelligence
            # pipeline rather than falling directly into free-form generation.
            plan.intent = "general"
            plan.operations.append("retrieve_and_answer")
            plan.needs_structured_data = True
            plan.needs_documents = True
            plan.reasoning.append(
                "No narrow intent was detected; use grounded retrieval before generating an answer."
            )

        # ------------------------------------------------------------
        # Secondary operations
        # ------------------------------------------------------------
        # A user can ask for "best projects under 2M" or
        # "compare projects and tell me which is a better investment".
        if plan.intent != "prediction" and self._contains_any(
            q_lower, self.PREDICTION_TERMS
        ):
            plan.needs_prediction = True
            if "predict" not in plan.operations:
                plan.operations.append("predict")

        if plan.intent != "investment" and self._contains_any(
            q_lower, self.INVESTMENT_TERMS
        ):
            plan.needs_analytics = True
            if "investment_analysis" not in plan.operations:
                plan.operations.append("investment_analysis")

        if self._contains_any(q_lower, self.COMPARISON_TERMS):
            plan.needs_comparison = True
            if "compare" not in plan.operations:
                plan.operations.append("compare")

        if self._contains_any(q_lower, self.RANKING_TERMS):
            plan.needs_ranking = True
            if "rank" not in plan.operations:
                plan.operations.append("rank")

        # ------------------------------------------------------------
        # Entity type
        # ------------------------------------------------------------
        plan.entity_type = self._detect_entity_type(q_lower)

        if plan.entity_type:
            plan.reasoning.append(
                f"Likely entity type: {plan.entity_type}."
            )

        # ------------------------------------------------------------
        # Filters
        # ------------------------------------------------------------
        plan.filters.update(self._extract_filters(q))

        if plan.filters:
            plan.reasoning.append(
                f"User constraints extracted: {plan.filters}."
            )

        # ------------------------------------------------------------
        # Required sources based on entity/operation
        # ------------------------------------------------------------
        if plan.intent in {
            "search",
            "ranking",
            "comparison",
            "investment",
            "calculation",
            "prediction",
        }:
            plan.needs_structured_data = True

        if plan.intent in {
            "information",
            "document",
            "general",
        }:
            plan.needs_documents = True

        # Investment/prediction should NOT be treated as pure brochure RAG.
        if plan.needs_analytics or plan.needs_prediction:
            plan.needs_structured_data = True

        # Keep operations deterministic and readable.
        plan.operations = self._dedupe(plan.operations)

        return plan

    def plan(
        self,
        question: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        """
        Understand the user's request semantically, then fall back to the
        deterministic planner if the LLM is unavailable or returns invalid data.
        """
        rule_plan = self._rule_plan(
            question=question,
            conversation_context=conversation_context,
        )

        # ------------------------------------------------------------
        # FAST PATH: simple structured/database questions
        # ------------------------------------------------------------
        # These queries can be understood deterministically from the
        # rule-based planner. Avoid spending a Gemini call just to
        # classify a request that only needs PostgreSQL data.
        if self._is_simple_structured_plan(rule_plan):
            rule_plan.reasoning.append(
                "Deterministic structured-query fast path used; LLM planning skipped."
            )
            return rule_plan

        if not self.use_llm or self._llm_client is None:
            if self._llm_error:
                rule_plan.reasoning.append(
                    f"LLM planning unavailable; deterministic planning used: {self._llm_error}"
                )
            return rule_plan

        try:
            llm_plan = self._llm_plan(
                question=question,
                conversation_context=conversation_context,
            )
            if llm_plan is None:
                return rule_plan

            merged = self._merge_plans(rule_plan, llm_plan)
            merged.reasoning.insert(
                0,
                "LLM semantic understanding was used before retrieval."
            )
            return merged

        except Exception as exc:
            self._llm_error = str(exc)
            rule_plan.reasoning.append(
                f"LLM planning failed; deterministic planning used: {exc}"
            )
            return rule_plan

    @staticmethod
    def _is_simple_structured_plan(plan: QueryPlan) -> bool:
        """Return True for deterministic database-only requests.

        These requests should not consume an LLM planning call because the
        rule planner already extracts their intent/entity/filter requirements.
        Complex analytical, ranking, comparison, prediction, and document
        queries continue through the LLM semantic planner.
        """
        if plan is None:
            return False

        if plan.intent != "search":
            return False

        if not plan.needs_structured_data:
            return False

        if (
            plan.needs_documents
            or plan.needs_analytics
            or plan.needs_prediction
            or plan.needs_ranking
            or plan.needs_comparison
        ):
            return False

        return True

    def _llm_plan(
        self,
        question: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[QueryPlan]:
        history = conversation_context or {}

        prompt = f"""
You are the query-understanding layer of ACOT, an AI-powered Dubai real-estate intelligence system.
Your job is to understand the user's request and create a retrieval/action plan.
Do NOT answer the user. Do NOT invent database values.

The downstream system has these capabilities:
- structured Supabase PostgreSQL data: communities, sub_communities, projects
- document RAG through Supabase pgvector: document_chunks
- deterministic calculations/analytics over retrieved data
- investment analysis over retrieved evidence
- ranking and comparison over retrieved candidates
- prediction only when the required historical/feature data exists

Classify the request by meaning, not just keywords. A question can need multiple sources at once.
For example, asking for a project's price AND amenities needs both structured data and documents.

Conversation context:
{json.dumps(history, ensure_ascii=False)}

User question:
{question}

Return ONLY valid JSON with this schema:
{{
  "intent": "investment|prediction|comparison|ranking|calculation|document|search|information|general",
  "operations": ["..."],
  "entity_type": "city|community|sub_community|project|property|developer|null",
  "entity_candidates": ["..."],
  "needs_structured_data": true,
  "needs_documents": false,
  "needs_analytics": false,
  "needs_prediction": false,
  "needs_ranking": false,
  "needs_comparison": false,
  "needs_conversation_context": false,
  "reasoning": ["..."]
}}

Rules:
1. Prefer both structured and document sources when the question asks for both database facts and brochure/document details.
2. Never assume a 'property/listing' table exists; the current structured dataset exposes projects, communities and sub-communities.
3. If the user asks for rental performance, yield, occupancy, transactions or historical trends, mark analytics as needed, but do not claim those values exist.
4. If the question asks for the 'best', 'top' or 'recommend', mark ranking as needed.
5. If the user asks to compare things, mark comparison as needed.
6. Preserve follow-up meaning from the conversation context when provided.
7. entity_candidates should contain useful names only, not filler words.
"""

        response = self._llm_client.generate(prompt)
        data = self._extract_json(response)
        if not isinstance(data, dict):
            return None

        return self._plan_from_llm_dict(data)

    @staticmethod
    def _extract_json(response: Any) -> Optional[Dict[str, Any]]:
        if response is None:
            return None

        if isinstance(response, dict):
            return response

        text = str(response).strip()
        if not text:
            return None

        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _plan_from_llm_dict(self, data: Dict[str, Any]) -> QueryPlan:
        intent = str(data.get("intent", "general")).strip().lower()
        if intent not in self.VALID_INTENTS:
            intent = "general"

        entity_type = data.get("entity_type")
        if entity_type is not None:
            entity_type = str(entity_type).strip().lower() or None
        if entity_type not in self.VALID_ENTITY_TYPES:
            entity_type = None

        operations = data.get("operations") or []
        if isinstance(operations, str):
            operations = [operations]
        operations = [str(value).strip() for value in operations if str(value).strip()]

        candidates = data.get("entity_candidates") or []
        if isinstance(candidates, str):
            candidates = [candidates]
        candidates = [str(value).strip() for value in candidates if str(value).strip()]

        reasoning = data.get("reasoning") or []
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        reasoning = [str(value).strip() for value in reasoning if str(value).strip()]

        return QueryPlan(
            intent=intent,
            operations=self._dedupe(operations),
            entity_type=entity_type,
            entity_candidates=self._dedupe(candidates),
            needs_structured_data=bool(data.get("needs_structured_data")),
            needs_documents=bool(data.get("needs_documents")),
            needs_analytics=bool(data.get("needs_analytics")),
            needs_prediction=bool(data.get("needs_prediction")),
            needs_ranking=bool(data.get("needs_ranking")),
            needs_comparison=bool(data.get("needs_comparison")),
            needs_conversation_context=bool(data.get("needs_conversation_context")),
            reasoning=reasoning,
        )

    def _merge_plans(
        self,
        rule_plan: QueryPlan,
        llm_plan: QueryPlan,
    ) -> QueryPlan:
        # Deterministic parsing remains authoritative for numeric/property filters.
        # The LLM supplies semantic intent and source requirements.
        llm_plan.entity_candidates = self._dedupe(
            list(llm_plan.entity_candidates) + list(rule_plan.entity_candidates)
        )

        if rule_plan.filters:
            llm_plan.filters.update(rule_plan.filters)

        if rule_plan.needs_conversation_context:
            llm_plan.needs_conversation_context = True

        llm_plan.needs_structured_data = (
            llm_plan.needs_structured_data or rule_plan.needs_structured_data
        )
        llm_plan.needs_documents = (
            llm_plan.needs_documents or rule_plan.needs_documents
        )
        llm_plan.needs_analytics = (
            llm_plan.needs_analytics or rule_plan.needs_analytics
        )
        llm_plan.needs_prediction = (
            llm_plan.needs_prediction or rule_plan.needs_prediction
        )
        llm_plan.needs_ranking = (
            llm_plan.needs_ranking or rule_plan.needs_ranking
        )
        llm_plan.needs_comparison = (
            llm_plan.needs_comparison or rule_plan.needs_comparison
        )

        # A ranking/investment/calculation/prediction request needs structured data.
        if llm_plan.intent in {
            "investment",
            "ranking",
            "comparison",
            "calculation",
            "prediction",
            "search",
        }:
            llm_plan.needs_structured_data = True

        if llm_plan.needs_analytics or llm_plan.needs_prediction:
            llm_plan.needs_structured_data = True

        if llm_plan.needs_ranking and "rank" not in llm_plan.operations:
            llm_plan.operations.append("rank")

        if llm_plan.needs_comparison and "compare" not in llm_plan.operations:
            llm_plan.operations.append("compare")

        if llm_plan.needs_prediction and "predict" not in llm_plan.operations:
            llm_plan.operations.append("predict")

        if llm_plan.needs_analytics and "investment_analysis" not in llm_plan.operations and llm_plan.intent == "investment":
            llm_plan.operations.append("investment_analysis")

        llm_plan.operations = self._dedupe(llm_plan.operations)
        llm_plan.reasoning = self._dedupe(
            list(llm_plan.reasoning) + list(rule_plan.reasoning)
        )
        return llm_plan

    @staticmethod
    def _contains_any(text: str, terms) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @staticmethod
    def _looks_like_follow_up(text: str) -> bool:
        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in QueryPlanner.FOLLOW_UP_PATTERNS
        )

    @staticmethod
    def _detect_entity_type(text: str) -> Optional[str]:
        # Order matters: a property/project phrase is more specific than
        # the generic word "place".
        if re.search(r"\b(project|projects|development|developments)\b", text):
            return "project"

        if re.search(r"\b(property|properties|listing|listings|unit|units|apartment|villa|townhouse)\b", text):
            return "property"

        if re.search(r"\b(community|communities|area|district|neighborhood|neighbourhood)\b", text):
            return "community"

        if re.search(r"\b(developer|developers)\b", text):
            return "developer"

        return None

    @staticmethod
    def _extract_filters(question: str) -> Dict[str, Any]:
        q = question.lower()
        filters: Dict[str, Any] = {}

        # ------------------------------------------------------------
        # Budget / price
        # ------------------------------------------------------------
        budget_patterns = [
            r"(?:under|below|less than|up to|max(?:imum)?(?: budget)?(?: of)?)\s*(?:aed|dh|د\.إ)?\s*([\d,.]+)\s*(million|m|k|thousand)?",
            r"(?:budget|price range)\s*(?:of|is|:)?\s*(?:aed|dh|د\.إ)?\s*([\d,.]+)\s*(million|m|k|thousand)?",
        ]

        for pattern in budget_patterns:
            match = re.search(pattern, q)
            if match:
                amount = QueryPlanner._parse_amount(
                    match.group(1),
                    match.group(2),
                )
                if amount is not None:
                    filters["max_price_aed"] = amount
                    break

        # ------------------------------------------------------------
        # Minimum price
        # ------------------------------------------------------------
        minimum_patterns = [
            r"(?:above|over|more than|at least)\s*(?:aed|dh|د\.إ)?\s*([\d,.]+)\s*(million|m|k|thousand)?",
        ]

        for pattern in minimum_patterns:
            match = re.search(pattern, q)
            if match:
                amount = QueryPlanner._parse_amount(
                    match.group(1),
                    match.group(2),
                )
                if amount is not None:
                    filters["min_price_aed"] = amount
                    break

        # ------------------------------------------------------------
        # Bedrooms
        # ------------------------------------------------------------
        bedroom_match = re.search(
            r"\b(\d+)\s*(?:to|-)\s*(\d+)\s*bed(?:room)?s?\b",
            q,
        )

        if bedroom_match:
            filters["bedroom_min"] = int(bedroom_match.group(1))
            filters["bedroom_max"] = int(bedroom_match.group(2))
        else:
            bedroom_match = re.search(
                r"\b(\d+)\s*[- ]?\s*bed(?:room)?s?\b",
                q,
            )
            if bedroom_match:
                bedrooms = int(bedroom_match.group(1))
                filters["bedrooms"] = bedrooms

        # ------------------------------------------------------------
        # Explicit property types
        # ------------------------------------------------------------
        property_types = []
        for term in (
            "apartment",
            "villa",
            "townhouse",
            "penthouse",
            "studio",
            "duplex",
        ):
            if re.search(rf"\b{re.escape(term)}s?\b", q):
                property_types.append(term)

        if property_types:
            filters["property_types"] = property_types

        # ------------------------------------------------------------
        # Rental / sale intent
        # ------------------------------------------------------------
        if re.search(r"\b(rent|rental|for rent|renting)\b", q):
            filters["transaction_type"] = "rent"

        if re.search(r"\b(buy|buying|sale|for sale|purchase)\b", q):
            filters["transaction_type"] = "sale"

        return filters

    @staticmethod
    def _parse_amount(value: str, multiplier: Optional[str]) -> Optional[float]:
        try:
            amount = float(value.replace(",", "").strip())
        except (TypeError, ValueError):
            return None

        if not multiplier:
            return amount

        multiplier = multiplier.lower()

        if multiplier in {"million", "m"}:
            return amount * 1_000_000

        if multiplier in {"thousand", "k"}:
            return amount * 1_000

        return amount


if __name__ == "__main__":
    planner = QueryPlanner()

    examples = [
        "Tell me about Sky Edition at Seahaven",
        "Show me the best properties in Dubai Marina under AED 2M",
        "Compare two projects for rental investment",
        "Is this a good investment?",
        "What will property prices look like next year?",
        "What amenities does this project have?",
        "What about the rental price?",
        "How much is the price per sqft?",
    ]

    for question in examples:
        print("\nQUESTION:", question)
        print(planner.plan(question).to_dict())
