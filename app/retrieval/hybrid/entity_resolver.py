"""
ACOT Entity Resolver

Resolves the geographic/project hierarchy from live Supabase data.

Hierarchy:

    City
      ↓
    Community
      ↓
    Sub-community
      ↓
    Project

Examples:

    "best properties in Dubai"
        → City: Dubai

    "best properties in Dubai Marina"
        → City: Dubai
        → Community: Dubai Marina

    "Tell me about Sky Edition at Seahaven"
        → City: Dubai
        → Community: Dubai Marina
        → Project: Sky Edition at Seahaven

Important:
- Dubai is a city/location.
- Dubai Marina is a community inside Dubai.
- Project resolution is based on project name only.
- Community names do NOT resolve projects.
- Data is loaded from live Supabase.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Any, Dict, List, Optional


# ============================================================
# RESOLVED ENTITY
# ============================================================

@dataclass
class ResolvedEntity:

    entity_type: Optional[str] = None

    name: Optional[str] = None

    entity_id: Optional[Any] = None

    confidence: float = 0.0

    record: Optional[Dict[str, Any]] = None

    candidates: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # Geographic hierarchy
    location: Optional[Dict[str, Any]] = None

    community: Optional[Dict[str, Any]] = None

    reasoning: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "entity_type": self.entity_type,
            "name": self.name,
            "entity_id": self.entity_id,
            "confidence": round(
                self.confidence,
                4
            ),
            "record": self.record,
            "candidates": self.candidates,
            "location": self.location,
            "community": self.community,
            "reasoning": self.reasoning,
        }

# ============================================================
# ENTITY RESOLVER
# ============================================================

class EntityResolver:

    # ========================================================
    # SUPABASE FIELDS
    # ========================================================

    COMMUNITY_FIELDS = """
        id,
        slug,
        name,
        city
    """

    PROJECT_FIELDS = """
        id,
        slug,
        name,
        developer_name,
        city,
        community,
        sub_community,
        price,
        bedroom_min,
        bedroom_max,
        property_types
    """

    # ========================================================
    # STOP WORDS
    # ========================================================

    STOP_WORDS = {
        "show",
        "give",
        "tell",
        "me",
        "about",
        "the",
        "best",
        "top",
        "properties",
        "property",
        "projects",
        "project",
        "apartments",
        "apartment",
        "villas",
        "villa",
        "homes",
        "home",
        "in",
        "at",
        "near",
        "around",
        "within",
        "for",
        "under",
        "below",
        "above",
        "over",
        "with",
        "and",
        "or",
        "rental",
        "rent",
        "investment",
        "invest",
        "buy",
        "sale",
        "selling",
        "aed",
        "price",
        "prices",
        "available",
        "options",
        "good",
        "better",
        "recommend",
        "recommendation",
        "looking",
        "find",
        "search",
        "which",
        "what",
        "are",
        "is",
        "can",
        "you",
        "city",
        "location",
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        structured_retriever
    ):

        if structured_retriever is None:
            raise ValueError(
                "structured_retriever is required"
            )

        self.structured_retriever = (
            structured_retriever
        )

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize(
        text: str
    ) -> str:

        if not text:
            return ""

        text = text.lower().strip()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        aliases = {
            "jvc": "jumeirah village circle",
        }

        return aliases.get(
            text,
            text
        )

    # ========================================================
    # TOKEN SET
    # ========================================================

    @classmethod
    def token_set(
        cls,
        text: str
    ) -> set:

        return {
            token
            for token in cls.normalize(
                text
            ).split()
            if token not in cls.STOP_WORDS
        }

    # ========================================================
    # LOAD COMMUNITIES
    # ========================================================

    def _load_communities(
        self
    ) -> List[Dict[str, Any]]:

        response = (
            self
            .structured_retriever
            .supabase
            .table("communities")
            .select(
                self.COMMUNITY_FIELDS
            )
            .limit(1000)
            .execute()
        )

        return response.data or []

    # ========================================================
    # LOAD PROJECTS
    # ========================================================

    def _load_projects(
        self
    ) -> List[Dict[str, Any]]:

        response = (
            self
            .structured_retriever
            .supabase
            .table("projects")
            .select(
                self.PROJECT_FIELDS
            )
            .limit(2000)
            .execute()
        )

        return response.data or []

    # ========================================================
    # SCORE
    # ========================================================

    @classmethod
    def _score(
        cls,
        query: str,
        name: str
    ) -> float:

        q = cls.normalize(query)

        n = cls.normalize(name)

        if not q or not n:
            return 0.0

        # Exact match
        if q == n:
            return 1.0

        # Name contained inside query
        if n in q:
            return 0.96

        q_tokens = cls.token_set(
            query
        )

        n_tokens = cls.token_set(
            name
        )

        if not n_tokens:
            return 0.0

        overlap = (
            len(
                q_tokens & n_tokens
            )
            /
            len(n_tokens)
        )

        sequence = SequenceMatcher(
            None,
            q,
            n
        ).ratio()

        score = (
            0.70 * overlap
            +
            0.30 * sequence
        )

        return min(
            score,
            0.95
        )

    # ========================================================
    # EXTRACT CANDIDATE PHRASES
    # ========================================================

    @classmethod
    def _candidate_phrases(
        cls,
        question: str
    ) -> List[str]:

        if not question:
            return []

        q = question.strip()

        phrases = []

        patterns = [

            # Example:
            # "in Dubai Marina"
            r"\b(?:in|at|near|around|within)\s+"
            r"([A-Za-z][A-Za-z0-9&' -]{2,80})",

            # Example:
            # "projects of Dubai Marina"
            r"\b(?:of|for)\s+"
            r"([A-Za-z][A-Za-z0-9&' -]{2,80})",
        ]

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                q,
                flags=re.IGNORECASE
            ):

                phrase = match.group(1)

                # Stop at common query constraints
                phrase = re.split(
                    r"\b(?:under|below|above|over|with|and|or|for|"
                    r"rental|rent|investment|buy|sale|selling|price|"
                    r"bedrooms?|apartments?|villas?|properties?|projects?)\b",
                    phrase,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip(
                    " ,.-"
                )

                if phrase:
                    phrases.append(
                        phrase
                    )

        # Full question is retained for
        # exact project-name matching.
        phrases.append(q)

        # Remove duplicates
        unique = []

        seen = set()

        for phrase in sorted(
            phrases,
            key=len,
            reverse=True
        ):

            key = cls.normalize(
                phrase
            )

            if key and key not in seen:

                seen.add(key)

                unique.append(
                    phrase
                )

        return unique

    # ========================================================
    # RESOLVE CITY
    # ========================================================

    def _resolve_city(
        self,
        phrases: List[str],
        communities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:

        cities = {}

        # Extract unique cities from communities
        for record in communities:

            city = record.get(
                "city"
            )

            if city:

                key = self.normalize(
                    city
                )

                cities.setdefault(
                    key,
                    {
                        "name": city,
                        "city": city,
                    }
                )

        matches = []

        for city in cities.values():

            best = 0.0

            matched_phrase = None

            for candidate in phrases:

                score = self._score(
                    candidate,
                    city["name"]
                )

                if score > best:

                    best = score

                    matched_phrase = (
                        candidate
                    )

            if best >= 0.70:

                matches.append(
                    {
                        "entity_type": "city",
                        "name": city["name"],
                        "score": best,
                        "matched_phrase":
                            matched_phrase,
                        "record": city,
                    }
                )

        matches.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return (
            matches[0]
            if matches
            else None
        )

    # ========================================================
    # RESOLVE PROJECT
    # ========================================================

    def _resolve_project(
        self,
        phrases: List[str],
        projects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:

        matches = []

        for record in projects:

            name = record.get(
                "name"
            )

            if not name:
                continue

            best_score = 0.0

            matched_phrase = None

            # IMPORTANT:
            #
            # Only project NAME is used.
            #
            # We do NOT score against:
            # - community
            # - city
            # - sub-community
            #
            # This prevents:
            #
            # "Dubai Marina"
            # ->
            # "Sky Edition at Seahaven"
            #
            for phrase in phrases:

                score = self._score(
                    phrase,
                    name
                )

                if score > best_score:

                    best_score = score

                    matched_phrase = (
                        phrase
                    )

            if best_score >= 0.70:

                matches.append(
                    {
                        "entity_type": "project",
                        "name": name,
                        "id": record.get("id"),
                        "score": best_score,
                        "matched_phrase":
                            matched_phrase,
                        "record": record,
                    }
                )

        matches.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return (
            matches[0]
            if matches
            else None
        )

    # ========================================================
    # RESOLVE COMMUNITY
    # ========================================================

    def _resolve_community(
        self,
        phrases: List[str],
        communities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:

        matches = []

        for record in communities:

            name = record.get(
                "name"
            )

            if not name:
                continue

            best_score = 0.0

            matched_phrase = None

            for phrase in phrases:

                score = self._score(
                    phrase,
                    name
                )

                if score > best_score:

                    best_score = score

                    matched_phrase = (
                        phrase
                    )

            if best_score >= 0.70:

                matches.append(
                    {
                        "entity_type":
                            "community",

                        "name": name,

                        "id":
                            record.get("id"),

                        "score":
                            best_score,

                        "matched_phrase":
                            matched_phrase,

                        "record":
                            record,
                    }
                )

        matches.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return (
            matches[0]
            if matches
            else None
        )

    # ========================================================
    # MAIN RESOLUTION
    # ========================================================

    def resolve(
        self,
        question: str,
        entity_type: Optional[str] = None,
        candidates_from_planner:
            Optional[List[str]] = None,
    ) -> ResolvedEntity:

        if not question:

            return ResolvedEntity(
                reasoning=[
                    "No question supplied."
                ]
            )

        # ====================================================
        # BUILD PHRASES
        # ====================================================

        phrases = list(
            candidates_from_planner
            or []
        )

        phrases.extend(
            self._candidate_phrases(
                question
            )
        )

        unique_phrases = []

        seen = set()

        for phrase in phrases:

            key = self.normalize(
                phrase
            )

            if key and key not in seen:

                seen.add(key)

                unique_phrases.append(
                    phrase
                )

        # ====================================================
        # LOAD LIVE SUPABASE DATA
        # ====================================================

        communities = (
            self._load_communities()
        )

        projects = (
            self._load_projects()
        )

        # ====================================================
        # 1. PROJECT
        # ====================================================

        if entity_type in (
            None,
            "project"
        ):

            project_match = (
                self._resolve_project(
                    unique_phrases,
                    projects
                )
            )

            if project_match:

                record = (
                    project_match["record"]
                )

                location = None

                community = None

                # Project → City
                if record.get(
                    "city"
                ):

                    location = {
                        "name":
                            record["city"],

                        "city":
                            record["city"],
                    }

                # Project → Community
                if record.get(
                    "community"
                ):

                    community = {
                        "name":
                            record["community"],
                    }

                return ResolvedEntity(

                    entity_type="project",

                    name=project_match[
                        "name"
                    ],

                    entity_id=project_match[
                        "id"
                    ],

                    confidence=project_match[
                        "score"
                    ],

                    record=record,

                    candidates=[
                        project_match
                    ],

                    location=location,

                    community=community,

                    reasoning=[

                        f"Resolved project "
                        f"'{project_match['name']}' "
                        f"from live Supabase data.",

                        f"Matched query phrase: "
                        f"'{project_match['matched_phrase']}'.",

                        f"City: "
                        f"{record.get('city')}.",

                        f"Community: "
                        f"{record.get('community')}.",

                        f"Sub-community: "
                        f"{record.get('sub_community')}.",

                        f"Confidence: "
                        f"{project_match['score']:.2f}.",
                    ],
                )

        # ====================================================
        # 2. COMMUNITY
        # ====================================================

        if entity_type in (
            None,
            "community"
        ):

            community_match = (
                self._resolve_community(
                    unique_phrases,
                    communities
                )
            )

            if community_match:

                record = (
                    community_match["record"]
                )

                location = None

                if record.get(
                    "city"
                ):

                    location = {
                        "name":
                            record["city"],

                        "city":
                            record["city"],
                    }

                return ResolvedEntity(

                    entity_type="community",

                    name=community_match[
                        "name"
                    ],

                    entity_id=community_match[
                        "id"
                    ],

                    confidence=community_match[
                        "score"
                    ],

                    record=record,

                    candidates=[
                        community_match
                    ],

                    location=location,

                    community=record,

                    reasoning=[

                        f"Resolved community "
                        f"'{community_match['name']}' "
                        f"from live Supabase data.",

                        f"City: "
                        f"{record.get('city')}.",

                        f"Matched query phrase: "
                        f"'{community_match['matched_phrase']}'.",

                        f"Confidence: "
                        f"{community_match['score']:.2f}.",
                    ],
                )

        # ====================================================
        # 3. CITY
        # ====================================================

        if entity_type in (
            None,
            "city",
            "location"
        ):

            city_match = (
                self._resolve_city(
                    unique_phrases,
                    communities
                )
            )

            if city_match:

                return ResolvedEntity(

                    entity_type="city",

                    name=city_match[
                        "name"
                    ],

                    entity_id=None,

                    confidence=city_match[
                        "score"
                    ],

                    record=city_match[
                        "record"
                    ],

                    candidates=[
                        city_match
                    ],

                    location=city_match[
                        "record"
                    ],

                    community=None,

                    reasoning=[

                        f"Resolved location "
                        f"'{city_match['name']}' "
                        f"from live Supabase data.",

                        f"Matched query phrase: "
                        f"'{city_match['matched_phrase']}'.",

                        f"Confidence: "
                        f"{city_match['score']:.2f}.",
                    ],
                )

        # ====================================================
        # 4. NOTHING FOUND
        # ====================================================

        return ResolvedEntity(

            entity_type=None,

            name=None,

            entity_id=None,

            confidence=0.0,

            record=None,

            candidates=[],

            location=None,

            community=None,

            reasoning=[

                "No sufficiently strong "
                "project, community, or "
                "city match was found in "
                "live Supabase data."
            ],
        )