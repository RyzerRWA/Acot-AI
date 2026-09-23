"""
ACOT Conversation Memory

Maintains short-term conversational context for ACOT.

Responsibilities:
- Store recent conversation turns.
- Track active entities:
    city
    community
    sub-community
    project
    developer
- Track the latest candidate list returned by retrieval.
- Resolve follow-up references deterministically where safe:
    there
    here
    this community
    that community
    this project
    that project
    this one
    their
    them
    these projects
    those projects
    first / second / third
    1 and 2 / 2 and 3
- Rewrite follow-up questions into standalone questions when possible.
- Fall back to the LLM only when deterministic resolution is not safe.

Important:
This module does NOT retrieve data from Supabase.
It only maintains and resolves conversational context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class ConversationMemory:
    """
    Short-term conversational memory for ACOT.

    The memory is intentionally limited to the current conversation
    and a small number of recent turns.
    """

    MAX_TURNS = 8
    MAX_CANDIDATES = 20

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

        # ---------------------------------------------------------
        # Conversation turns
        # ---------------------------------------------------------
        self.turns: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Currently active entities
        #
        # ACOT hierarchy:
        #
        # city
        #   -> community
        #       -> sub-community
        #           -> project
        # ---------------------------------------------------------
        self.active_entities: Dict[str, Optional[str]] = {
            "city": None,
            "community": None,
            "sub_community": None,
            "project": None,
            "developer": None,
        }

        # ---------------------------------------------------------
        # Candidate records from the latest retrieval.
        #
        # Example:
        #
        # [
        #     {
        #         "name": "Samana Waves",
        #         "entity_type": "project"
        #     },
        #     {
        #         "name": "Azizi Ruby",
        #         "entity_type": "project"
        #     }
        # ]
        # ---------------------------------------------------------
        self.active_candidates: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Last questions
        # ---------------------------------------------------------
        self.last_user_question: Optional[str] = None
        self.last_standalone_question: Optional[str] = None

    # =============================================================
    # BASIC HELPERS
    # =============================================================

    @staticmethod
    def _clean(value: Any) -> Optional[str]:
        """
        Convert a value into a clean string.

        Returns None for empty values.
        """
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value

    @staticmethod
    def _normalise(value: Any) -> str:
        """
        Normalise text for comparisons.
        """
        value = ConversationMemory._clean(value)

        if not value:
            return ""

        value = value.lower()
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    @staticmethod
    def _get_name(item: Any) -> Optional[str]:
        """
        Safely extract a candidate name.
        """
        if isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("project_name")
                or item.get("community_name")
                or item.get("sub_community_name")
            )

            return ConversationMemory._clean(name)

        if isinstance(item, str):
            return ConversationMemory._clean(item)

        return None

    # =============================================================
    # CONTEXT
    # =============================================================

    def context(self) -> Dict[str, Any]:
        """
        Return the current conversation context.

        This object is passed to the planner/retriever.
        """
        return {
            "active_entities": dict(self.active_entities),
            "active_candidates": list(self.active_candidates),
            "recent_turns": list(self.turns[-self.MAX_TURNS :]),
            "last_user_question": self.last_user_question,
            "last_standalone_question": self.last_standalone_question,
        }

    # =============================================================
    # ACTIVE ENTITY MANAGEMENT
    # =============================================================

    def _set_entity(
        self,
        entity_type: str,
        value: Any,
    ) -> None:
        """
        Safely update one active entity.
        """
        if entity_type not in self.active_entities:
            return

        cleaned = self._clean(value)

        if cleaned:
            self.active_entities[entity_type] = cleaned

    def _set_active_entities_from_resolved(
        self,
        resolved_entity: Optional[Dict[str, Any]],
    ) -> None:
        """
        Update active entities from a resolved entity object.

        Supports objects/dicts commonly returned by the resolver.
        """

        if not resolved_entity:
            return

        # ---------------------------------------------------------
        # Support both dict and object-style resolved entities.
        # ---------------------------------------------------------
        if isinstance(resolved_entity, dict):
            entity_type = (
                resolved_entity.get("entity_type")
                or resolved_entity.get("type")
            )

            name = (
                resolved_entity.get("name")
                or resolved_entity.get("entity_name")
            )

            city = resolved_entity.get("city")
            community = resolved_entity.get("community")
            sub_community = (
                resolved_entity.get("sub_community")
                or resolved_entity.get("subcommunity")
            )
            developer = resolved_entity.get("developer")

        else:
            entity_type = getattr(
                resolved_entity,
                "entity_type",
                None,
            )

            name = getattr(
                resolved_entity,
                "name",
                None,
            )

            city = getattr(
                resolved_entity,
                "city",
                None,
            )

            community = getattr(
                resolved_entity,
                "community",
                None,
            )

            sub_community = getattr(
                resolved_entity,
                "sub_community",
                None,
            )

            developer = getattr(
                resolved_entity,
                "developer",
                None,
            )

        entity_type = self._normalise(entity_type)

        # ---------------------------------------------------------
        # First use explicit values returned by the resolver.
        # ---------------------------------------------------------
        self._set_entity("city", city)
        self._set_entity("community", community)
        self._set_entity("sub_community", sub_community)
        self._set_entity("developer", developer)

        # ---------------------------------------------------------
        # Then use entity type + name.
        # ---------------------------------------------------------
        if name:
            if entity_type in {
                "city",
                "community",
                "sub_community",
                "project",
                "developer",
            }:
                self._set_entity(entity_type, name)

    # =============================================================
    # FOLLOW-UP DETECTION
    # =============================================================

    def _looks_like_follow_up(
        self,
        question: str,
    ) -> bool:
        """
        Determine whether a question probably depends on
        previous conversation context.

        This is intentionally broad because the method only decides
        whether contextual rewriting should be attempted.
        It does NOT itself decide what the reference means.
        """

        question = (question or "").strip()

        if not question:
            return False

        q_lower = question.lower()

        # ---------------------------------------------------------
        # Explicit conversational phrases.
        # ---------------------------------------------------------
        follow_up_phrases = [
            "what about",
            "how about",
            "tell me more",
            "more details",
            "more information",
            "give me more",
            "the project",
            "the projects",
            "the property",
            "the properties",
            "the community",
            "the communities",
            "the developer",
            "the one",
            "the two",
            "which one",
            "which ones",
            "which of them",
            "compare them",
            "compare these",
            "compare those",
            "compare the two",
            "prices",
            "price",
            "how much",
            "how many",
            "their price",
            "their prices",
            "their amenities",
            "their features",
            "their details",
            "available there",
            "available here",
        ]

        for phrase in follow_up_phrases:
            if phrase in q_lower:
                return True

        # ---------------------------------------------------------
        # Location/context references.
        #
        # Added explicitly:
        #   there
        #   here
        # ---------------------------------------------------------
        contextual_words = [
            "there",
            "here",
            "this",
            "that",
            "these",
            "those",
            "it",
            "they",
            "them",
            "their",
            "theirs",
            "its",
            "one",
        ]

        for word in contextual_words:
            if re.search(
                rf"\b{re.escape(word)}\b",
                q_lower,
            ):
                return True

        # ---------------------------------------------------------
        # Numeric references:
        #
        # 1
        # 2
        # 1 and 2
        # 2 and 3
        # ---------------------------------------------------------
        if re.search(r"\b\d+\b", q_lower):
            return True

        # ---------------------------------------------------------
        # Ordinal references.
        # ---------------------------------------------------------
        ordinal_words = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]

        for word in ordinal_words:
            if re.search(
                rf"\b{re.escape(word)}\b",
                q_lower,
            ):
                return True

        # ---------------------------------------------------------
        # Question starters often used for contextual questions.
        # ---------------------------------------------------------
        question_starters = [
            "which",
            "what",
            "how much",
            "how many",
            "where",
            "when",
            "why",
        ]

        for starter in question_starters:
            if q_lower.startswith(starter):
                return True

        return False

    # =============================================================
    # CANDIDATE HELPERS
    # =============================================================

    def _candidate_names(self) -> List[str]:
        """
        Return clean candidate names while preserving order.
        """
        names: List[str] = []
        seen = set()

        for item in self.active_candidates:
            name = self._get_name(item)

            if not name:
                continue

            key = self._normalise(name)

            if key in seen:
                continue

            seen.add(key)
            names.append(name)

        return names

    # =============================================================
    # NUMERIC / ORDINAL REFERENCE HELPERS
    # =============================================================

    @staticmethod
    def _ordinal_to_number(word: str) -> Optional[int]:
        mapping = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }

        return mapping.get(
            ConversationMemory._normalise(word)
        )

    def _extract_numeric_pair(
        self,
        question: str,
    ) -> Optional[List[int]]:
        """
        Extract numeric references such as:

            1 and 2
            2 and 3
            1, 2
        """

        matches = re.findall(
            r"\b(\d+)\b",
            question or "",
        )

        if len(matches) < 2:
            return None

        values: List[int] = []

        for match in matches[:2]:
            try:
                values.append(int(match))
            except ValueError:
                return None

        return values

    def _extract_ordinal_pair(
        self,
        question: str,
    ) -> Optional[List[int]]:
        """
        Extract ordinal references such as:

            first and second
            second and third
        """

        pattern = (
            r"\b(first|second|third|fourth|fifth|"
            r"sixth|seventh|eighth|ninth|tenth)\b"
        )

        matches = re.findall(
            pattern,
            question or "",
            flags=re.IGNORECASE,
        )

        if len(matches) < 2:
            return None

        values: List[int] = []

        for match in matches[:2]:
            number = self._ordinal_to_number(match)

            if number is None:
                return None

            values.append(number)

        return values

    def _resolve_candidate_positions(
        self,
        positions: Optional[List[int]],
    ) -> List[str]:
        """
        Convert 1-based candidate positions into names.
        """
        if not positions:
            return []

        names = self._candidate_names()

        resolved: List[str] = []

        for position in positions:
            index = position - 1

            if index < 0 or index >= len(names):
                return []

            resolved.append(names[index])

        return resolved

    # =============================================================
    # NUMERIC / ORDINAL CANDIDATE REWRITING
    # =============================================================

    def _rewrite_candidate_reference(
        self,
        question: str,
    ) -> Optional[str]:
        """
        Resolve numeric and ordinal candidate references.

        Examples:

            "Compare 1 and 2"
                ->
            'Compare "Samana Waves" and "Azizi Ruby"'

            "Compare the second and third projects"
                ->
            'Compare "Azizi Ruby" and "Dawn by Binghatti"'
        """

        if not self.active_candidates:
            return None

        original = (question or "").strip()

        if not original:
            return None

        # ---------------------------------------------------------
        # Numeric pair.
        # ---------------------------------------------------------
        positions = self._extract_numeric_pair(
            original
        )

        if positions:
            names = self._resolve_candidate_positions(
                positions
            )

            if names:
                rewritten = original

                # Replace the exact numeric values only.
                for number, name in zip(
                    positions,
                    names,
                ):
                    rewritten = re.sub(
                        rf"\b{number}\b",
                        f'"{name}"',
                        rewritten,
                        count=1,
                    )

                if rewritten != original:
                    return rewritten

        # ---------------------------------------------------------
        # Ordinal pair.
        # ---------------------------------------------------------
        positions = self._extract_ordinal_pair(
            original
        )

        if positions:
            names = self._resolve_candidate_positions(
                positions
            )

            if names:
                rewritten = original

                ordinal_words = {
                    1: "first",
                    2: "second",
                    3: "third",
                    4: "fourth",
                    5: "fifth",
                    6: "sixth",
                    7: "seventh",
                    8: "eighth",
                    9: "ninth",
                    10: "tenth",
                }

                for position, name in zip(
                    positions,
                    names,
                ):
                    ordinal = ordinal_words.get(
                        position
                    )

                    if not ordinal:
                        continue

                    rewritten = re.sub(
                        rf"\b{ordinal}\b",
                        f'"{name}"',
                        rewritten,
                        count=1,
                        flags=re.IGNORECASE,
                    )

                if rewritten != original:
                    return rewritten

        return None

    # =============================================================
    # CONTEXT REFERENCE REWRITING
    # =============================================================

    def _rewrite_context_reference(
        self,
        question: str,
    ) -> Optional[str]:
        """
        Resolve safe conversational references without using Gemini.

        This is important for ACOT because simple references such as
        "there" do not require semantic reasoning.

        Examples:

            What projects are available there?
                ->
            What projects are available in Jumeirah Village Circle?

            Tell me more about this community.
                ->
            Tell me more about Jumeirah Village Circle.

            Tell me more about this project.
                ->
            Tell me more about Samana Waves.

            What are their prices?
                ->
            What are the prices of "Samana Waves" and "Azizi Ruby"?
            (only when exactly two active candidates exist)
        """

        original = (question or "").strip()

        if not original:
            return None

        q_lower = original.lower()

        # =========================================================
        # 1. "there" / "here"
        #
        # For location references, use the parent community first.
        #
        # This matches the actual ACOT database relationship:
        #
        # projects.community -> communities.name
        #
        # Therefore:
        #
        # "What projects are available there?"
        #
        # should resolve to:
        #
        # "What projects are available in Jumeirah Village Circle?"
        # =========================================================

        location = (
            self.active_entities.get("community")
            or self.active_entities.get("sub_community")
            or self.active_entities.get("city")
        )

        if location and re.search(
            r"\b(?:there|here)\b",
            q_lower,
        ):
            rewritten = re.sub(
                r"\b(?:there|here)\b",
                f"in {location}",
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        # =========================================================
        # 2. "this community" / "that community"
        # =========================================================

        community = self.active_entities.get(
            "community"
        )

        if community and re.search(
            r"\b(?:this|that)\s+community\b",
            q_lower,
        ):
            rewritten = re.sub(
                r"\b(?:this|that)\s+community\b",
                community,
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        # =========================================================
        # 3. "this sub-community" / "that sub-community"
        # =========================================================

        sub_community = self.active_entities.get(
            "sub_community"
        )

        if sub_community and re.search(
            r"\b(?:this|that)\s+sub[-\s]?community\b",
            q_lower,
        ):
            rewritten = re.sub(
                r"\b(?:this|that)\s+sub[-\s]?community\b",
                sub_community,
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        # =========================================================
        # 4. "this project" / "that project"
        # =========================================================

        project = self.active_entities.get(
            "project"
        )

        if project and re.search(
            r"\b(?:this|that)\s+project\b",
            q_lower,
        ):
            rewritten = re.sub(
                r"\b(?:this|that)\s+project\b",
                project,
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        # =========================================================
        # 5. "this one" / "that one"
        #
        # Safe only when exactly one project is active.
        # =========================================================

        if project and re.search(
            r"\b(?:this|that)\s+one\b",
            q_lower,
        ):
            rewritten = re.sub(
                r"\b(?:this|that)\s+one\b",
                project,
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        # =========================================================
        # 6. "their" / "them"
        #
        # Only resolve automatically for exactly TWO candidates.
        #
        # If there are five projects, "their" can be ambiguous.
        # We should not inject all five names into the query.
        # =========================================================

        candidate_names = self._candidate_names()

        if (
            len(candidate_names) == 2
            and re.search(
                r"\b(?:their|them)\b",
                q_lower,
            )
        ):
            first = candidate_names[0]
            second = candidate_names[1]

            replacement = (
                f'"{first}" and "{second}"'
            )

            rewritten = re.sub(
                r"\b(?:their|them)\b",
                replacement,
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        # =========================================================
        # 7. "these projects" / "those projects"
        #
        # Safe only when the candidate list is reasonably small.
        # =========================================================

        if (
            1 <= len(candidate_names) <= 5
            and re.search(
                r"\b(?:these|those)\s+projects\b",
                q_lower,
            )
        ):
            names = ", ".join(
                f'"{name}"'
                for name in candidate_names
            )

            rewritten = re.sub(
                r"\b(?:these|those)\s+projects\b",
                names,
                original,
                count=1,
                flags=re.IGNORECASE,
            )

            if rewritten != original:
                return rewritten

        return None

    # =============================================================
    # QUESTION REWRITING
    # =============================================================

    def rewrite_question(
        self,
        question: str,
    ) -> str:
        """
        Convert a conversational follow-up into a standalone question.

        Processing order:

        1. Return original for first question.
        2. Detect whether this is a follow-up.
        3. Resolve numeric/ordinal references.
        4. Resolve deterministic context references.
        5. Only then call Gemini if necessary.

        This order reduces unnecessary Gemini usage.
        """

        original = (question or "").strip()

        if not original:
            return original

        # ---------------------------------------------------------
        # First user question has no previous context.
        # ---------------------------------------------------------
        if not self.turns:
            return original

        # ---------------------------------------------------------
        # If this does not look contextual, keep it unchanged.
        # ---------------------------------------------------------
        if not self._looks_like_follow_up(
            original
        ):
            return original

        # ---------------------------------------------------------
        # STEP 1:
        # Numeric / ordinal references.
        #
        # Example:
        # "compare 1 and 2"
        # ---------------------------------------------------------
        deterministic = self._rewrite_candidate_reference(
            original
        )

        if deterministic:
            return deterministic

        # ---------------------------------------------------------
        # STEP 2:
        # Deterministic conversational references.
        #
        # Examples:
        #
        # "there"
        # "here"
        # "this community"
        # "this project"
        # "their prices"
        # ---------------------------------------------------------
        context_reference = self._rewrite_context_reference(
            original
        )

        if context_reference:
            return context_reference

        # ---------------------------------------------------------
        # STEP 3:
        # No LLM available.
        # ---------------------------------------------------------
        if self.llm_client is None:
            return original

        # ---------------------------------------------------------
        # STEP 4:
        # LLM fallback.
        #
        # Only use Gemini when deterministic resolution is not
        # sufficient.
        # ---------------------------------------------------------

        active_entities = {
            key: value
            for key, value in self.active_entities.items()
            if value
        }

        candidate_names = self._candidate_names()

        recent_turns = self.turns[
            -self.MAX_TURNS :
        ]

        conversation_lines = []

        for turn in recent_turns:
            user_question = turn.get(
                "user_question"
            )

            standalone_question = turn.get(
                "standalone_question"
            )

            if user_question:
                conversation_lines.append(
                    f"User: {user_question}"
                )

            if standalone_question:
                conversation_lines.append(
                    f"Resolved: {standalone_question}"
                )

        conversation_text = "\n".join(
            conversation_lines
        )

        prompt = f"""
You are the conversation-reference resolver for ACOT,
an AI-powered Dubai real estate intelligence system.

Rewrite the user's latest question into a standalone question.

Use ONLY the information present in the conversation context.

Current active entities:
{active_entities}

Current candidate entities:
{candidate_names}

Recent conversation:
{conversation_text}

Latest user question:
{original}

Rules:

1. Preserve the user's original intent.
2. Resolve references such as:
   - it
   - this
   - that
   - these
   - those
   - there
   - here
   - they
   - them
   - their
   - the project
   - the property
   - the community
   - which one
   - the first one
   - the second one
   - numeric references such as 1 and 2
3. Do not invent entities.
4. Do not introduce information that is not in the context.
5. If the reference is ambiguous, preserve the question rather than guessing.
6. Keep the rewritten question concise.
7. Return ONLY the rewritten question.
"""

        try:
            rewritten = self.llm_client.generate(
                prompt
            )

            rewritten = self._clean(
                rewritten
            )

            if rewritten:
                return rewritten

        except Exception:
            # Conversation rewriting must never break the
            # main ACOT retrieval flow.
            pass

        return original

    # =============================================================
    # CANDIDATE EXTRACTION
    # =============================================================

    def _extract_candidates(
        self,
        retrieval_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate entities from a retrieval result.

        Candidate priority:

            projects
            communities
            sub_communities

        This preserves the current ACOT behavior where a project
        result set becomes the active candidate set.
        """

        if not retrieval_result:
            return []

        candidates: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Helper for adding candidate objects.
        # ---------------------------------------------------------
        def add_candidates(
            items: Any,
            entity_type: str,
        ) -> None:

            if not isinstance(items, list):
                return

            for item in items:
                if isinstance(item, dict):
                    name = self._get_name(item)

                    if not name:
                        continue

                    candidate = dict(item)

                    candidate.setdefault(
                        "name",
                        name,
                    )

                    candidate.setdefault(
                        "entity_type",
                        entity_type,
                    )

                    candidates.append(
                        candidate
                    )

                elif isinstance(item, str):
                    name = self._clean(item)

                    if not name:
                        continue

                    candidates.append(
                        {
                            "name": name,
                            "entity_type": entity_type,
                        }
                    )

        # ---------------------------------------------------------
        # Structured result locations.
        # ---------------------------------------------------------
        structured_data = retrieval_result.get(
            "structured_data"
        )

        if not isinstance(
            structured_data,
            dict,
        ):
            structured_data = {}

        # ---------------------------------------------------------
        # Preserve project priority.
        # ---------------------------------------------------------
        projects = (
            retrieval_result.get("projects")
            or structured_data.get("projects")
        )

        communities = (
            retrieval_result.get("communities")
            or structured_data.get("communities")
        )

        sub_communities = (
            retrieval_result.get("sub_communities")
            or structured_data.get(
                "sub_communities"
            )
        )

        if projects:
            add_candidates(
                projects,
                "project",
            )

            if candidates:
                return candidates[
                    : self.MAX_CANDIDATES
                ]

        if communities:
            add_candidates(
                communities,
                "community",
            )

            if candidates:
                return candidates[
                    : self.MAX_CANDIDATES
                ]

        if sub_communities:
            add_candidates(
                sub_communities,
                "sub_community",
            )

            if candidates:
                return candidates[
                    : self.MAX_CANDIDATES
                ]

        return candidates[
            : self.MAX_CANDIDATES
        ]

    # =============================================================
    # UPDATE MEMORY
    # =============================================================

    def update(
        self,
        user_question: Optional[str] = None,
        standalone_question: Optional[str] = None,
        retrieval_result: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        question_type: Optional[str] = None,
        question: Optional[str] = None,
    ) -> None:
        """
        Update memory after an ACOT question has been processed.
        """

        # run.py calls this argument user_question.
        # Keep question as a small backward-compatible alias.
        if user_question is None:
            user_question = question

        user_question = (
            user_question or ""
        ).strip()

        standalone_question = (
            standalone_question
            or user_question
        ).strip()

        # ---------------------------------------------------------
        # Update last-question state.
        # ---------------------------------------------------------
        self.last_user_question = user_question
        self.last_standalone_question = (
            standalone_question
        )

        # ---------------------------------------------------------
        # Resolve active entity.
        # ---------------------------------------------------------
        resolved_entity = None

        if retrieval_result:
            resolved_entity = retrieval_result.get(
                "resolved_entity"
            )

            if resolved_entity:
                self._set_active_entities_from_resolved(
                    resolved_entity
                )

        # ---------------------------------------------------------
        # Some retrievers expose entity information directly.
        # ---------------------------------------------------------
        if retrieval_result:
            self._set_entity(
                "city",
                retrieval_result.get(
                    "city_name"
                ),
            )

            self._set_entity(
                "community",
                retrieval_result.get(
                    "community_name"
                ),
            )

            self._set_entity(
                "sub_community",
                retrieval_result.get(
                    "sub_community_name"
                ),
            )

            self._set_entity(
                "project",
                retrieval_result.get(
                    "project_name"
                ),
            )

            self._set_entity(
                "developer",
                retrieval_result.get(
                    "developer_name"
                ),
            )

        # ---------------------------------------------------------
        # Extract candidates from the latest result.
        # ---------------------------------------------------------
        # Keep the previously verified multi-project candidate list when
        # the user asks about ONE project that is already inside that list.
        # Example:
        #   1) Show me the projects in Jumeirah Village
        #   2) What is the handover date of Azizi Ruby?
        #   3) Which of these have 2 bedroom options?
        #
        # In step 2, the active entity should become Azizi Ruby, but the
        # candidate list must remain the original project list so that
        # "which of these" still refers to all previously shown projects.
        candidates = self._extract_candidates(
            retrieval_result
        )

        previous_candidates = list(
            self.active_candidates or []
        )

        preserve_previous_project_list = False

        if (
            previous_candidates
            and len(previous_candidates) > 1
            and len(candidates) == 1
        ):
            only_candidate = candidates[0]

            candidate_type = self._normalise(
                only_candidate.get("entity_type")
            )

            candidate_name = self._get_name(
                only_candidate
            )

            if candidate_type == "project" and candidate_name:
                searchable_question = " ".join(
                    [
                        str(user_question or ""),
                        str(standalone_question or ""),
                    ]
                ).lower()

                previous_project_names = {
                    self._get_name(item).strip().lower()
                    for item in previous_candidates
                    if isinstance(item, dict)
                    and self._normalise(
                        item.get("entity_type")
                    ) == "project"
                    and self._get_name(item)
                }

                # Preserve only when the single project is explicitly
                # selected from the previous candidate list.
                preserve_previous_project_list = (
                    candidate_name.strip().lower()
                    in previous_project_names
                    and candidate_name.strip().lower()
                    in searchable_question
                )

        if candidates:
            if not preserve_previous_project_list:
                self.active_candidates = candidates
            # Otherwise keep the previous multi-project list intact.

        # ---------------------------------------------------------
        # If there is exactly one project, make it the active
        # project and inherit its parent context where available.
        # ---------------------------------------------------------
        project_candidates = [
            item
            for item in candidates
            if self._normalise(
                item.get("entity_type")
            ) == "project"
        ]

        if len(project_candidates) == 1:
            project = project_candidates[0]

            project_name = self._get_name(
                project
            )

            self._set_entity(
                "project",
                project_name,
            )

            # -----------------------------------------------------
            # Actual ACOT projects contain these fields.
            # -----------------------------------------------------
            self._set_entity(
                "community",
                project.get("community"),
            )

            self._set_entity(
                "sub_community",
                project.get(
                    "sub_community"
                ),
            )

            self._set_entity(
                "city",
                project.get("city"),
            )

            self._set_entity(
                "developer",
                project.get(
                    "developer_name"
                ),
            )

        # ---------------------------------------------------------
        # If exactly one community is returned, make it active.
        # ---------------------------------------------------------
        community_candidates = [
            item
            for item in candidates
            if self._normalise(
                item.get("entity_type")
            ) == "community"
        ]

        if len(community_candidates) == 1:
            community = community_candidates[0]

            self._set_entity(
                "community",
                self._get_name(
                    community
                ),
            )

            self._set_entity(
                "city",
                community.get("city"),
            )

        # ---------------------------------------------------------
        # If exactly one sub-community is returned, make it active.
        # ---------------------------------------------------------
        sub_candidates = [
            item
            for item in candidates
            if self._normalise(
                item.get("entity_type")
            ) == "sub_community"
        ]

        if len(sub_candidates) == 1:
            sub_community = sub_candidates[0]

            self._set_entity(
                "sub_community",
                self._get_name(
                    sub_community
                ),
            )

            self._set_entity(
                "community",
                sub_community.get(
                    "community"
                ),
            )

            self._set_entity(
                "city",
                sub_community.get(
                    "city"
                ),
            )

        # ---------------------------------------------------------
        # Store conversation turn.
        # ---------------------------------------------------------
        turn = {
            "user_question": user_question,
            "standalone_question": (
                standalone_question
            ),
            "answer": answer,
            "question_type": question_type,
            "active_entities": dict(
                self.active_entities
            ),
            "candidate_names": self._candidate_names(),
        }

        self.turns.append(turn)

        # ---------------------------------------------------------
        # Keep memory bounded.
        # ---------------------------------------------------------
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[
                -self.MAX_TURNS :
            ]

    # =============================================================
    # CLEAR / RESET
    # =============================================================

    def clear(self) -> None:
        """
        Clear all conversational memory.
        """

        self.turns.clear()

        self.active_entities = {
            "city": None,
            "community": None,
            "sub_community": None,
            "project": None,
            "developer": None,
        }

        self.active_candidates = []

        self.last_user_question = None
        self.last_standalone_question = None

    # =============================================================
    # DEBUG
    # =============================================================

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a compact representation useful for debugging.
        """

        return {
            "active_entities": dict(
                self.active_entities
            ),
            "active_candidates": self._candidate_names(),
            "last_user_question": (
                self.last_user_question
            ),
            "last_standalone_question": (
                self.last_standalone_question
            ),
            "turn_count": len(self.turns),
        }