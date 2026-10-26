import re

from app.database.postgres import fetch
from app.database.postgres_client import PostgresClient


_PROJECT_NAME_STOP = {
    "the", "and", "for", "of", "by", "at", "in", "on",
    "project", "projects", "property", "properties",
}


class StructuredRetriever:
    """
    Structured retrieval directly from PostgreSQL.

    Tables used:
        - communities
        - projects
        - sub_communities
    """

    def __init__(self):

        # Listings and catalog search still read this client.
        self.db = PostgresClient()

    # =========================================================
    # COMMON PROJECT FIELDS
    # =========================================================

    PROJECT_FIELDS = """
        id,
        slug,
        name,
        description,
        status,
        project_status,
        pool_type,
        developer_id,
        developer_name,
        city,
        community,
        sub_community,
        latitude,
        longitude,
        price,
        size_min,
        size_max,
        bedroom_min,
        bedroom_max,
        handover_time,
        photos,
        property_types,
        amenities,
        brochure_url,
        property_finder_url,
        knowledge_text,
        source,
        fetched_at,
        updated_at
    """

    # =========================================================
    # PROJECTS BY COMMUNITY
    # =========================================================

    def search_projects_by_community(
        self,
        community_name: str,
        limit: int = 20
    ):
        """
        Retrieve projects belonging to a community.
        """

        if not community_name:
            return []

        query = community_name.strip()

        return fetch(
            f"select {self.PROJECT_FIELDS} from projects where community ilike %s limit %s",
            (query, limit),
        )

    # =========================================================
    # PROJECTS BY NAME
    # =========================================================

    @staticmethod
    def _project_name_tokens(value: str) -> list:
        tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
        return [
            token
            for token in tokens
            if len(token) > 2 and token not in _PROJECT_NAME_STOP
        ]

    @classmethod
    def _rank_project_name_matches(cls, query: str, rows: list) -> list:
        """Prefer projects whose names cover the query tokens (any order)."""

        needle = cls._project_name_tokens(query)
        if not needle or not rows:
            return list(rows or [])

        needle_set = set(needle)
        ranked = []
        for row in rows:
            name = str(row.get("name") or "")
            name_norm = re.sub(r"[^a-z0-9\s]+", " ", name.lower())
            name_norm = re.sub(r"\s+", " ", name_norm).strip()
            query_norm = re.sub(r"[^a-z0-9\s]+", " ", str(query or "").lower())
            query_norm = re.sub(r"\s+", " ", query_norm).strip()
            name_tokens = set(cls._project_name_tokens(name))
            if not name_tokens:
                continue
            overlap = needle_set & name_tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(needle_set), 1)
            if query_norm and (
                query_norm in name_norm or name_norm in query_norm
            ):
                score += 0.35
            # Prefer names that include every query token.
            if overlap == needle_set:
                score += 0.4
            # Prefer tighter names when scores tie.
            score += max(0.0, 0.15 - (len(name_tokens) * 0.01))
            ranked.append((score, len(name), row))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked if item[0] >= 0.5]

    def search_projects_by_name(
        self,
        project_name: str,
        limit: int = 10
    ):
        """
        Retrieve projects by project name.

        Exact substring first, then token AND for reordered names such as
        "Binghatti Mercedes" → "Mercedes Benz Places by Binghatti".
        """

        if not project_name:
            return []

        query = project_name.strip()

        if not query:
            return []

        rows = fetch(
            f"select {self.PROJECT_FIELDS} from projects where name ilike %s limit %s",
            (f"%{query}%", limit),
        )
        if rows:
            return rows

        tokens = self._project_name_tokens(query)
        if not tokens:
            return []

        if len(tokens) == 1:
            rows = fetch(
                f"select {self.PROJECT_FIELDS} from projects "
                f"where name ilike %s limit %s",
                (f"%{tokens[0]}%", max(limit * 5, 40)),
            )
            return self._rank_project_name_matches(query, rows)[:limit]

        where = " and ".join(["name ilike %s"] * len(tokens))
        params = tuple(f"%{token}%" for token in tokens)
        rows = fetch(
            f"select {self.PROJECT_FIELDS} from projects where {where} limit %s",
            (*params, max(limit * 5, 40)),
        )
        return self._rank_project_name_matches(query, rows)[:limit]

    # =========================================================
    # PROJECTS BY NAME, DEVELOPER, AND COMMUNITY
    # =========================================================

    def search_projects_filtered(
        self,
        name_pattern: str = None,
        starts_with: bool = False,
        developer_name: str = None,
        community_name: str = None,
        limit: int = 200,
    ):
        """Return projects matching a name stem and optional developer or community.

        Developer names are stored with a leading space on some rows.
        The comparison uses the stripped name, and the lookup itself
        uses a contains match so that space does not hide the row.
        """

        name_pattern = (name_pattern or "").strip()
        developer_name = (developer_name or "").strip()
        community_name = (community_name or "").strip()

        if not name_pattern and not developer_name and not community_name:
            return []

        where, params = [], []

        if name_pattern:
            safe = name_pattern.replace("%", "").replace("_", " ").strip()
            where.append("name ilike %s")
            params.append(f"{safe}%" if starts_with else f"%{safe}%")

        if developer_name:
            safe_developer = developer_name.replace("%", "").replace("_", " ").strip()
            where.append("developer_name ilike %s")
            params.append(f"%{safe_developer}%")

        if community_name:
            safe_community = community_name.replace("%", "").replace("_", " ").strip()
            where.append("community ilike %s")
            params.append(f"%{safe_community}%")

        rows = fetch(
            f"select {self.PROJECT_FIELDS} from projects where {' and '.join(where)} limit %s",
            (*params, limit),
        )

        if community_name and rows:
            wanted_community = community_name.strip().lower()
            exact = [
                row for row in rows
                if str(row.get("community") or "").strip().lower() == wanted_community
            ]
            if exact:
                rows = exact

        if not developer_name:
            return rows

        wanted = developer_name.lower()
        return [
            row for row in rows
            if wanted in str(row.get("developer_name") or "").strip().lower()
        ]

    # =========================================================
    # PROJECT SEARCH
    # =========================================================

    def search_projects(
        self,
        project_name: str = None,
        community_name: str = None,
        limit: int = 20
    ):
        """
        Search projects using project name and/or community.

        Priority:
            1. Project name
            2. Community
        """

        if project_name:
            results = self.search_projects_by_name(
                project_name=project_name,
                limit=limit
            )

            if results:
                return results

        if community_name:
            return self.search_projects_by_community(
                community_name=community_name,
                limit=limit
            )

        return []

    # =========================================================
    # COMMUNITIES
    # =========================================================

    def search_community(
        self,
        community_name: str
    ):
        """
        Retrieve exact community information.
        """

        if not community_name:
            return []

        query = community_name.strip()

        return fetch(
            """
            select id, slug, name, city, latitude, longitude, photos,
                   sell_properties_count, rent_properties_count, projects_count,
                   pool_projects_count, total_count, assigned_agents, knowledge_text,
                   source, fetched_at, updated_at
            from communities where name ilike %s limit 5
            """,
            (query,),
        )

    # =========================================================
    # SUB-COMMUNITIES
    # =========================================================

    def search_sub_communities(
        self,
        community_name: str,
        limit: int = 20
    ):
        """
        Retrieve sub-communities belonging to a community.
        """

        if not community_name:
            return []

        query = community_name.strip()

        return fetch(
            """
            select id, slug, name, city, community, community_slug, latitude, longitude,
                   photos, knowledge_text, source, fetched_at, updated_at
            from sub_communities where community ilike %s limit %s
            """,
            (query, limit),
        )

    # =========================================================
    # GENERAL STRUCTURED RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        community_name: str = None,
        project_name: str = None,
        include_projects: bool = True,
        include_community: bool = True,
        include_sub_communities: bool = True
    ):
        """
        Retrieve structured information from PostgreSQL.

        Supports:
            - community-based retrieval
            - project-name retrieval
        """

        result = {
            "community": [],
            "projects": [],
            "sub_communities": []
        }

        # -----------------------------------------------------
        # PROJECT NAME SEARCH
        # -----------------------------------------------------

        if project_name:

            result["projects"] = self.search_projects_by_name(
                project_name=project_name
            )

            # If project found, don't require community
            if result["projects"]:

                # Try to get the community automatically
                project_community = result["projects"][0].get(
                    "community"
                )

                if project_community and include_community:
                    result["community"] = self.search_community(
                        project_community
                    )

                if project_community and include_sub_communities:
                    result["sub_communities"] = (
                        self.search_sub_communities(
                            project_community
                        )
                    )

                return result

        # -----------------------------------------------------
        # COMMUNITY SEARCH
        # -----------------------------------------------------

        if not community_name:
            return result

        if include_community:
            result["community"] = self.search_community(
                community_name
            )

        if include_projects:
            result["projects"] = (
                self.search_projects_by_community(
                    community_name
                )
            )

        if include_sub_communities:
            result["sub_communities"] = (
                self.search_sub_communities(
                    community_name
                )
            )

        return result