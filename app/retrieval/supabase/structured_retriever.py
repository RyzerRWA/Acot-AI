import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class SupabaseStructuredRetriever:
    """
    Structured retrieval directly from Supabase PostgreSQL.

    Tables used:
        - communities
        - projects
        - sub_communities
    """

    def __init__(self):

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SECRET_KEY")

        if not supabase_url:
            raise ValueError("SUPABASE_URL is not set")

        if not supabase_key:
            raise ValueError("SUPABASE_SECRET_KEY is not set")

        self.supabase: Client = create_client(
            supabase_url,
            supabase_key
        )

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

        response = (
            self.supabase
            .table("projects")
            .select(self.PROJECT_FIELDS)
            .ilike("community", query)
            .limit(limit)
            .execute()
        )

        return response.data or []

    # =========================================================
    # PROJECTS BY NAME
    # =========================================================

    def search_projects_by_name(
        self,
        project_name: str,
        limit: int = 10
    ):
        """
        Retrieve projects by project name.

        Example:
            "Sky Edition at Seahaven"
        """

        if not project_name:
            return []

        query = project_name.strip()

        if not query:
            return []

        response = (
            self.supabase
            .table("projects")
            .select(self.PROJECT_FIELDS)
            .ilike("name", f"%{query}%")
            .limit(limit)
            .execute()
        )

        return response.data or []

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

        query = (
            self.supabase
            .table("projects")
            .select(self.PROJECT_FIELDS)
        )

        if name_pattern:
            safe = name_pattern.replace("%", "").replace("_", " ").strip()
            pattern = f"{safe}%" if starts_with else f"%{safe}%"
            query = query.ilike("name", pattern)

        if developer_name:
            safe_developer = developer_name.replace("%", "").replace("_", " ").strip()
            query = query.ilike("developer_name", f"%{safe_developer}%")

        if community_name:
            safe_community = community_name.replace("%", "").replace("_", " ").strip()
            query = query.ilike("community", f"%{safe_community}%")

        response = query.limit(limit).execute()
        rows = response.data or []

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

        response = (
            self.supabase
            .table("communities")
            .select(
                """
                id,
                slug,
                name,
                city,
                latitude,
                longitude,
                photos,
                sell_properties_count,
                rent_properties_count,
                projects_count,
                pool_projects_count,
                total_count,
                assigned_agents,
                knowledge_text,
                source,
                fetched_at,
                updated_at
                """
            )
            .ilike("name", query)
            .limit(5)
            .execute()
        )

        return response.data or []

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

        response = (
            self.supabase
            .table("sub_communities")
            .select(
                """
                id,
                slug,
                name,
                city,
                community,
                community_slug,
                latitude,
                longitude,
                photos,
                knowledge_text,
                source,
                fetched_at,
                updated_at
                """
            )
            .ilike("community", query)
            .limit(limit)
            .execute()
        )

        return response.data or []

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
        Retrieve structured information from Supabase.

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