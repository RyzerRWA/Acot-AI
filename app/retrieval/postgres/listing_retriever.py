"""Search public.listings for rental questions.

Structured filters (community, bedrooms, price, type) use SQL.
Vague wording also uses match_listings, the same cosine search as the
other tables. Results are individual rental listings, not projects.
"""

import re

from app.database.postgres_client import PostgresClient


_LISTING_COLUMNS = """
    listing_id, title, purpose, property_type, category, beds, baths,
    size_sqft, price_aed, rent_frequency, furnishing, completion_status,
    community, sub_community, location, agency_name,
    cover_photo, image_urls, url
"""

_DROP_KEYS = {"embedding", "knowledge_text", "sim"}


class ListingRetriever:

    _shared_model = None
    _shared_communities = None

    def __init__(self, client: PostgresClient = None):
        self.client = client or PostgresClient()

    def search(self, question: str, filters: dict = None, limit: int = 12) -> dict:
        filters = filters or {}
        community = self._match_community(question)
        constraints = self._constraints(filters)
        limit = max(1, int(limit))

        vector_rows = self._vector_search(
            question,
            community,
            constraints,
            limit,
        )
        sql_rows = []
        total = None
        if community or constraints["active"]:
            sql_rows, total = self._sql_search(
                community,
                constraints,
                limit,
            )

        merged = []
        seen = set()
        for row in vector_rows + sql_rows:
            key = str(row.get("listing_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(row)
            if len(merged) >= limit:
                break

        if total is None:
            total = len(merged)

        return {
            "listings": merged,
            "listing_total": total,
            "subject": community or "that request",
        }

    def _match_community(self, question: str) -> str:
        text = (question or "").lower()
        best = ""
        for name in self._communities():
            needle = re.sub(r"\s*\([^)]*\)", "", name or "")
            needle = re.sub(r"\s+", " ", needle).strip().lower()
            if len(needle) < 4:
                continue
            if needle in text and len(needle) > len(best):
                best = needle
        return best

    def _communities(self):
        if ListingRetriever._shared_communities is None:
            with self.client.pool.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        select distinct community
                        from public.listings
                        where community is not null
                          and btrim(community) <> ''
                        """
                    )
                    ListingRetriever._shared_communities = [
                        row["community"] for row in cursor.fetchall()
                    ]
        return ListingRetriever._shared_communities

    @staticmethod
    def _constraints(filters: dict) -> dict:
        types = [
            str(item).strip().lower()
            for item in (filters.get("property_types") or [])
            if str(item).strip()
        ]
        bedrooms = filters.get("bedrooms")
        bed_min = filters.get("bedroom_min")
        bed_max = filters.get("bedroom_max")
        if bedrooms is None and types == ["studio"]:
            bedrooms = 0

        constraints = {
            "beds": bedrooms,
            "bed_min": bed_min,
            "bed_max": bed_max,
            "min_price": filters.get("min_price_aed"),
            "max_price": filters.get("max_price_aed"),
            "types": types,
        }
        constraints["active"] = any(
            constraints[key] not in (None, [], "")
            for key in ("beds", "bed_min", "bed_max", "min_price", "max_price", "types")
        )
        return constraints

    def _sql_search(self, community: str, constraints: dict, limit: int):
        clauses = ["true"]
        params = []

        if community:
            clauses.append("community ilike %s")
            params.append(f"%{community}%")

        self._append_filter_clauses(clauses, params, constraints)
        where = " and ".join(clauses)

        count_sql = f"select count(*) as total from public.listings where {where}"
        row_sql = f"""
            select {_LISTING_COLUMNS}
            from public.listings
            where {where}
            order by price_aed asc nulls last
            limit %s
        """

        with self.client.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(count_sql, params)
                total = int(cursor.fetchone()["total"])
                cursor.execute(row_sql, [*params, limit])
                rows = [self._public_row(row) for row in cursor.fetchall()]
        return rows, total

    def _vector_search(self, question, community, constraints, limit):
        if not question:
            return []

        if ListingRetriever._shared_model is None:
            from app.rag.embeddings.embedding_model import EmbeddingModel
            ListingRetriever._shared_model = EmbeddingModel()

        embedding = ListingRetriever._shared_model.embed_query(question)
        values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        response = self.client.rpc(
            "match_listings",
            {
                "query_embedding": values,
                "match_threshold": 0.35,
                "match_count": max(limit * 3, 24),
            },
        ).execute()

        matched = []
        for row in response.data or []:
            if not isinstance(row, dict):
                continue
            if not self._row_matches(row, community, constraints):
                continue
            matched.append(self._public_row(row))
            if len(matched) >= limit:
                break
        return matched

    @staticmethod
    def _append_filter_clauses(clauses, params, constraints):
        if constraints["beds"] is not None:
            clauses.append("beds = %s")
            params.append(int(constraints["beds"]))
        else:
            if constraints["bed_min"] is not None:
                clauses.append("beds >= %s")
                params.append(int(constraints["bed_min"]))
            if constraints["bed_max"] is not None:
                clauses.append("beds <= %s")
                params.append(int(constraints["bed_max"]))

        if constraints["min_price"] is not None:
            clauses.append("price_aed >= %s")
            params.append(constraints["min_price"])
        if constraints["max_price"] is not None:
            clauses.append("price_aed <= %s")
            params.append(constraints["max_price"])

        types = [
            item for item in constraints["types"]
            if item != "studio" or constraints["beds"] != 0
        ]
        if types:
            type_clauses = []
            for item in types:
                type_clauses.append("property_type ilike %s")
                params.append(f"%{item}%")
            clauses.append("(" + " or ".join(type_clauses) + ")")

    @classmethod
    def _row_matches(cls, row: dict, community: str, constraints: dict) -> bool:
        if community and community not in str(row.get("community") or "").lower():
            return False

        beds = row.get("beds")
        if constraints["beds"] is not None and beds != int(constraints["beds"]):
            return False
        if constraints["beds"] is None:
            if constraints["bed_min"] is not None and (beds is None or beds < int(constraints["bed_min"])):
                return False
            if constraints["bed_max"] is not None and (beds is None or beds > int(constraints["bed_max"])):
                return False

        price = row.get("price_aed")
        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None
        if constraints["min_price"] is not None and (price is None or price < float(constraints["min_price"])):
            return False
        if constraints["max_price"] is not None and (price is None or price > float(constraints["max_price"])):
            return False

        types = constraints["types"]
        if types:
            label = str(row.get("property_type") or "").lower()
            if not any(item in label or (item == "studio" and beds == 0) for item in types):
                return False
        return True

    @staticmethod
    def _public_row(row: dict) -> dict:
        return {
            key: value
            for key, value in row.items()
            if key not in _DROP_KEYS
        }
