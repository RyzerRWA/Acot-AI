"""Retired Bayut listing search.

The old `public.listings` table is no longer queried at runtime.
All search methods return empty results; use Prop-AI instead.
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

    def search(self, question: str, filters: dict = None, limit: int = 10) -> dict:
        """Retired: old Bayut `listings` table is not queried."""
        _ = question, filters, limit
        return {
            "listings": [],
            "listing_total": 0,
            "subject": "that request",
        }

    def _match_community(self, question: str) -> str:
        _ = question
        return ""

    def _communities(self):
        # Do not scan public.listings — table is retired.
        if ListingRetriever._shared_communities is None:
            ListingRetriever._shared_communities = []
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
        _ = community, constraints, limit
        return [], 0

    def _vector_search(self, question, community, constraints, limit):
        _ = question, community, constraints, limit
        return []

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
