"""Browse answers that do not name a single project or community."""

_LIMIT = 12

_PROJECT_COLUMNS = """
    id, slug, name, description, status, project_status, pool_type,
    developer_id, developer_name, city, community, sub_community,
    price, size_min, size_max, bedroom_min, bedroom_max, handover_time,
    photos, property_types, brochure_url, property_finder_url, source
"""

_COMMUNITY_COLUMNS = "id, slug, name, city"

_LISTING_COLUMNS = """
    listing_id, title, purpose, property_type, category, beds, baths,
    size_sqft, price_aed, rent_frequency, furnishing, completion_status,
    community, sub_community, location, agency_name,
    cover_photo, image_urls, url
"""


def _safe_place(place: str) -> str:
    return (place or "").replace("%", "").replace("_", " ").strip()


def _query(client, statement: str, params: list):
    with client.pool.connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(statement, params)
            return cursor.fetchall()


def _place_clause(columns: str, place: str):
    if not place:
        return "true", []
    needle = f"%{place}%"
    parts = [f"{column.strip()} ilike %s" for column in columns.split(",")]
    return "(" + " or ".join(parts) + ")", [needle] * len(parts)


def retrieve_catalog(client, request: dict) -> dict:
    """Return a short sample of communities, projects, or rental listings."""

    place = _safe_place(request.get("place") or "")
    bedrooms = request.get("bedrooms")
    result = {
        "communities": [],
        "projects": [],
        "sub_communities": [],
        "listings": [],
        "listing_query": bool(request.get("listings")),
        "catalog": True,
        "catalog_place": place.title() if place else None,
    }

    if request.get("communities"):
        where, params = _place_clause("city, name", place)
        total = _query(
            client,
            f"select count(*) as total from public.communities where {where}",
            params,
        )
        rows = _query(
            client,
            f"""
            select {_COMMUNITY_COLUMNS}
            from public.communities
            where {where}
            order by name
            limit {_LIMIT}
            """,
            params,
        )
        result["communities"] = rows
        result["community_total"] = int(total[0]["total"]) if total else 0

    if request.get("projects"):
        where, params = _place_clause("city, community", place)
        bed_sql = ""
        if bedrooms is not None:
            bed_sql = " and bedroom_min <= %s and bedroom_max >= %s"
            params = [*params, int(bedrooms), int(bedrooms)]
        total = _query(
            client,
            f"select count(*) as total from public.projects where {where}{bed_sql}",
            params,
        )
        rows = _query(
            client,
            f"""
            select {_PROJECT_COLUMNS}
            from public.projects
            where {where}{bed_sql}
            order by case when price is null or price <= 0 then 1 else 0 end,
                     price asc,
                     name
            limit {_LIMIT}
            """,
            params,
        )
        result["projects"] = rows
        result["project_total"] = int(total[0]["total"]) if total else 0

    if request.get("listings"):
        where, params = _place_clause("community, location", place)
        bed_sql = ""
        if bedrooms is not None:
            bed_sql = " and beds = %s"
            params = [*params, int(bedrooms)]
        total = _query(
            client,
            f"select count(*) as total from public.listings where {where}{bed_sql}",
            params,
        )
        rows = _query(
            client,
            f"""
            select {_LISTING_COLUMNS}
            from public.listings
            where {where}{bed_sql}
            order by price_aed asc nulls last
            limit {_LIMIT}
            """,
            params,
        )
        result["listings"] = rows
        result["listing_total"] = int(total[0]["total"]) if total else 0
        result["listing_query"] = True

    if not result["communities"] and not result["projects"] and not result["listings"]:
        result["understood_subject"] = place.title() if place else "that request"
        result["understood_kind"] = "listing" if request.get("listings") else "place"

    print(
        "Catalog browse:",
        place or "anywhere",
        "communities", len(result["communities"]),
        "projects", len(result["projects"]),
        "listings", len(result["listings"]),
    )
    return result
