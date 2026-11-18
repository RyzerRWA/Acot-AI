from app.retrieval.place_lock import (
    contains_place_sql,
    drop_numbered_siblings,
    is_numbered_sibling,
    sales_place_sql,
)


def test_numbered_siblings_are_distinct_places():
    assert is_numbered_sibling("Arabian Ranches", "Arabian Ranches 2")
    assert is_numbered_sibling("Arabian Ranches", "Arabian Ranches 3")
    assert is_numbered_sibling("Arabian Ranches", "Arabian Ranches III")
    assert not is_numbered_sibling("Arabian Ranches", "Arabian Ranches")
    assert not is_numbered_sibling("Arabian Ranches 3", "Arabian Ranches")
    assert not is_numbered_sibling("Arabian Ranches", "Dubai Hills Estate")


def test_drop_numbered_siblings_keeps_the_asked_place():
    rows = [
        {"name": "Alma", "community": "Arabian Ranches"},
        {"name": "Anya", "community": "Arabian Ranches 3"},
        {"name": "Spring", "community": "Arabian Ranches 3"},
    ]
    kept = drop_numbered_siblings(rows, ["Arabian Ranches"])
    assert [row["name"] for row in kept] == ["Alma"]


def test_contains_place_sql_excludes_numbered_suffix():
    sql, params = contains_place_sql(["master_project"], ["Arabian Ranches"])
    assert "ILIKE" in sql
    assert "not" in sql.lower()
    assert any("ARABIAN RANCHES" in str(item).upper() for group in params for item in group)
    assert any("3" in str(item) or "[0-9]" in str(item) for group in params for item in group)


def test_sales_place_sql_requires_master_lock():
    sql, params = sales_place_sql(["Arabian Ranches"], ["WADI AL SAFA 6"])
    assert "master_project_prop_ai" in sql
    assert params[0] == ["ARABIAN RANCHES"]
    assert "WADI AL SAFA 6" in params[1]
    assert params[2] == ["ARABIAN RANCHES"]
