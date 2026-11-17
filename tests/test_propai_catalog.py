from datetime import datetime
from random import Random

from app.listings.agent import _ID_TAG
from app.retrieval.propai_catalog import (
    _agent_id,
    _focus,
    _handover,
    _order_projects,
    _photos,
    _public_photo,
    agent_projects,
    is_outside_dubai,
    known_dubai_label,
    resolve_dubai_place,
)


def test_focus_uses_the_alias_before_the_rest_of_the_sentence():
    assert _focus("Show me properties in Dubai Marina") == "Dubai Marina"
    assert _focus("projects in jvc") == "Jumeirah Village Circle"
    assert _focus("show me properties in marinna") == "Dubai Marina"
    assert len(_focus("hi")) < 3
    assert _focus("show me properties") == ""


def test_expired_signed_links_are_dropped_and_stored_files_are_public():
    signed = "https://example.r2.cloudflarestorage.com/a.jpg?X-Amz-Signature=abc"
    assert _photos([signed, "https://cdn.example/a.jpg"]) == ["https://cdn.example/a.jpg"]
    assert _photos(None) == []
    assert _public_photo("projects/p_085281b696/00.jpg").endswith("/projects/p_085281b696/00.jpg")
    assert _public_photo("../secret") is None


def test_new_project_ids_are_stable_and_the_agent_can_tag_them():
    assert _agent_id("NADD HESSA_SILICON OASIS_AL FALAK RESIDENCE") == _agent_id(
        "NADD HESSA_SILICON OASIS_AL FALAK RESIDENCE"
    )
    tagged = f"[[{_agent_id('same-key')}]]"
    assert _ID_TAG.findall(tagged) == [_agent_id("same-key")]


def test_size_filters_stay_on_the_older_project_table():
    assert agent_projects({"min_size": 1000}, limit=3) == ([], 0)


def test_latest_handover_ranks_new_projects_first():
    rows = [
        {"name": "Marina Park", "handover_time": datetime(2006, 1, 1), "price": 900_000},
        {"name": "Ciel", "handover_time": datetime(2025, 4, 1), "price": 800_000},
        {"name": "Pier 8", "handover_time": datetime(2026, 3, 1), "price": 777_548},
        {"name": "Future Bay", "handover_time": datetime(2028, 6, 1), "price": 2_000_000},
    ]
    names = [row["name"] for row in _order_projects(rows, "latest_handover", 4)]
    assert names == ["Future Bay", "Pier 8", "Ciel", "Marina Park"]
    cheapest = [row["name"] for row in _order_projects(rows, "cheapest", 4)]
    assert cheapest == ["Pier 8", "Ciel", "Marina Park", "Future Bay"]


def test_bedroom_filter_ranks_narrow_ranges_first():
    rows = [
        {
            "name": "Wide Tower",
            "bedroom_min": 1,
            "bedroom_max": 5,
            "handover_time": datetime(2029, 1, 1),
            "price": 3_000_000,
        },
        {
            "name": "Two Bed Focus",
            "bedroom_min": 1,
            "bedroom_max": 2,
            "handover_time": datetime(2026, 1, 1),
            "price": 1_500_000,
        },
        {
            "name": "Studio To Two",
            "bedroom_min": 0,
            "bedroom_max": 2,
            "handover_time": datetime(2028, 1, 1),
            "price": 1_200_000,
        },
    ]
    names = [
        row["name"]
        for row in _order_projects(
            rows, "latest_handover", 3, prefer_bedrooms=2
        )
    ]
    assert names[0] == "Two Bed Focus"
    assert names[-1] == "Wide Tower"


def test_more_mix_skips_shown_and_keeps_recent_best():
    rows = [
        {"name": f"P{index}", "handover_time": datetime(2006 + index, 1, 1), "price": 100_000 + index}
        for index in range(12)
    ]
    picked = _order_projects(
        rows, "more", 4, skip_names={"P10", "P11"}, rng=Random(1)
    )
    names = [row["name"] for row in picked]
    assert "P10" not in names
    assert "P11" not in names
    assert "P9" in names


def test_handover_parses_an_iso_date():
    parsed = _handover("2027-05-30T00:00:00")
    assert parsed is not None and parsed.year == 2027


def test_resolve_dubai_place_rejects_foreign_cities():
    assert is_outside_dubai("chennai") is True
    assert is_outside_dubai("Chennai, India") is True
    assert is_outside_dubai("abu dhabi") is True
    assert is_outside_dubai("Palm Jumeirah") is False
    assert resolve_dubai_place("chennai") is None
    assert resolve_dubai_place("mumbai") is None
    assert resolve_dubai_place("london") is None
    assert resolve_dubai_place("jvc") == "Jumeirah Village Circle"
    assert resolve_dubai_place("Palm Jumeirah") == "Palm Jumeirah"
    assert resolve_dubai_place("Business Bay") == "Business Bay"
    assert resolve_dubai_place("businessbay") == "Business Bay"
    assert resolve_dubai_place("palmjumeirah") == "Palm Jumeirah"
    assert is_outside_dubai("anantapur") is True
    assert resolve_dubai_place("anantapur") is None
    assert resolve_dubai_place("dubai") == "Dubai"


def test_resolve_dubai_place_reads_near_miss_spellings():
    assert resolve_dubai_place("busines bay") == "Business Bay"
    assert resolve_dubai_place("busnessbay") == "Business Bay"
    assert resolve_dubai_place("marinna") == "Dubai Marina"
    assert known_dubai_label("marinna") == "Dubai Marina"
    assert _focus("show me properties in marinna") == "Dubai Marina"
    assert resolve_dubai_place("chennai") is None
    assert resolve_dubai_place("mumbai") is None
    assert resolve_dubai_place("anantapur") is None
    assert resolve_dubai_place("jumera") is None
    assert known_dubai_label("jumera") is None


def test_place_browse_uses_latest_handover_then_more_mix(monkeypatch):
    from app.retrieval.hybrid.hybrid_retriever import HybridRetriever

    seen = {}

    def fake_projects(filters, limit=10, sort="cheapest", skip_names=None):
        seen["sort"] = sort
        seen["skip_names"] = set(skip_names or [])
        return [{"name": "Future Bay", "community": "Dubai Marina"}], 133

    monkeypatch.setattr("app.retrieval.propai_catalog.agent_projects", fake_projects)
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._browse_skip_names = set()
    retriever.structured_retriever = object()
    result = HybridRetriever._place_projects(retriever, "Dubai Marina")
    assert seen["sort"] == "latest_handover"
    assert seen["skip_names"] == set()
    assert result["project_total"] == 133

    retriever._browse_skip_names = {"pier 8", "ciel"}
    HybridRetriever._place_projects(retriever, "Dubai Marina")
    assert seen["sort"] == "more"
    assert seen["skip_names"] == {"pier 8", "ciel"}


def test_catalog_projects_do_not_copy_brochures_from_the_old_table(monkeypatch):
    """Prop-AI rows do not read brochure URLs from the old projects catalog."""
    from app.retrieval import propai_catalog

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("old projects table must not be queried")

    monkeypatch.setattr(propai_catalog, "fetch", fail_fetch)
    projects = [{"name": "Sobha One", "brochure_url": None}, {"name": "Hanover Square", "brochure_url": None}]
    propai_catalog._attach_brochures(projects)
    assert projects[0]["brochure_url"] is None
    assert projects[1]["brochure_url"] is None
