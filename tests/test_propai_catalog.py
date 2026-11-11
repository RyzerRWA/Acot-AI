from app.listings.agent import _ID_TAG
from app.retrieval.propai_catalog import (
    _agent_id,
    _focus,
    _handover,
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
