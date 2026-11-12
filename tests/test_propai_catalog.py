from app.listings.agent import _ID_TAG
from app.retrieval.propai_catalog import _agent_id, _focus, _handover, _photos, _public_photo, agent_projects


def test_focus_uses_the_alias_before_the_rest_of_the_sentence():
    assert _focus("Show me properties in Dubai Marina") == "Dubai Marina"
    assert _focus("projects in jvc") == "Jumeirah Village Circle"
    assert len(_focus("hi")) < 3


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


def test_catalog_projects_get_the_brochure_from_the_projects_table(monkeypatch):
    """Catalog rows have no brochure; 'Send me the brochure of Sobha One' must still find it."""
    from app.retrieval import propai_catalog

    monkeypatch.setattr(
        propai_catalog, "fetch",
        lambda sql, params: [{"name": "sobha one", "brochure_url": "https://example.test/sobha-one.pdf"}],
    )
    projects = [{"name": "Sobha One", "brochure_url": None}, {"name": "Hanover Square", "brochure_url": None}]
    propai_catalog._attach_brochures(projects)
    assert projects[0]["brochure_url"] == "https://example.test/sobha-one.pdf"
    assert projects[1]["brochure_url"] is None
