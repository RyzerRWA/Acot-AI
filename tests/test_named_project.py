import asyncio

import run
from app.retrieval.propai_catalog import _numbered


def test_numbered_fields_skip_gaps_and_placeholders():
    payload = {"amenities_1": "Pool", "amenities_2": "Not Found", "amenities_4": "Gym", "amenities_5": None}
    assert _numbered(payload, "amenities") == ["Pool", "Gym"]


def test_compare_and_agent_asks_never_take_the_named_project_path():
    for question in ("Compare Serenz by Danube and Sartoria", "Top agents for Serenz by Danube"):
        assert asyncio.run(run._named_project_turn(question, None, None)) is None
