from app.memory.conversation_memory import ConversationMemory
from app.memory.dialogue_state import (
    empty_dialogue_state,
    normalize_dialogue_state,
    patch_dialogue_state,
    sale_filters_from_state,
    state_from_sale_filters,
)
from run import _remember_sale_query, _prior_sale_query, _transaction_turn


def test_empty_dialogue_state_has_expected_keys():
    state = empty_dialogue_state()
    assert state["action"] is None
    assert state["place"] is None
    assert state["beds"] is None
    assert state["ready_only"] is False
    assert state["exclude_names"] == []


def test_patch_dialogue_state_merges_slots():
    current = state_from_sale_filters(
        action="sales_count",
        place="Jumeirah Village Circle",
        beds=1,
        month=11,
        year=2026,
        count=47,
    )
    updated = patch_dialogue_state(
        current,
        {"action": "sales_list", "offset": 5, "last_result": {"count": 47}},
    )
    assert updated["place"] == "Jumeirah Village Circle"
    assert updated["beds"] == 1
    assert updated["month"] == 11
    assert updated["year"] == 2026
    assert updated["action"] == "sales_list"
    assert updated["offset"] == 5


def test_conversation_memory_set_and_clear_dialogue_state():
    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "sales_count",
        "place": "Business Bay",
        "beds": 2,
        "year": 2026,
        "month": 10,
        "last_result": {"count": 12},
    })
    state = memory.get_dialogue_state()
    assert state["place"] == "Business Bay"
    assert state["beds"] == 2
    assert memory.sale_query["place"] == "Business Bay"
    assert memory.sale_query["bedrooms"] == 2
    assert "dialogue_state" in memory.context()
    memory.clear()
    assert memory.get_dialogue_state()["place"] is None
    assert memory.sale_query is None


def test_sale_count_turn_writes_dialogue_state(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "count_sales",
        lambda **kwargs: {"count": 47, "first_sale": None, "last_sale": None},
    )
    monkeypatch.setattr(catalog, "resolve_dubai_place", lambda place: "Jumeirah Village Circle")
    monkeypatch.setattr(catalog, "place_named", lambda text: "Jumeirah Village Circle")

    memory = ConversationMemory(llm_client=None)
    out = _transaction_turn(
        "How many 1-bedroom apartments in JVC were sold in November 2026?",
        memory,
    )
    assert out is not None
    state = memory.get_dialogue_state()
    assert state["action"] == "sales_count"
    assert state["place"] == "Jumeirah Village Circle"
    assert state["beds"] == 1
    assert state["month"] == 11
    assert state["year"] == 2026
    assert state["last_result"]["count"] == 47
    prior = _prior_sale_query(memory)
    assert prior["place"] == "Jumeirah Village Circle"
    assert prior["bedrooms"] == 1
    assert prior["month"] == 11


def test_remember_sale_query_updates_dialogue_state():
    memory = ConversationMemory(llm_client=None)
    _remember_sale_query(
        memory,
        {
            "action": "sales_list",
            "place": "Dubai Marina",
            "bedrooms": 2,
            "listed": True,
            "source_question": "Show me recent sales in Dubai Marina",
        },
    )
    state = memory.get_dialogue_state()
    assert state["action"] == "sales_list"
    assert state["place"] == "Dubai Marina"
    assert state["beds"] == 2
    assert sale_filters_from_state(state)["bedrooms"] == 2


def test_normalize_rejects_unknown_action():
    state = normalize_dialogue_state({"action": "hack", "place": "JVC", "beds": "1"})
    assert state["action"] is None
    assert state["beds"] == 1
