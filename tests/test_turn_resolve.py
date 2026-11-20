"""Phase 5: turn understanding without demo-sentence routes in run.py."""
import asyncio

from app.answers.turn_resolve import offline_resolve_turn, resolve_turn


def test_offline_project_count_any_wording():
    a = offline_resolve_turn("How many residential projects are there in Dubai?")
    b = offline_resolve_turn("What's the number of projects across Dubai?")
    assert a["action"] == "project_count"
    assert b["action"] == "project_count"
    assert (a["place"] or "Dubai") == "Dubai"


def test_offline_sales_count_and_list():
    count = offline_resolve_turn(
        "How many 1-bedroom apartments in JVC were sold in November 2026?"
    )
    assert count["action"] == "sales_count"
    assert count["beds"] == 1
    assert "Jumeirah" in (count["place"] or "") or count["place"] == "JVC"
    assert count["month"] == 11
    assert count["year"] == 2026

    listed = offline_resolve_turn("Show me recent sales in Business Bay")
    assert listed["action"] == "sales_list"
    assert listed["place"] == "Business Bay"


def test_offline_recommend_family_buy():
    route = offline_resolve_turn(
        "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    )
    assert route["action"] == "recommend"
    assert route["goal"] == "family"
    assert route["beds"] == 3
    assert route["budget_aed"] == 1_800_000


def test_offline_search_browse():
    route = offline_resolve_turn("Show me 2 bedroom apartments in Dubai Marina")
    assert route["action"] == "search"
    assert route["beds"] == 2
    assert route["place"] == "Dubai Marina"


def test_offline_incomplete_browse_is_clarify():
    route = offline_resolve_turn("looking for 2 bedroom apartments")
    assert route["action"] == "clarify"


def test_llm_resolve_unusual_wording_without_new_regex():
    async def fake_llm(_prompt):
        return """{
          "action": "sales_count",
          "goal": null,
          "place": "Jumeirah Village Circle",
          "beds": 1,
          "budget_aed": null,
          "month": 11,
          "year": 2026,
          "ready_only": false,
          "prefer_cheaper": false,
          "kind": null
        }"""

    resolution = asyncio.run(
        resolve_turn(
            "Could you check how many one-beds changed hands in JVC last November?",
            None,
            llm_generate=fake_llm,
        )
    )
    assert resolution is not None
    assert resolution["source"] == "llm"
    assert resolution["action"] == "sales_count"
    assert resolution["place"] == "Jumeirah Village Circle"
    assert resolution["beds"] == 1


def test_top_agents_for_place_is_agents_not_projects():
    """Regression: 'top agents for Dubai Hills' must not list projects."""
    from app.answers.turn_resolve import _apply_agents_override

    q = "Tell me the top agents for Dubai Hills Estate"
    offline = offline_resolve_turn(q)
    assert offline is not None
    assert offline["action"] == "agents"
    assert "Hills" in (offline.get("place") or "")

    # Even if a model wrongly chooses search, force agents.
    fixed = _apply_agents_override(
        {"action": "search", "place": "Dubai Hills Estate", "confidence": 0.9},
        q,
    )
    assert fixed["action"] == "agents"
    assert "Hills" in (fixed.get("place") or "")

    called = {"n": 0}

    async def boom(_prompt):
        called["n"] += 1
        raise AssertionError("clear agents ask must skip LLM")

    resolution = asyncio.run(resolve_turn(q, None, llm_generate=boom))
    assert called["n"] == 0
    assert resolution["action"] == "agents"


def test_clear_recommend_skips_llm_for_latency():
    """High-confidence offline recommend must not wait on an LLM hop."""
    called = {"n": 0}

    async def boom(_prompt):
        called["n"] += 1
        raise AssertionError("LLM should not run for clear offline recommend")

    resolution = asyncio.run(
        resolve_turn(
            "Which Dubai areas are best for families?",
            None,
            llm_generate=boom,
        )
    )
    assert called["n"] == 0
    assert resolution is not None
    assert resolution["source"] == "offline"
    assert resolution["action"] == "recommend"
    assert resolution["kind"] == "family_areas"


def test_understood_turn_executes_search_from_resolve(monkeypatch):
    import run as run_mod
    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    calls = {}

    def fake_agent_projects(filters, limit=10, sort="latest_handover", skip_names=None):
        calls["filters"] = filters
        return (
            [{
                "name": "Marina Gate",
                "community": "Dubai Marina",
                "price": 1_200_000,
                "entity_type": "project",
            }],
            1,
        )

    monkeypatch.setattr(
        "app.retrieval.propai_catalog.agent_projects",
        fake_agent_projects,
    )
    out = asyncio.run(
        run_mod._understood_turn(
            "Show me 2 bedroom apartments in Dubai Marina",
            memory,
            {},
        )
    )
    assert out is not None
    assert out["structured_summary"]["projects"]
    assert memory.get_dialogue_state()["action"] == "search"
    assert calls["filters"].get("bedrooms") == 2
