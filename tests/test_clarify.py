"""Phase 7: confidence gate asks instead of guessing."""
import asyncio

from app.answers.clarify import (
    CLARIFY_CONFIDENCE,
    clarify_question_text,
    clarify_turn_payload,
    enrich_resolution_from_state,
    score_confidence,
    should_clarify,
)
from app.answers.turn_resolve import offline_resolve_turn, resolve_turn


def test_incomplete_browse_should_clarify():
    route = offline_resolve_turn("looking for 2 bedroom apartments")
    assert route["action"] == "clarify"
    assert should_clarify(route, "looking for 2 bedroom apartments")
    ask = clarify_question_text(route, "looking for 2 bedroom apartments")
    assert "community" in ask.lower() or "area" in ask.lower()
    assert "2-bedroom" in ask or "2 bedroom" in ask.lower() or "bedroom" in ask.lower()


def test_complete_search_does_not_clarify():
    route = offline_resolve_turn("Show me 2 bedroom apartments in Dubai Marina")
    assert route["action"] == "search"
    route["confidence"] = score_confidence(route, "Show me 2 bedroom apartments in Dubai Marina")
    assert not should_clarify(route, "Show me 2 bedroom apartments in Dubai Marina")
    assert route["confidence"] >= CLARIFY_CONFIDENCE


def test_sales_without_place_clarifies():
    route = {
        "action": "sales_list",
        "place": None,
        "confidence": 0.9,
        "source": "offline",
    }
    assert should_clarify(route, "show me recent sales")
    assert "sales" in clarify_question_text(route).lower()


def test_sales_with_place_executes():
    route = offline_resolve_turn("Show me recent sales in Business Bay")
    route["confidence"] = score_confidence(route)
    assert route["action"] == "sales_list"
    assert not should_clarify(route)


def test_open_buy_recommend_does_not_force_area():
    route = {
        "action": "recommend",
        "goal": "open_buy",
        "kind": "open_best",
        "place": None,
        "beds": None,
        "budget_aed": None,
        "confidence": 0.85,
        "source": "offline",
    }
    assert not should_clarify(route)


def test_weak_recommend_clarifies():
    route = {
        "action": "recommend",
        "goal": "budget",
        "place": None,
        "beds": None,
        "budget_aed": None,
        "confidence": 0.9,
        "source": "llm",
    }
    assert should_clarify(route)
    assert "recommend" in clarify_question_text(route).lower()


def test_low_confidence_forces_clarify():
    route = {
        "action": "search",
        "place": "Dubai Marina",
        "beds": 2,
        "confidence": 0.4,
        "source": "llm",
    }
    assert should_clarify(route)


def test_state_place_upgrades_clarify_to_search():
    resolution = {
        "action": "clarify",
        "beds": 2,
        "place": None,
        "confidence": 0.4,
        "source": "offline",
    }
    enriched = enrich_resolution_from_state(
        resolution,
        {"place": "Jumeirah Village Circle", "beds": 2},
    )
    assert enriched["action"] == "search"
    assert enriched["place"] == "Jumeirah Village Circle"
    assert not should_clarify(enriched, "looking for 2 bedroom apartments")


def test_clarify_turn_payload_has_no_fake_projects():
    route = offline_resolve_turn("I want 1 bedroom apartments")
    payload = clarify_turn_payload("I want 1 bedroom apartments", route)
    assert payload["use_fast"] is True
    assert payload["structured_summary"]["projects"] == []
    assert payload["structured_summary"]["clarify"] is True
    assert "community" in payload["fast_answer"].lower() or "area" in payload["fast_answer"].lower()


def test_understood_turn_clarifies_incomplete_browse(monkeypatch):
    import run as run_mod
    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    out = asyncio.run(
        run_mod._understood_turn(
            "looking for 2 bedroom apartments",
            memory,
            {},
        )
    )
    assert out is not None
    assert out["use_fast"] is True
    assert out["structured_summary"]["projects"] == []
    assert "community" in out["fast_answer"].lower() or "area" in out["fast_answer"].lower()
    assert memory.get_dialogue_state()["action"] == "clarify"


def test_best_projects_in_marina_stays_place_scoped():
    q = "i am looking best projects in dubai marina"
    route = offline_resolve_turn(q)
    assert route["action"] == "recommend"
    assert route["place"] == "Dubai Marina"
    assert route["kind"] == "filtered"
    assert route["goal"] != "open_buy"
    assert not should_clarify(route, q)


def test_llm_open_best_with_marina_in_question_is_coerced():
    async def fake_llm(_prompt):
        return """{
          "action": "recommend",
          "goal": "open_buy",
          "place": null,
          "beds": null,
          "budget_aed": null,
          "month": null,
          "year": null,
          "ready_only": false,
          "prefer_cheaper": false,
          "kind": "open_best",
          "confidence": 0.9,
          "clarify_ask": null
        }"""

    resolution = asyncio.run(
        resolve_turn(
            "i am looking best projects in dubai marina",
            None,
            llm_generate=fake_llm,
        )
    )
    assert resolution["place"] == "Dubai Marina"
    assert resolution["kind"] == "filtered"
    assert resolution["goal"] != "open_buy"


def test_llm_low_confidence_clarify_ask():
    async def fake_llm(_prompt):
        return """{
          "action": "clarify",
          "goal": null,
          "place": null,
          "beds": null,
          "budget_aed": null,
          "month": null,
          "year": null,
          "ready_only": false,
          "prefer_cheaper": false,
          "kind": null,
          "confidence": 0.3,
          "clarify_ask": "Are you buying or renting, and in which Dubai area?"
        }"""

    resolution = asyncio.run(
        resolve_turn(
            "something nice maybe around there",
            None,
            llm_generate=fake_llm,
        )
    )
    assert resolution["action"] == "clarify"
    assert should_clarify(resolution)
    assert "buying or renting" in clarify_question_text(resolution)
