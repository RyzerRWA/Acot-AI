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
        "source_question": "looking for 2 bedroom apartments",
    }
    enriched = enrich_resolution_from_state(
        resolution,
        {"place": "Jumeirah Village Circle", "beds": 2},
    )
    assert enriched["action"] == "search"
    assert enriched["place"] == "Jumeirah Village Circle"
    assert not should_clarify(enriched, "looking for 2 bedroom apartments")


def test_dubai_citywide_ask_does_not_inherit_sticky_jvc():
    """'Projects in Dubai above 5M' must not become empty JVC 1-bed search."""
    from app.answers.turn_resolve import offline_resolve_turn

    question = "Show me projects in Dubai starting above AED 5 million"
    offline = offline_resolve_turn(question, None)
    enriched = enrich_resolution_from_state(
        offline,
        {
            "action": "search",
            "place": "Jumeirah Village Circle",
            "beds": 1,
            "budget_aed": 700_000,
            "ready_only": True,
            "source_question": "ready 1-bedroom in JVC under 700k",
        },
    )
    assert enriched["place"] is None
    assert enriched["beds"] is None
    assert enriched["budget_aed"] == 5_000_000
    assert enriched["action"] == "search"


def test_latest_sales_details_is_sales_list_not_cheapest():
    """Typo repair must not turn 'latest' into 'lowest' (edit distance 2)."""
    from app.retrieval.hybrid.question_text import repair_typed_question
    from app.answers.turn_resolve import offline_resolve_turn

    question = "latest sales details on The Woods at Sobha Sanctuary"
    assert "latest" in repair_typed_question(question).lower()
    assert "lowest" not in repair_typed_question(question).lower()
    offline = offline_resolve_turn(question, None)
    assert offline["action"] == "sales_list"
    assert offline.get("stat") is None
    assert "woods" in str(offline.get("place") or "").lower()


def test_sales_clarify_does_not_echo_sticky_budget():
    from app.answers.turn_resolve import offline_resolve_turn

    question = "Show me the recent transaction details"
    offline = offline_resolve_turn(question, None)
    enriched = enrich_resolution_from_state(
        offline,
        {
            "action": "search",
            "place": None,
            "budget_aed": 5_000_000,
            "source_question": "Show me projects in Dubai starting above AED 5 million",
        },
    )
    assert enriched["action"] == "sales_list"
    assert enriched.get("budget_aed") is None
    text = clarify_question_text(enriched, question)
    assert "5,000,000" not in text
    assert "under AED" not in text.lower()


def test_bare_sales_does_not_inherit_browse_place():
    """Bare 'transaction details' after recommend/search must not paste sticky JVC."""
    from app.answers.turn_resolve import offline_resolve_turn

    question = "Show me the recent transaction details"
    offline = offline_resolve_turn(question, None)
    enriched = enrich_resolution_from_state(
        offline,
        {
            "action": "recommend",
            "place": "Jumeirah Village Circle",
            "source_question": "best projects in jvc",
        },
    )
    assert enriched["action"] == "sales_list"
    assert enriched.get("place") is None
    assert should_clarify(enriched, question)


def test_sales_thread_still_inherits_place():
    """Same sales thread continues with sticky place when no new place is named."""
    resolution = {
        "action": "sales_list",
        "place": None,
        "confidence": 0.85,
        "source": "offline",
        "source_question": "Show me the recent transaction details",
    }
    enriched = enrich_resolution_from_state(
        resolution,
        {"action": "sales_list", "place": "Business Bay"},
    )
    assert enriched["place"] == "Business Bay"


def test_marina_family_recommend_does_not_inherit_sticky_50k():
    """'Best project in Dubai Marina for living family' must not invent AED 50k."""
    from app.answers.explain_contract import build_understood
    from app.answers.turn_resolve import offline_resolve_turn
    from app.retrieval.hybrid.question_text import repair_typed_question

    question = "suggest me the best project in Dubai Marina for living family"
    assert "living" in repair_typed_question(question).lower()
    assert "listing" not in repair_typed_question(question).lower()

    offline = offline_resolve_turn(question, None)
    assert offline["action"] == "recommend"
    assert offline["place"] == "Dubai Marina"
    assert offline.get("budget_aed") is None

    enriched = enrich_resolution_from_state(
        offline,
        {
            "action": "search",
            "place": "Jumeirah Village Circle",
            "budget_aed": 50_000,
            "beds": 1,
            "source_question": "studio under 50k",
        },
    )
    assert enriched.get("budget_aed") is None
    assert enriched["place"] == "Dubai Marina"

    understood = build_understood(
        place="Dubai Marina",
        beds=3,
        budget_aed=None,
        goal="family",
        intent="recommend",
    )
    assert "50,000" not in understood
    assert "dubai marina" in understood.lower()
    assert "projects in that community" in understood.lower()
    assert "rank communities on ready villa" not in understood.lower()


def test_bare_agents_does_not_inherit_browse_place():
    """Bare 'top agents' after Motor City browse must not paste sticky place."""
    from app.answers.turn_resolve import offline_resolve_turn

    question = "i want top agents"
    offline = offline_resolve_turn(question, None)
    assert offline["action"] == "agents"
    enriched = enrich_resolution_from_state(
        offline,
        {
            "action": "search",
            "place": "Motor City",
            "source_question": "show me projects in Motor City",
        },
    )
    assert enriched["action"] == "agents"
    assert enriched.get("place") is None
    assert should_clarify(enriched, question)
    text = clarify_question_text(enriched, question)
    assert "agent" in text.lower()
    # Sticky browse place must not appear in the Understood line.
    # ACOT QA AI-006: no empty "Understood: you are looking for your request." lead.
    assert "your request" not in text.lower()
    assert "in motor city" not in text.lower()


def test_agents_thread_still_inherits_place():
    resolution = {
        "action": "agents",
        "place": None,
        "confidence": 0.9,
        "source": "offline",
        "source_question": "show me the top ones",
    }
    enriched = enrich_resolution_from_state(
        resolution,
        {"action": "agents", "place": "Motor City"},
    )
    assert enriched["place"] == "Motor City"
    assert not should_clarify(enriched, "show me the top ones")


def test_sales_in_ac3_does_not_inherit_sticky_jvc(monkeypatch):
    from app.answers.turn_resolve import offline_resolve_turn
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(catalog, "place_named", lambda text: None)
    monkeypatch.setattr(
        catalog,
        "resolve_dubai_place",
        lambda place: "AC3" if str(place or "").strip().upper() == "AC3" else None,
    )

    question = "Show me sales details in AC3"
    offline = offline_resolve_turn(question, None)
    assert offline["place"] == "AC3"
    enriched = enrich_resolution_from_state(
        offline,
        {
            "action": "sales_list",
            "place": "Jumeirah Village Circle",
            "beds": 1,
            "budget_aed": 5_000_000,
            "source_question": "ok jvc",
        },
    )
    assert enriched["place"] == "AC3"
    assert enriched.get("beds") is None
    assert enriched.get("budget_aed") is None
    assert not should_clarify(enriched, question)


def test_clarify_turn_payload_has_no_fake_projects():
    route = offline_resolve_turn("I want 1 bedroom apartments")
    payload = clarify_turn_payload("I want 1 bedroom apartments", route)
    assert payload["use_fast"] is True
    assert payload["structured_summary"]["projects"] == []
    assert payload["structured_summary"]["clarify"] is True
    # Finalize reads prepared["documents"] — clarify must ship the same shape.
    assert payload.get("documents") == []
    assert "investment_analysis" in payload
    assert "community" in payload["fast_answer"].lower() or "area" in payload["fast_answer"].lower()


def test_clarify_payload_survives_finalize_shape():
    """Regression: ACOT processing error KeyError 'documents' on clarify turns."""
    from run import _ensure_prepared_turn, build_frontend_response

    route = offline_resolve_turn("looking for apartments")
    payload = clarify_turn_payload("looking for apartments", route)
    prepared = _ensure_prepared_turn(dict(payload))
    response = build_frontend_response(
        question=prepared["question"],
        standalone_question=prepared["standalone_question"],
        answer=prepared["fast_answer"],
        question_type=prepared["question_type"],
        structured_summary=prepared["structured_summary"],
        documents=prepared["documents"],
        investment_analysis=prepared.get("investment_analysis"),
    )
    assert response["status"] == "success"
    assert response["data"]["documents"] == []


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
