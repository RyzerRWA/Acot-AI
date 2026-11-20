import asyncio

from app.answers.followup_resolve import (
    apply_followup,
    looks_like_state_followup,
    resolve_followup,
)
from app.memory.conversation_memory import ConversationMemory
from app.memory.dialogue_state import state_from_sale_filters
from run import _dialogue_followup_turn, _transaction_turn


def test_looks_like_followup_requires_active_action():
    assert not looks_like_state_followup("show me more", {"action": None})
    assert looks_like_state_followup(
        "show me more",
        state_from_sale_filters(
            action="sales_list", place="Business Bay", beds=None
        ),
    )


def test_offline_show_me_more_continues_sales():
    state = state_from_sale_filters(
        action="sales_list",
        place="Jumeirah Village Circle",
        beds=1,
        month=11,
        year=2026,
        count=47,
    )

    async def _run():
        return await resolve_followup("show me more", state, llm_generate=None)

    resolution = asyncio.run(_run())
    assert resolution["kind"] == "continue"
    assert resolution["more"] is True
    assert resolution["action"] == "sales_list"
    merged = apply_followup(state, resolution)
    assert merged["place"] == "Jumeirah Village Circle"
    assert merged["beds"] == 1


def test_offline_what_about_place_patches_state():
    state = state_from_sale_filters(
        action="sales_list", place="Business Bay"
    )

    async def _run():
        return await resolve_followup(
            "what about Dubai Marina?", state, llm_generate=None
        )

    resolution = asyncio.run(_run())
    assert resolution["kind"] == "continue"
    assert resolution["patch"].get("place") == "Dubai Marina"
    merged = apply_followup(state, resolution)
    assert merged["place"] == "Dubai Marina"
    assert merged["action"] == "sales_list"


def test_full_new_browse_is_not_a_followup():
    state = state_from_sale_filters(
        action="sales_list", place="Business Bay"
    )

    async def _run():
        return await resolve_followup(
            "Show me projects in Palm Jumeirah", state, llm_generate=None
        )

    resolution = asyncio.run(_run())
    # Offline-first returns None for kind=new so the main turn resolver runs (no LLM hop).
    assert resolution is None or resolution.get("kind") == "new"


def test_top_agents_after_project_browse_is_not_a_followup():
    """Regression: agents chip must not re-list '200 projects in JVC'."""
    state = {
        "action": "search",
        "place": "Jumeirah Village Circle",
        "source_question": "show me projects in JVC",
    }

    async def _run():
        return await resolve_followup(
            "Tell me the top agents for Jumeirah Village Circle (JVC)",
            state,
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution is None or resolution.get("kind") == "new"


def test_transaction_chip_after_search_is_not_a_followup():
    state = {
        "action": "search",
        "place": "Motor City",
        "source_question": "show me projects in Motor City",
    }

    async def _run():
        return await resolve_followup(
            "Show me the recent transaction details",
            state,
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution is None or resolution.get("kind") == "new"


def test_short_place_tweak_still_continues_search():
    """'what about Marina?' after a browse may patch place — not a full new task."""
    state = {
        "action": "search",
        "place": "Jumeirah Village Circle",
        "source_question": "show me projects in JVC",
    }

    async def _run():
        return await resolve_followup(
            "what about Dubai Marina",
            state,
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution is not None
    assert resolution.get("kind") == "continue"
    assert resolution.get("patch", {}).get("place")


def test_ack_only_does_not_continue_recommend():
    state = {
        "action": "recommend",
        "place": "Dubai Marina",
        "goal": "family",
        "kind": "filtered",
    }

    async def _run():
        return await resolve_followup(
            "Ok that's cool",
            state,
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution is None or resolution.get("kind") == "new"


def test_family_kids_refinement_is_not_a_continue():
    state = {
        "action": "recommend",
        "kind": "family_areas",
        "goal": "family",
        "source_question": "Which Dubai areas are best for families?",
        "focus": {"kids": False, "avoid_towers": False},
    }

    async def _run():
        return await resolve_followup(
            "Best communities for a family with kids, not just the busiest towers.",
            state,
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution is None or resolution.get("kind") == "new"


def test_sales_data_freshness_is_not_sticky_recommend_explain():
    """After a recommend, meta 'most recent sales data' must not re-explain picks."""
    state = {
        "action": "recommend",
        "kind": "family_areas",
        "goal": "family",
        "source_question": "Which Dubai areas are best for families?",
        "budget_aed": 4000000,
    }

    async def _run(question):
        return await resolve_followup(question, state, llm_generate=None)

    for question in (
        "What is the recent, uh, data we have for this sales? Sales.",
        "what is themost recent data you have in the dubai sales data ?",
    ):
        resolution = asyncio.run(_run(question))
        assert resolution is None or resolution.get("kind") == "new", question


def test_show_more_followup_skips_llm():
    state = state_from_sale_filters(
        action="sales_list",
        place="Business Bay",
        beds=1,
    )
    called = {"n": 0}

    async def boom(_prompt):
        called["n"] += 1
        raise AssertionError("show me more must not call followup LLM")

    async def _run():
        return await resolve_followup("show me more", state, llm_generate=boom)

    resolution = asyncio.run(_run())
    assert called["n"] == 0
    assert resolution["kind"] == "continue"
    assert resolution["more"] is True


def test_dialogue_followup_turn_pages_sales(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    calls = []

    def fake_latest_sales(**kwargs):
        calls.append(kwargs)
        offset = int(kwargs.get("offset") or 0)
        name = "PAGE2" if offset else "PAGE1"
        return {
            "count": 1,
            "sales": [{
                "project": name,
                "date": "2026-11-03",
                "price": 1000000,
                "bedrooms": 1,
                "size_sqft": 700,
                "property_type": "Unit",
            }],
        }

    monkeypatch.setattr(catalog, "latest_sales", fake_latest_sales)
    memory = ConversationMemory(llm_client=None)
    first = _transaction_turn("Show me recent sales in Business Bay", memory)
    assert "PAGE1" in first["fast_answer"]

    out = asyncio.run(_dialogue_followup_turn("show me more", memory, {}))
    assert out is not None
    assert "PAGE2" in out["fast_answer"]
    assert calls[-1].get("offset", 0) > 0
    assert "I don't have a project called" not in out["fast_answer"]


def test_studios_after_split_deepens_studio_path_not_same_split():
    """After studio+villa split, 'studios' must deepen studios — not repeat the split."""
    import asyncio

    from app.answers.followup_resolve import apply_followup, resolve_followup
    from app.memory.dialogue_state import normalize_dialogue_state
    from run import _dialogue_followup_turn, _execute_recommend_from_state

    state = normalize_dialogue_state({
        "action": "recommend",
        "kind": "split",
        "goal": "open_buy",
        "budget_aed": 900_000,
        "source_question": (
            "The budget is AED 900,000. I need a 3-bedroom villa and, "
            "at the same time, only a studio apartment."
        ),
    })

    async def _resolve():
        return await resolve_followup("studios", state, llm_generate=None)

    resolution = asyncio.run(_resolve())
    assert resolution is not None
    assert resolution["kind"] == "continue"
    assert resolution["patch"]["beds"] == 0
    assert resolution["patch"]["kind"] == "filtered"
    new_state = apply_followup(state, resolution)
    assert new_state["kind"] == "filtered"
    assert new_state["beds"] == 0
    assert "studio" in (new_state.get("source_question") or "").lower()

    class _Memory:
        def __init__(self):
            self.dialogue_state = new_state
            self.active_candidates = []

        def get_dialogue_state(self):
            return dict(self.dialogue_state)

        def set_dialogue_state(self, value):
            self.dialogue_state = value

    memory = _Memory()
    # Should build a filtered studio route, not re-split on the conflict prior.
    from app.answers.intent_route import classify_intent

    prior_route = classify_intent(state["source_question"])
    assert prior_route["kind"] == "split"
    out = _execute_recommend_from_state("studios", memory, more=False, mode="patch")
    assert out is not None
    answer = out["fast_answer"]
    assert "Which path should I go deeper on" not in answer
    assert "Acot Recommend" in answer or "Understood" in answer or "studio" in answer.lower()


def test_developer_of_these_is_not_a_sales_continue():
    state = state_from_sale_filters(
        action="sales_list",
        place="Jumeirah Village Circle",
        beds=2,
        limit=5,
    )
    # With an active task we still *consult* the follow-up resolver...
    assert looks_like_state_followup(
        "Who is the developer of each of these projects?",
        state,
    )

    async def _run():
        return await resolve_followup(
            "Who is the developer of each of these projects?",
            state,
            llm_generate=None,
        )

    # ...but offline understanding marks it as a new ask (not a sales continue).
    resolution = asyncio.run(_run())
    assert resolution is None or resolution.get("kind") == "new"


def test_developer_followup_after_sales_lists_developers(monkeypatch):
    import app.retrieval.propai_catalog as catalog
    from run import _shown_attribute_response

    monkeypatch.setattr(
        catalog,
        "agent_projects",
        lambda filters, limit=1, sort="latest_handover", skip_names=None: (
            [{
                "name": filters.get("name"),
                "developer_name": {
                    "NEVA RESIDENCES": "Sobha",
                    "ALLURA RESIDENCES": "Danube",
                    "BINGHATTI AZURE": "Binghatti",
                }.get(str(filters.get("name") or "").upper(), "Unknown Dev"),
                "community": "Jumeirah Village Circle",
                "id": "x",
            }],
            1,
        ),
    )
    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "sales_list",
        "place": "Jumeirah Village Circle",
        "beds": 2,
        "limit": 5,
    })
    memory.last_sale_projects = [
        {"name": "NEVA RESIDENCES", "entity_type": "project"},
        {"name": "ALLURA RESIDENCES", "entity_type": "project"},
        {"name": "BINGHATTI AZURE", "entity_type": "project"},
    ]
    memory.active_candidates = []

    out = _shown_attribute_response(
        "Who is the developer of each of these projects?",
        memory,
        {},
        0,
        None,
    )
    assert out is not None
    answer = out["fast_answer"].lower()
    assert "neva" in answer
    assert "sobha" in answer
    assert "allura" in answer
    assert "danube" in answer
    assert "binghatti" in answer
    assert "recent transaction details" not in answer


def test_same_give_in_2_bedrooms_keeps_sales_place_and_limit(monkeypatch):
    """After 'last 5 … 1-bed in JVC', 'same give in 2 bedrooms' only changes beds."""
    import app.retrieval.propai_catalog as catalog

    calls = []

    def fake_latest_sales(**kwargs):
        calls.append(kwargs)
        bed = kwargs.get("bedrooms")
        return {
            "count": 2,
            "sales": [
                {
                    "project": f"PROJ-{bed}",
                    "date": "2026-11-03",
                    "price": 1_000_000 + int(bed or 0) * 1000,
                    "bedrooms": bed,
                    "size_sqft": 800,
                    "property_type": "Unit",
                }
            ],
        }

    monkeypatch.setattr(catalog, "latest_sales", fake_latest_sales)
    monkeypatch.setattr(
        catalog, "resolve_dubai_place", lambda place: "Jumeirah Village Circle"
    )
    monkeypatch.setattr(
        catalog, "place_named", lambda text: "Jumeirah Village Circle" if "jvc" in (text or "").lower() else None
    )

    memory = ConversationMemory(llm_client=None)
    first = _transaction_turn(
        "Show the last 5 sales of 1-bedroom apartments in JVC, with sale date and price",
        memory,
    )
    assert first is not None
    assert calls[0].get("limit") == 5
    assert calls[0].get("bedrooms") == 1
    assert memory.get_dialogue_state()["limit"] == 5
    assert memory.get_dialogue_state()["place"] == "Jumeirah Village Circle"

    async def _resolve():
        return await resolve_followup(
            "same give in 2 bedrooms",
            memory.get_dialogue_state(),
            llm_generate=None,
        )

    resolution = asyncio.run(_resolve())
    assert resolution["kind"] == "continue"
    assert resolution["action"] == "sales_list"
    assert resolution["patch"].get("beds") == 2

    out = asyncio.run(
        _dialogue_followup_turn("same give in 2 bedrooms", memory, {})
    )
    assert out is not None
    assert calls[-1].get("bedrooms") == 2
    assert calls[-1].get("limit") == 5
    assert calls[-1].get("place") == "Jumeirah Village Circle"
    assert "PROJ-2" in out["fast_answer"]
    assert "last 5" in out["fast_answer"].lower()
    assert memory.get_dialogue_state()["beds"] == 2
    assert memory.get_dialogue_state()["limit"] == 5


def test_project_count_show_me_lists_projects(monkeypatch):
    import app.retrieval.propai_catalog as catalog
    from run import _project_count_turn

    monkeypatch.setattr(catalog, "count_projects", lambda: 4110)
    monkeypatch.setattr(
        catalog,
        "agent_projects",
        lambda filters, limit=10, sort="latest_handover", skip_names=None: (
            [
                {
                    "name": "Marina Gate",
                    "community": "Dubai Marina",
                    "price": 1_200_000,
                    "entity_type": "project",
                },
                {
                    "name": "Bay Square",
                    "community": "Business Bay",
                    "price": 1_500_000,
                    "entity_type": "project",
                },
            ],
            4110,
        ),
    )
    memory = ConversationMemory(llm_client=None)
    count = _project_count_turn(
        "How many residential projects are there in Dubai?",
        memory,
    )
    assert count is not None
    assert memory.get_dialogue_state()["action"] == "project_count"

    async def _resolve():
        return await resolve_followup("show me", memory.get_dialogue_state())

    resolution = asyncio.run(_resolve())
    assert resolution["kind"] == "continue"
    assert resolution["action"] == "search"

    out = asyncio.run(_dialogue_followup_turn("show me", memory, {}))
    assert out is not None
    assert "Marina Gate" in out["fast_answer"] or out["structured_summary"]["projects"]
    assert memory.get_dialogue_state()["action"] == "search"
    assert "don't have any project" not in out["fast_answer"].lower()


def test_offline_why_these_explains_recommend_not_rerun():
    state = {
        "action": "recommend",
        "goal": "family",
        "place": "Business Bay",
        "beds": 3,
        "budget_aed": None,
        "source_question": (
            "ok i am looking for the best project in business bay for families"
        ),
    }

    async def _run():
        return await resolve_followup(
            "Why are these projects good for the family",
            state,
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution["kind"] == "continue"
    assert resolution["mode"] == "explain"
    assert resolution["more"] is False
    assert resolution["action"] == "recommend"


def test_dialogue_explain_followup_does_not_duplicate_search(monkeypatch):
    import app.answers.recommend as recommend_mod

    real_rank = recommend_mod.rank_projects

    def fake_rank(rows, **kwargs):
        kwargs["attach_sales"] = False
        return real_rank(rows, **kwargs)

    monkeypatch.setattr(recommend_mod, "rank_projects", fake_rank)
    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "recommend",
        "goal": "family",
        "place": "Business Bay",
        "beds": 3,
        "source_question": (
            "ok i am looking for the best project in business bay for families"
        ),
    })
    memory.active_candidates = [
        {
            "name": "Damac Towers by Paramount",
            "entity_type": "project",
            "community": "Business Bay",
            "price": 2_550_000,
            "sales_count": 25,
        },
        {
            "name": "Damac Maison Aykon City",
            "entity_type": "project",
            "community": "Business Bay",
            "price": 2_294_800,
            "sales_count": 6,
        },
        {
            "name": "Executive Towers - East Heights",
            "entity_type": "project",
            "community": "Business Bay",
            "price": 2_988_730,
            "sales_count": 16,
        },
    ]
    out = asyncio.run(
        _dialogue_followup_turn(
            "Why are these projects good for the family",
            memory,
            {},
        )
    )
    assert out is not None
    text = out["fast_answer"]
    assert "**Acot Recommend:**" in text
    assert "Why" in text
    assert "on-screen" in text.lower() or "ranked" in text.lower()
    # Must not restart as a fresh family search intro.
    assert "A family of 4 typically needs" not in text


def test_ok_show_me_after_count_via_followup_turn(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "count_sales",
        lambda **kwargs: {"count": 47, "first_sale": None, "last_sale": None},
    )
    monkeypatch.setattr(
        catalog, "resolve_dubai_place", lambda place: "Jumeirah Village Circle"
    )
    monkeypatch.setattr(
        catalog, "place_named", lambda text: "Jumeirah Village Circle"
    )
    monkeypatch.setattr(
        catalog,
        "latest_sales",
        lambda **kwargs: {
            "count": 1,
            "sales": [{
                "project": "HANOVER SQUARE",
                "date": "2026-11-03",
                "price": 700000,
                "bedrooms": 1,
                "size_sqft": 660,
                "property_type": "Unit",
            }],
        },
    )
    memory = ConversationMemory(llm_client=None)
    _transaction_turn(
        "How many 1-bedroom apartments in JVC were sold in November 2026?",
        memory,
    )
    assert memory.get_dialogue_state()["action"] == "sales_count"
    out = asyncio.run(_dialogue_followup_turn("ok show me", memory, {}))
    assert out is not None
    assert "HANOVER SQUARE" in out["fast_answer"]
    assert memory.get_dialogue_state()["action"] == "sales_list"
