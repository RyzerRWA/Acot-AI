from app.retrieval.hybrid.entity_resolver import EntityResolver
from app.retrieval.hybrid.hybrid_retriever import HybridRetriever
from app.retrieval.hybrid.query_planner import QueryPlanner
from app.retrieval.hybrid.question_text import (
    interpret_search,
    is_group_attribute_question,
    prepare_user_question,
    repair_typed_question,
)
from run import (
    _ambiguous_place_question,
    _apply_pending_place_choice,
    _build_comparison,
    _build_fast_structured_answer,
    _broker_band,
    _broker_place,
    _broker_turn,
    _classify_small_talk,
    _data_freshness_turn,
    _detect_response_type,
    _developer_rank_turn,
    _followup_questions,
    _is_broker_question,
    _is_fast_structured_question,
    _format_brokers,
    _format_transactions,
    _listing_question,
    _compare_shown_turn,
    _place_choice_turn,
    _project_count_turn,
    _project_is_full,
    _shortlist_recommend_turn,
    _shown_project_followup,
    _sql_project_browse_turn,
    _TRANSACTION,
    _transaction_subject,
    _transaction_turn,
    _cheapest_sale_turn,
    _small_talk_answer,
    build_frontend_response,
)


def _projects():
    return [
        {
            "name": "Serenz",
            "developer_name": "Danube",
            "community": "Jumeirah Village Circle",
            "price": 840000,
            "bedroom_min": 0,
            "bedroom_max": 3,
            "handover_time": "2029-09-30T07:57:00+00:00",
            "status": "ACTIVE",
        },
        {
            "name": "Samana Waves",
            "developer_name": "Sanama",
            "community": "Jumeirah Village Circle",
            "price": 483277,
            "bedroom_min": 0,
            "bedroom_max": 2,
            "status": "ACTIVE",
        },
        {
            "name": "Dawn by Binghatti",
            "developer_name": "Binghatti",
            "community": "Jumeirah Village Circle",
            "price": 734116,
            "bedroom_min": 0,
            "bedroom_max": 1,
            "status": "ACTIVE",
        },
        {
            "name": "Binghatti Etherea",
            "developer_name": "Binghatti",
            "community": "Jumeirah Village Circle",
            "price": 764999,
            "bedroom_min": 0,
            "bedroom_max": 2,
            "status": "ACTIVE",
        },
        {
            "name": "Azizi Ruby",
            "developer_name": "Azizi",
            "community": "Jumeirah Village Circle",
            "price": 618000,
            "bedroom_min": 0,
            "bedroom_max": 3,
            "status": "ACTIVE",
        },
    ]


def _summary(projects=None):
    return {
        "projects": projects if projects is not None else _projects(),
        "community": [],
        "sub_communities": [],
    }


def _planner():
    return QueryPlanner(use_llm=False)


def test_apartment_filter_does_not_drop_propai_projects_without_types():
    from app.retrieval.hybrid.hybrid_retriever import HybridRetriever

    projects = [
        {"name": "Marina Cove", "bedroom_min": 1, "bedroom_max": 4, "property_types": None},
        {"name": "Rove Home Dubai Marina", "bedroom_min": 0, "bedroom_max": 2, "property_types": []},
        {"name": "A Villa Only", "bedroom_min": 1, "bedroom_max": 3, "property_types": ["Villa"]},
    ]
    kept = HybridRetriever._apply_project_filters(
        projects,
        {"bedrooms": 1, "property_types": ["apartment"], "transaction_type": "rent"},
    )
    names = {row["name"] for row in kept}
    assert "Marina Cove" in names
    assert "Rove Home Dubai Marina" in names
    assert "A Villa Only" not in names


def test_browse_uses_fast_project_list():
    question = "Show me the projects in Jumeirah Village Circle"
    summary = _summary()
    plan = _planner()._rule_plan(question)

    assert plan.intent == "search"
    assert _is_fast_structured_question(
        question,
        "search",
        summary,
        [],
    )

    answer = _build_fast_structured_answer(question, summary)
    # Browse lists lead with the count — no duplicated Understood block.
    assert not answer.startswith("Understood:")
    assert "Understood:" not in answer
    assert "I found 5" in answer
    assert "Jumeirah Village Circle" in answer
    assert "AED 483,277" in answer
    assert "AED 840,000" in answer
    assert "Here are 1" not in answer
    assert "- **Serenz**" in answer
    assert "AED 840,000" in answer
    assert "Studios to 3 bedrooms" in answer
    assert "September 2029" in answer

    response = build_frontend_response(
        question=question,
        standalone_question=question,
        answer=answer,
        question_type="search",
        structured_summary=summary,
        documents=[],
    )
    assert response["response_type"] == "project_list"


def test_propai_browse_says_recent_sale_not_starting_price():
    question = "Show me projects in Dubai Land Residence Complex"
    summary = _summary([
        {
            "name": "Sky Line By Peace Homes",
            "developer_name": "PEACE HOMES REAL ESTATE L.L.C.",
            "price": 750146,
            "price_kind": "recent_sale",
            "source": "propai",
            "community": "Dubai Land Residence Complex",
            "bedroom_min": 0,
            "bedroom_max": 2,
            "handover_time": "2030-05-01",
            "last_sold_on": "2026-10-01",
        }
    ])
    summary["project_total"] = 128
    summary["catalog_place"] = "Dubai Land Residence Complex"
    answer = _build_fast_structured_answer(question, summary)
    assert "I found 128 projects in Dubai Land Residence Complex" in answer
    assert "Recent sale AED 750,146" in answer
    assert "Starting price" not in answer
    assert "Handover May 2030" in answer or "2030" in answer
    assert "Let me know if you need more details on any of these projects!" in answer


def test_budget_miss_browse_shows_closest_above():
    """Marina under 700k with no hits must show closest above-budget sales, not a dead end."""
    question = "Show only ready 1-bedroom apartments in Dubai Marina under AED 700,000"
    summary = _summary([
        {
            "name": "Pier 8",
            "developer_name": "EMAAR",
            "price": 741424,
            "price_kind": "recent_sale",
            "source": "propai",
            "community": "Dubai Marina",
            "bedroom_min": 1,
            "bedroom_max": 1,
            "handover_time": "2026-03-08",
            "last_sold_on": "2026-03-11",
        }
    ])
    summary["project_total"] = 1
    summary["catalog_place"] = "Dubai Marina"
    summary["budget_miss"] = True
    summary["asked_max_price"] = 700_000
    summary["lowest_price"] = 741_424
    answer = _build_fast_structured_answer(question, summary)
    assert "Understood:" not in answer
    assert "don't have" in answer.lower() or "do not have" in answer.lower()
    assert "700,000" in answer
    assert "741,424" in answer
    assert "Pier 8" in answer
    assert "related options" in answer.lower()
    assert "Try a higher budget" not in answer


def test_sql_browse_marina_budget_miss_and_jvc_hit():
    """Live path: Marina under 700k → related above; JVC under 700k → exact hits."""
    from run import _sql_project_browse_turn

    marina = _sql_project_browse_turn(
        "Show only ready 1-bedroom apartments in dubai marina under AED 700,000",
        None,
    )
    assert marina is not None
    msum = marina["structured_summary"]
    assert msum.get("budget_miss") is True
    assert len(msum.get("projects") or []) >= 1
    assert "related options" in marina["fast_answer"].lower()
    assert "Try a higher budget" not in marina["fast_answer"]
    assert "Pier 8" in marina["fast_answer"] or "741" in marina["fast_answer"]

    jvc = _sql_project_browse_turn(
        "Show only ready 1-bedroom apartments in JVC under AED 700,000",
        None,
    )
    assert jvc is not None
    jsum = jvc["structured_summary"]
    assert not jsum.get("budget_miss")
    assert len(jsum.get("projects") or []) >= 1
    assert "under AED 700,000" in jvc["fast_answer"]
    assert "don't have" not in jvc["fast_answer"].lower()


def test_ready_browse_omits_ancient_handover_years():
    """Ready stock must not lead with decade-old handover years."""
    question = "Show only ready 1-bedroom apartments in JVC under AED 700,000"
    summary = _summary([
        {
            "name": "Hanover Square",
            "developer_name": "ACW HOLDING LTD.",
            "price": 635000,
            "price_kind": "recent_sale",
            "source": "propai",
            "community": "Jumeirah Village Circle",
            "bedroom_min": 1,
            "bedroom_max": 1,
            "handover_time": "2015-12-01",
            "last_sold_on": "2026-10-15",
        }
    ])
    summary["project_total"] = 1
    summary["catalog_place"] = "Jumeirah Village Circle"
    answer = _build_fast_structured_answer(question, summary)
    assert "Understood:" not in answer
    assert answer.startswith("Here are 1 ready 1-bedroom option")
    assert "Recent sale AED 635,000" in answer
    assert "2026-10-15" in answer
    assert "Handover December 2015" not in answer
    assert "Handover November 2010" not in answer


def test_browse_more_says_here_are_more():
    question = (
        "Show me properties in Dubai Marina. "
        "Show additional options for the same request, not the same list again."
    )
    summary = _summary()
    summary["project_total"] = 133
    summary["catalog_place"] = "Dubai Marina"
    answer = _build_fast_structured_answer(question, summary)
    assert "I found 133 projects in Dubai Marina." in answer
    assert "Here are 5 more of them" in answer


def test_investment_answer_with_projects_shows_cards():
    question = "Find me a good investment property in Business Bay"
    assert _detect_response_type(question, "investment", _summary()) == "project_list"
    assert _detect_response_type(
        question, "investment", _summary(projects=[])
    ) == "investment_analysis"


def test_snapshot_fact_with_projects_shows_cards():
    question = "what about Arabian Ranches only — typical 3-bed and 4-bed sale?"
    summary = _summary()
    assert _detect_response_type(question, "fact", summary) == "project_list"
    assert _detect_response_type(question, "search", summary) == "project_list"
    response = build_frontend_response(
        question=question,
        standalone_question=question,
        answer="Arabian Ranches, recorded sales by bedroom",
        question_type="search",
        structured_summary=summary,
        documents=[],
    )
    assert response["response_type"] == "project_list"
    assert response["data"]["projects"]


def test_agent_comparison_shows_comparison_even_without_compare_words():
    # The agent sets question_type "comparison" when it used compare_projects.
    question = "Which is better for a family, Serenz or Sobha One?"
    assert _detect_response_type(question, "comparison", _summary()) == "project_comparison"
    assert _detect_response_type(question, "search", _summary()) == "project_comparison"
    one = _summary(projects=_projects()[:1])
    assert _detect_response_type(question, "comparison", one) != "project_comparison"


def test_price_follow_up_is_written_information():
    question = (
        "What are the starting prices of the projects in "
        "Jumeirah Village Circle?"
    )
    summary = _summary()
    plan = _planner()._rule_plan(question)

    assert plan.intent == "information"
    assert plan.needs_documents is False
    # Field asks with rows already retrieved use the grounded template.
    assert _is_fast_structured_question(
        question,
        plan.intent,
        summary,
        [],
    )
    assert _detect_response_type(
        question,
        plan.intent,
        summary,
    ) == "general"


def test_handover_follow_up_stays_on_the_current_projects():
    rewritten = (
        "What are the handover dates for the projects in "
        "Jumeirah Village Circle?"
    )
    shape = interpret_search(rewritten)
    assert shape["kind"] == "community"
    assert shape["community"] == "jumeirah village circle"
    assert is_group_attribute_question("What are their handover dates?")
    assert is_group_attribute_question(rewritten)
    assert not is_group_attribute_question(
        "Show me Azizi projects in Meydan"
    )

    plan = _planner()._rule_plan(rewritten)
    summary = _summary()
    assert plan.intent == "information"
    assert _is_fast_structured_question(
        rewritten,
        plan.intent,
        summary,
        [],
    )

    class _CommunityPlan:
        entity_type = "community"

    scope = HybridRetriever._requested_data_scope(
        "What are their handover dates in JVC?",
        query_plan=_CommunityPlan(),
    )
    assert "projects" in scope


def test_family_community_question_is_a_catalog_lookup():
    question = "Which communities suit families?"
    plan = _planner()._rule_plan(question)
    assert HybridRetriever._asks_for_catalog(question)
    ranked = HybridRetriever._prefer_matching_communities(
        [
            {"name": "Business Bay", "knowledge_text": "Offices and towers."},
            {
                "name": "Arabian Ranches",
                "knowledge_text": "Villas and family communities with schools.",
            },
        ],
        question,
    )
    assert ranked[0]["name"] == "Arabian Ranches"
    assert plan.needs_structured_data is True


def test_yield_community_question_keeps_projects():
    question = "Best communities for rental yield"
    plan = _planner()._rule_plan(question)
    scope = HybridRetriever._requested_data_scope(
        question,
        query_plan=plan,
    )
    assert plan.intent == "investment"
    assert "communities" in scope
    assert "projects" in scope


def test_bedroom_follow_up_uses_fast_path_when_rows_exist():
    question = "Which of these have 2-bedroom options?"
    plan = _planner()._rule_plan(question)

    assert plan.intent == "information"
    assert _is_fast_structured_question(
        question,
        plan.intent,
        _summary(),
        [],
    )


def test_comparison_uses_fast_path_when_rows_exist():
    question = (
        "Compare the first and second ones based on price, "
        "bedroom range, and handover date."
    )
    summary = _summary()
    plan = _planner()._rule_plan(question)

    assert plan.intent == "comparison"
    assert _is_fast_structured_question(
        question,
        plan.intent,
        summary,
        [],
    )
    assert _detect_response_type(
        question,
        plan.intent,
        summary,
    ) == "project_comparison"


def test_comparison_answer_always_includes_a_side_by_side_table():
    question = "Compare Binghatti Etherea and Serenz. Which is best?"
    summary = {
        "comparison": True,
        "comparison_entities": [
            {
                "resolved_name": "Binghatti Etherea",
                "data": {
                    "projects": [{
                        "price": 800000,
                        "bedroom_min": 0,
                        "bedroom_max": 2,
                        "size_min": 600,
                        "size_max": 600,
                    }],
                },
            },
            {
                "resolved_name": "Serenz",
                "data": {
                    "projects": [{
                        "price": 800000,
                        "bedroom_min": 0,
                        "bedroom_max": 2,
                        "size_min": 600,
                        "size_max": 600,
                    }],
                },
            },
        ],
    }

    comparison = _build_comparison(summary)

    assert comparison is not None
    assert "| Factor | Binghatti Etherea | Serenz |" in comparison["answer"]
    assert "| Price | AED 800,000 | AED 800,000 |" in comparison["answer"]
    assert "| Size | 600 sqft | 600 sqft |" in comparison["answer"]
    assert "| Score | 50% | 50% |" in comparison["answer"]
    assert "do not favor either one" in comparison["answer"]
    assert "**Acot Recommend:**" in comparison["answer"]

    response = build_frontend_response(
        question=question,
        standalone_question=question,
        answer=comparison["answer"],
        question_type="comparison",
        structured_summary=summary,
        documents=[],
    )
    assert response["response_type"] == "comparison"
    assert response["answer"] == comparison["answer"]


def test_comparison_without_comparable_fields_still_shows_table():
    summary = {
        "comparison_entities": [
            {
                "resolved_name": "Binghatti Etherea",
                "data": {"projects": [{"name": "Binghatti Etherea"}]},
            },
            {
                "resolved_name": "Serenz",
                "data": {"projects": [{"name": "Serenz"}]},
            },
        ],
    }

    comparison = _build_comparison(summary)

    assert comparison is not None
    assert "| Factor | Binghatti Etherea | Serenz |" in comparison["answer"]
    assert "| Score | 50% | 50% |" in comparison["answer"]
    assert "useful options to review" in comparison["answer"]
    assert "**Acot Recommend:**" in comparison["answer"]
    assert "can't recommend" not in comparison["answer"].lower()


def test_named_compare_serenz_aura_never_clarifies_for_area_beds_budget():
    """Demo-critical: named project compare must not ask for area/beds/budget."""
    import asyncio

    from app.answers.clarify import should_clarify
    from app.answers.turn_resolve import resolve_turn
    from app.memory.conversation_memory import ConversationMemory
    from app.retrieval.hybrid.question_text import comparison_pair
    from run import _understood_turn

    questions = [
        "Compare Serenz by Danube and Aura Heights",
        "Serenz by Danube and Aura Heights",
    ]
    for question in questions:
        assert comparison_pair(question) == ("Serenz by Danube", "Aura Heights")
        assert should_clarify(
            {
                "action": "recommend",
                "goal": "compare",
                "kind": "shortlist",
                "confidence": 0.95,
            },
            question,
        ) is False

        async def _fake_llm(_prompt):
            return (
                '{"action":"recommend","kind":"filtered","goal":"compare",'
                '"confidence":0.95}'
            )

        memory = ConversationMemory(llm_client=None)
        resolved = asyncio.run(
            resolve_turn(question, memory, llm_generate=_fake_llm)
        )
        assert resolved is None
        understood = asyncio.run(_understood_turn(question, memory, {}))
        assert understood is None


def test_bare_project_pair_plans_as_comparison():
    """'A and B' retry after a clarify must reach the comparison retriever."""
    from app.retrieval.hybrid.query_planner import QueryPlanner

    planner = QueryPlanner.__new__(QueryPlanner)
    plan = planner._rule_plan("Serenz by Danube and Aura Heights")
    assert plan.needs_comparison is True
    assert plan.intent == "comparison"
    assert plan.entity_candidates == ["Serenz by Danube", "Aura Heights"]


def test_transaction_details_after_single_project_answer_uses_that_project():
    """Agent answers may leave no active_candidates; tx details still mean that project."""
    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    memory.active_entities["project"] = "Serenz by Danube"
    memory.active_entities["community"] = "Jumeirah Village Circle (JVC)"
    memory.turns = [
        {
            "user_question": "What is the starting price of Serenz by Danube?",
            "candidate_names": [],
        }
    ]
    subject = _transaction_subject("Show me the recent transaction details", memory)
    assert subject and subject[0]["name"] == "Serenz by Danube"
    # A named place is still its own ask.
    assert _transaction_subject("Show me recent sales in Business Bay", memory) is None


def test_fresh_marina_buy_ask_is_not_patched_into_old_task():
    import asyncio

    from app.answers.followup_resolve import resolve_followup
    from app.answers.turn_resolve import resolve_turn
    from app.memory.conversation_memory import ConversationMemory

    question = (
        "So I need few properties to buy and in Dubai Marina and also my "
        "budget is around 3 million."
    )
    for state in (
        {"action": "clarify"},
        {"action": "sales_list", "place": "Jumeirah Village Circle", "beds": 0},
        {"action": "search", "place": "Jumeirah Village Circle", "beds": 1},
    ):
        memory = ConversationMemory(llm_client=None)
        memory.set_dialogue_state(state)
        assert asyncio.run(
            resolve_followup(question, memory.get_dialogue_state(), llm_generate=None)
        ) is None
        resolved = asyncio.run(resolve_turn(question, memory, llm_generate=None))
        assert resolved["action"] == "recommend"
        assert resolved["place"] == "Dubai Marina"
        assert resolved["budget_aed"] == 3_000_000
        # Stale studio/1-bed from the old task must not leak into the new ask.
        assert resolved["beds"] is None


def test_named_compare_does_not_prefer_handover_only_over_recorded_price():
    """Serenz with price/beds must beat Aura Heights that only has earlier handover."""
    summary = {
        "comparison": True,
        "comparison_entities": [
            {
                "resolved_name": "Serenz by Danube",
                "data": {
                    "projects": [{
                        "name": "Serenz by Danube",
                        "price": 1_187_220,
                        "bedroom_min": 1.5,
                        "bedroom_max": 1.5,
                        "handover_time": "2029-09-01",
                    }],
                },
            },
            {
                "resolved_name": "Aura Heights",
                "data": {
                    "projects": [{
                        "name": "Aura Heights",
                        "handover_time": "2029-04-01",
                    }],
                },
            },
        ],
    }

    comparison = _build_comparison(summary)

    assert comparison is not None
    answer = comparison["answer"]
    assert "| Factor | Serenz by Danube | Aura Heights |" in answer
    assert "AED 1,187,220" in answer
    assert "Not recorded" in answer
    assert "**Acot Recommend:** Serenz by Danube" in answer
    assert "**Acot Recommend:** Aura Heights" not in answer
    assert "Aura Heights scores 100%" not in answer
    assert "Stands out for" not in answer
    assert "1,187,220" in answer.split("**Acot Recommend:**")[-1]
    assert "Aura Heights" in answer.split("**Acot Recommend:**")[-1]
    # Fuller record should lead the scores.
    serenz_score = next(
        side["score"] for side in comparison["sides"] if side["name"] == "Serenz by Danube"
    )
    aura_score = next(
        side["score"] for side in comparison["sides"] if side["name"] == "Aura Heights"
    )
    assert serenz_score > aura_score


def test_rental_comparison_escapes_title_pipes_and_explains_tradeoff():
    summary = {
        "comparison_entities": [
            {
                "resolved_name": "Villa | Sea View",
                "data": {
                    "listings": [{
                        "title": "Villa | Sea View",
                        "price_aed": 260000,
                        "rent_frequency": "yearly",
                        "beds": 4,
                        "size_sqft": 2920,
                    }],
                },
            },
            {
                "resolved_name": "Villa | Garden",
                "data": {
                    "listings": [{
                        "title": "Villa | Garden",
                        "price_aed": 269999,
                        "rent_frequency": "yearly",
                        "beds": 4,
                        "size_sqft": 4023,
                    }],
                },
            },
        ],
    }

    comparison = _build_comparison(summary)

    assert comparison is not None
    assert "| Factor | Villa \\| Sea View | Villa \\| Garden |" in comparison["answer"]
    assert "| Yearly rent | AED 260,000 | AED 269,999 |" in comparison["answer"]
    assert "| Score | 50% | 50% |" in comparison["answer"]
    assert "Conclusion: The comparison is balanced." in comparison["answer"]
    assert "**Acot Recommend:** Villa | Sea View and Villa | Garden" in comparison["answer"]
    assert "Stands out for" not in comparison["answer"]
    assert "260,000" in comparison["answer"].split("**Acot Recommend:**")[-1] or (
        "lower yearly rent" in comparison["answer"].lower()
    )
    assert "more space" in comparison["answer"].lower() or "4,023" in comparison["answer"]


def test_comparison_table_includes_every_selected_listing_in_order():
    names = ["Rental one", "Rental two", "Rental three"]
    summary = {
        "comparison_entities": [
            {
                "resolved_name": name,
                "data": {
                    "listings": [{
                        "title": name,
                        "price_aed": 58_000 + index * 1_000,
                        "rent_frequency": "yearly",
                        "beds": 2,
                        "size_sqft": 800 + index * 50,
                    }],
                },
            }
            for index, name in enumerate(names)
        ],
    }

    comparison = _build_comparison(summary)

    assert comparison is not None
    assert len(comparison["sides"]) == 3
    assert [side["name"] for side in comparison["sides"]] == names
    assert "| Factor | Rental one | Rental two | Rental three |" in comparison["answer"]
    assert "| Yearly rent | AED 58,000 | AED 59,000 | AED 60,000 |" in comparison["answer"]
    assert "**Acot Recommend:**" in comparison["answer"]


def test_small_talk_skips_projects():
    cases = {
        "Hi": "greeting",
        "Who are you?": "identity",
        "What can you do?": "capabilities",
    }

    for question, kind in cases.items():
        assert _classify_small_talk(question) == kind
        answer = _small_talk_answer(kind)
        assert "Found" not in answer
        response = build_frontend_response(
            question=question,
            standalone_question=question,
            answer=answer,
            question_type="general",
            structured_summary=_summary(projects=[]),
            documents=[],
        )
        assert response["response_type"] == "general"
        assert response["data"]["projects"] == []


def test_greeting_variants_are_small_talk():
    for text in (
        "Good morning!", "good morning acot", "Good morning sir", "Morning",
        "gm", "good mornin", "Good afternoon everyone", "how are you",
        "Assalamu alaikum", "good morning, how are you",
        "Hello. So, uh,", "hi there uh",
    ):
        assert _classify_small_talk(text) == "greeting", text
    assert _classify_small_talk("good night") == "bye"
    for text in (
        "good projects in Marina", "good morning show me projects",
        "how much is a 2 bedroom", "morning view apartments",
    ):
        assert _classify_small_talk(text) is None, text


def test_general_plan_skips_documents_without_brochure_terms():
    plan = _planner()._rule_plan("Hello. So, uh,")
    assert plan.intent == "general"
    assert plan.needs_documents is False
    assert QueryPlanner._is_simple_structured_plan(plan) is True


def test_sql_project_browse_turn_returns_fast_answer():
    from run import _sql_project_browse_turn

    out = _sql_project_browse_turn(
        "Show me projects in Jumeirah Village Circle.",
        None,
    )
    assert out is not None
    assert out["use_fast"] is True
    assert "I found" in (out["fast_answer"] or "")
    assert "Acot Recommend" not in (out["fast_answer"] or "")
    assert out["structured_summary"]["projects"]


def test_phase3_sql_browse_defers_buy_to_recommend():
    from run import _sql_project_browse_turn

    assert _sql_project_browse_turn(
        "I want a 3-bedroom apartment in JVC for a family of 4 to buy around AED 1.8M.",
        None,
    ) is None
    assert _sql_project_browse_turn(
        "Show me the best apartments in Dubai Marina",
        None,
    ) is None


def test_phase3_recommend_turn_has_winner_and_why(monkeypatch):
    from app.answers import intent_turns

    rows = [
        {
            "name": "Alpha Towers",
            "community": "Jumeirah Village Circle",
            "price": 1_750_000,
            "bedrooms": "3",
            "size_sqft": 1400,
            "sales_count": 12,
        },
        {
            "name": "Beta Residences",
            "community": "Jumeirah Village Circle",
            "price": 1_820_000,
            "bedrooms": "3",
            "size_sqft": 1500,
            "sales_count": 8,
        },
    ]
    monkeypatch.setattr(
        intent_turns,
        "_recommend_pool",
        lambda filters, skip_names=None: (rows, []),
    )
    turn = intent_turns.recommend_data_turn(
        "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    )
    assert turn is not None
    text = turn["fast_answer"]
    assert "**Acot Recommend:**" in text
    assert "**Why" in text
    assert turn.get("structured_summary", {}).get("projects")


def test_greeting_uses_the_account_name():
    assert _small_talk_answer("greeting", "Priya") == (
        "Hello Priya, I'm ACOT, a Dubai real estate assistant. How can I help you?"
    )
    assert _small_talk_answer("greeting").startswith("Hello, I'm ACOT")


def test_greeting_plus_search_stays_search():
    question = "Hi, show me projects in Jumeirah Village Circle"
    assert _classify_small_talk(question) is None
    assert _planner()._rule_plan(question).intent == "search"


def test_close_community_spelling_scores_as_a_match():
    score = EntityResolver._score("Ai Jaddaf", "Al Jaddaf")
    assert score >= 0.70

    weak = EntityResolver._score("Ai Jaddaf", "Aljada")
    assert weak < 0.70


def test_community_suffix_does_not_steal_a_longer_project_name():
    assert EntityResolver._community_is_suffix_of_longer_name(
        "i want full detail empire gardens",
        "THE GARDENS",
    )
    assert not EntityResolver._community_is_suffix_of_longer_name(
        "Tell me about Dubai Marina",
        "Dubai Marina",
    )


def test_project_detail_turn_uses_the_on_screen_project(monkeypatch):
    from run import _project_detail_turn

    class _Memory:
        active_candidates = [
            {
                "name": "Empire Gardens",
                "entity_type": "project",
                "price": 939777,
                "price_kind": "recent_sale",
                "source": "propai",
                "community": "Dubai Land Residence Complex",
                "developer_name": "EMPIRE GARDENS REAL ESTATE DEVELOPMENT L.L.C.",
                "sales_count": 131,
                "size_sqft": 558,
            }
        ]

    monkeypatch.setattr(
        "app.retrieval.propai_catalog.agent_projects",
        lambda *args, **kwargs: ([], 0),
    )
    turn = _project_detail_turn("i want full detail Empire Gardens", _Memory())
    assert turn is not None
    assert "Empire Gardens" in turn["fast_answer"]
    assert "couldn't retrieve" not in turn["fast_answer"].lower()
    assert "THE GARDENS" not in turn["fast_answer"]
    assert turn["structured_summary"]["projects"][0]["name"] == "Empire Gardens"


def test_short_project_name_does_not_beat_the_place_it_sits_inside():
    prefix = EntityResolver._score(
        "Show me projects in Dubai Land Residence Complex",
        "Dubai Land",
    )
    place = EntityResolver._score(
        "Dubai Land Residence Complex",
        "Dubai Land Residence Complex",
    )
    assert prefix < 0.90
    assert place == 1.0
    assert EntityResolver._prefer_community(
        "Show me projects in Dubai Land Residence Complex",
        {"name": "Dubai Land", "score": prefix},
        {"name": "Dubai Land Residence Complex", "score": place},
    )


def test_developer_question_resolves_to_the_developer():
    projects = [
        {"name": "Dawn by Binghatti", "developer_name": " Binghatti"},
        {"name": "Binghatti Etherea", "developer_name": " Binghatti"},
        {"name": "Serenz", "developer_name": "Danube"},
    ]
    match = EntityResolver._resolve_developer(
        EntityResolver,
        ["SHOW ME ALL BINGHATTI PROJECTS"],
        projects,
    )
    assert match is not None
    assert match["name"] == "Binghatti"
    assert match["count"] == 2


def test_transaction_followup_uses_the_project_on_screen():
    class _Memory:
        active_candidates = [{
            "name": "15 Northside",
            "entity_type": "project",
            "price": 1072500,
            "price_kind": "recent_sale",
        }]

    found = _transaction_subject("show me the recent transcation details", _Memory())
    assert found and found[0]["name"] == "15 Northside"
    assert _transaction_subject("Show me projects in Business Bay", _Memory()) is None


def test_latest_transactions_work_without_projects_on_screen(monkeypatch):
    sales = {
        "count": 1,
        "sales": [{
            "project": "Mayfair Tower",
            "date": "2026-10-19",
            "price": 1000000,
            "bedrooms": 1,
            "size_sqft": 807,
        }],
    }

    def fake_latest(**kwargs):
        captured.update(kwargs)
        return sales

    captured = {}
    monkeypatch.setattr("app.retrieval.propai_catalog.latest_sales", fake_latest)
    monkeypatch.setattr(
        "app.retrieval.propai_catalog.resolve_dubai_place",
        lambda place: None if not place else {
            "busines bay": "Business Bay",
            "business bay": "Business Bay",
        }.get(str(place).strip().lower()),
    )
    monkeypatch.setattr("app.retrieval.propai_catalog.is_outside_dubai", lambda place: False)
    monkeypatch.setattr(
        "app.retrieval.propai_catalog.recent_sales",
        lambda *args, **kwargs: {"count": 0, "sales": []},
    )

    turn = _transaction_turn("show me the latest transcations", None)
    assert turn is not None
    assert turn["use_fast"] is True
    assert "Mayfair Tower" in turn["fast_answer"]
    assert "couldn't find an exact match" not in turn["fast_answer"].lower()
    assert captured.get("place") in (None, "")
    assert captured.get("limit") == 8

    captured.clear()
    area = _transaction_turn("show me the latest transcations in busines bay", None)
    assert area is not None
    assert "Mayfair Tower" in area["fast_answer"]
    assert captured.get("place") == "Business Bay"

    assert _transaction_turn("Show me projects in Business Bay", None) is None


def test_rental_browse_drops_duplicates_and_formats_size():
    listings = [
        {
            "title": "Spacious | Prime Location | Marina View",
            "sub_community": "Marina Diamonds",
            "community": "Dubai Marina",
            "agency_name": "AEON & Trisl Real Estate",
            "price_aed": 65000,
            "rent_frequency": "yearly",
            "beds": 1,
            "size_sqft": 714.0,
        },
        {
            "title": "BIG LAYOUT I GOOD BUILDING I NEAR TO TRAM",
            "sub_community": "Elite Residence",
            "community": "Dubai Marina",
            "agency_name": "Banke International Properties - Commercial",
            "price_aed": 74999,
            "rent_frequency": "yearly",
            "beds": 1,
            "size_sqft": 1160.0,
        },
        {
            "title": "BIG LAYOUT I GOOD BUILDING I NEAR TO TRAM",
            "sub_community": "Elite Residence",
            "community": "Dubai Marina",
            "agency_name": "Banke International Properties",
            "price_aed": 74999,
            "rent_frequency": "yearly",
            "beds": 1,
            "size_sqft": 1160.0,
        },
    ]
    summary = {
        "listings": listings,
        "listing_total": 69,
        "catalog_place": "Dubai Marina",
        "projects": [],
        "community": [],
        "sub_communities": [],
    }
    answer = _build_fast_structured_answer(
        "Show me 1 bedroom apartments for rent in Dubai Marina.",
        summary,
    )
    assert "I found 69 rental listings in Dubai Marina." in answer
    assert "Here are 2 of them" in answer
    assert "714.0" not in answer
    assert "714 sqft" in answer
    assert "Marina Diamonds" in answer
    assert "AEON" not in answer
    assert answer.count("BIG LAYOUT") == 1
    assert len(summary["listings"]) == 2


def test_transaction_followup_on_listings_says_there_are_none(monkeypatch):
    class _Memory:
        active_candidates = [{
            "title": "Spacious | Prime Location | Marina View",
            "name": "Spacious | Prime Location | Marina View",
            "entity_type": "listing",
            "community": "Dubai Marina",
            "beds": 1,
        }]
        last_standalone_question = "Show me 1 bedroom apartments for rent in Dubai Marina."
        last_user_question = "Show me 1 bedroom apartments for rent in Dubai Marina."
        transaction_names = None
        transaction_offset = 0

    def boom(**kwargs):
        raise AssertionError("listing ads have no transaction data")

    monkeypatch.setattr("app.retrieval.propai_catalog.latest_sales", boom)
    monkeypatch.setattr("app.retrieval.propai_catalog.recent_sales", boom)

    turn = _transaction_turn("Show me the recent transaction details", _Memory())
    assert turn is not None
    answer = turn["fast_answer"].lower()
    assert "no recorded transactions" in answer
    assert "listings" in answer
    assert (turn["structured_summary"].get("transactions") or []) == []
    assert not turn["structured_summary"].get("transaction_followup")
    text = _format_transactions(
        ["Mayfair Tower", "Mayfair Residency"],
        {
            "sales": [
                {"project": "Mayfair Tower", "date": "2026-10-19", "price": 1000000, "bedrooms": 1, "size_sqft": 807},
                {"project": "Mayfair Tower", "date": "2026-08-19", "price": 1100000, "bedrooms": 1, "size_sqft": 807},
                {"project": "Mayfair Residency", "date": "2026-09-03", "price": 400000, "bedrooms": 1, "size_sqft": 298},
            ],
        },
        label="the Deyaar projects in Business Bay",
    )
    assert "Here are the recent transaction details for the Deyaar projects in Business Bay:" in text
    assert "**Mayfair Tower**" in text
    assert "**• Recent Transactions:**" in text
    assert "• **2026-10-19**: AED 1,000,000, 1 bedroom, 807 sqft" in text
    # Newest project activity first (Tower 2026-10-19 beats Residency 2026-09-03).
    assert text.index("Mayfair Tower") < text.index("Mayfair Residency")
    # Even if card order listed Residency first, freshest sale still leads.
    fresher_first = _format_transactions(
        ["Mayfair Residency", "Mayfair Tower"],
        {
            "sales": [
                {"project": "Mayfair Residency", "date": "2026-09-03", "price": 400000, "bedrooms": 1, "size_sqft": 298},
                {"project": "Mayfair Tower", "date": "2026-11-18", "price": 1000000, "bedrooms": 1, "size_sqft": 807},
            ],
        },
    )
    assert fresher_first.index("Mayfair Tower") < fresher_first.index("Mayfair Residency")


def test_shortlist_recommend_uses_projects_on_screen():
    class _Memory:
        active_candidates = [
            {
                "name": "Cheaper Ready",
                "entity_type": "project",
                "price": 1_000_000,
                "bedroom_min": 1,
                "bedroom_max": 2,
                "status": "Ready",
                "sales_count": 20,
                "handover_time": "2026-01-01",
            },
            {
                "name": "Pricier Later",
                "entity_type": "project",
                "price": 2_500_000,
                "bedroom_min": 2,
                "bedroom_max": 3,
                "status": "On Sale",
                "sales_count": 2,
                "handover_time": "2029-01-01",
            },
        ]

    turn = _shortlist_recommend_turn("Which of these is best?", _Memory())
    assert turn is not None
    assert turn["use_fast"] is True
    assert "**Acot Recommend:** Cheaper Ready" in turn["fast_answer"]
    assert "Why this is the strongest match" in turn["fast_answer"]
    best_to_buy = _shortlist_recommend_turn(
        "Tell me which project is best to buy",
        _Memory(),
    )
    assert best_to_buy is not None
    assert "**Acot Recommend:**" in best_to_buy["fast_answer"]
    assert "Starting price" not in best_to_buy["fast_answer"]
    assert _shortlist_recommend_turn(
        "Which of these have 2 bedroom options?",
        _Memory(),
    ) is None


def test_compare_first_two_uses_table_and_only_two_projects():
    class _Memory:
        active_candidates = [
            {
                "name": "Creek Beach - Bayshore",
                "entity_type": "project",
                "price": 1_600_000,
                "size_sqft": 694,
                "sales_count": 53,
                "rent_count": 249,
                "handover_time": "2022-10-01",
            },
            {
                "name": "Creek Beach - Orchid",
                "entity_type": "project",
                "price": 1_850_000,
                "size_sqft": 811,
                "sales_count": 80,
                "rent_count": 151,
                "handover_time": "2025-09-01",
            },
            {
                "name": "Creek Beach - Summer",
                "entity_type": "project",
                "price": 1_695_000,
                "size_sqft": 756,
                "sales_count": 28,
                "rent_count": 232,
                "handover_time": "2023-03-01",
            },
            {
                "name": "Creek Beach - Lotus",
                "entity_type": "project",
                "price": 1_700_000,
                "size_sqft": 742,
                "sales_count": 33,
                "rent_count": 185,
                "handover_time": "2025-09-01",
            },
        ]

    turn = _compare_shown_turn(
        "Compare the first two projects and tell me which is best",
        _Memory(),
    )
    assert turn is not None
    assert len(turn["structured_summary"]["projects"]) == 2
    assert turn["structured_summary"]["projects"][0]["name"] == "Creek Beach - Bayshore"
    assert turn["structured_summary"]["projects"][1]["name"] == "Creek Beach - Orchid"
    assert "| Factor |" in turn["fast_answer"]
    assert "**Acot Recommend:**" in turn["fast_answer"]
    assert "Creek Beach - Summer" not in turn["fast_answer"]
    assert _shortlist_recommend_turn(
        "Compare the first two projects and tell me which is best",
        _Memory(),
    ) is None


def test_compare_all_projects_uses_every_candidate_in_a_table():
    class _Memory:
        active_candidates = [
            {
                "name": f"Creek Beach - {name}",
                "entity_type": "project",
                "price": price,
                "size_sqft": size,
                "sales_count": sales,
                "rent_count": rents,
                "handover_time": "2023-01-01",
            }
            for name, price, size, sales, rents in (
                ("Bayshore", 1_600_000, 694, 53, 249),
                ("Orchid", 1_850_000, 811, 80, 151),
                ("Summer", 1_695_000, 756, 28, 232),
                ("Lotus", 1_700_000, 742, 33, 185),
                ("Savanna-Cedar-Mangrove", 2_167_944, 1011, 160, 320),
            )
        ]

    turn = _compare_shown_turn(
        "Compare all projects, which one is best to buy",
        _Memory(),
    )
    assert turn is not None
    assert len(turn["structured_summary"]["projects"]) == 5
    assert "| Factor |" in turn["fast_answer"]
    assert "Creek Beach - Summer" in turn["fast_answer"]
    assert "Creek Beach - Savanna-Cedar-Mangrove" in turn["fast_answer"]
    assert "**Acot Recommend:**" in turn["fast_answer"]


def test_compare_all_after_afford_does_not_need_preloaded_sales_counts():
    """On-screen compare keeps every card in the table even before sales attach."""
    class _Memory:
        active_candidates = [
            {
                "name": name,
                "entity_type": "project",
                "community": "Business Bay",
                "price": price,
                "bedroom_min": 2,
                "bedroom_max": 2,
            }
            for name, price in (
                ("Bay Square", 1_200_000),
                ("Executive Towers", 1_400_000),
                ("The Binary", 1_100_000),
            )
        ]
        dialogue_state = {
            "action": "recommend",
            "kind": "afford",
            "goal": "afford",
            "place": "Business Bay",
            "beds": 2,
            "budget_aed": 1_500_000,
        }

        def get_dialogue_state(self):
            return self.dialogue_state

    question = "compare all and tell me the best"
    turn = _compare_shown_turn(question, _Memory())
    assert turn is not None
    assert len(turn["structured_summary"]["projects"]) == 3
    assert "Bay Square" in turn["fast_answer"]
    assert "Executive Towers" in turn["fast_answer"]
    assert "The Binary" in turn["fast_answer"]
    assert "**Acot Recommend:**" in turn["fast_answer"]
    assert "Understood:" not in turn["fast_answer"]
    assert "1,500,000" not in turn["fast_answer"]
    assert _shortlist_recommend_turn(question, _Memory()) is None


def test_compare_all_after_afford_beats_sticky_understood_resolve():
    """LLM sticky afford must not replay when cards are on screen for compare-all."""
    import asyncio
    from app.answers.intent_route import classify_intent
    from app.answers.turn_resolve import resolve_turn
    from run import _dialogue_followup_turn, _understood_turn

    class _Memory:
        active_candidates = [
            {
                "name": name,
                "entity_type": "project",
                "community": "Business Bay",
                "price": price,
                "bedroom_min": 2,
                "bedroom_max": 2,
                "sales_count": sales,
                "rent_count": rents,
            }
            for name, price, sales, rents in (
                ("Bay Square", 1_200_000, 40, 100),
                ("Executive Towers", 1_400_000, 55, 120),
                ("The Binary", 1_100_000, 30, 90),
            )
        ]
        dialogue_state = {
            "action": "recommend",
            "kind": "afford",
            "goal": "afford",
            "place": "Business Bay",
            "beds": 2,
            "budget_aed": 1_500_000,
            "source_question": "Is AED 1.5M enough for a 2-bedroom in Business Bay?",
        }
        llm_client = None

        def get_dialogue_state(self):
            return self.dialogue_state

        def set_dialogue_state(self, state):
            self.dialogue_state = state

    question = "compare all and tell me the best"
    memory = _Memory()
    route = classify_intent(question, memory)
    assert route.get("kind") == "shortlist"
    assert route.get("subject") == "on_screen"

    async def _fake_afford_llm(_prompt):
        return (
            '{"action":"recommend","kind":"afford","goal":"afford",'
            '"place":"Business Bay","beds":2,"budget_aed":1500000,'
            '"confidence":0.95}'
        )

    resolved = asyncio.run(
        resolve_turn(question, memory, llm_generate=_fake_afford_llm)
    )
    # Must not invent sticky afford — leave the turn to compare/shortlist.
    assert resolved is None or (
        resolved.get("kind") == "shortlist"
        and resolved.get("action") == "recommend"
    )
    assert (resolved or {}).get("kind") != "afford"
    assert (resolved or {}).get("place") in {None, "", "on_screen"}

    followup = asyncio.run(_dialogue_followup_turn(question, memory, {}))
    assert followup is None

    understood = asyncio.run(_understood_turn(question, memory, {}))
    assert understood is not None
    assert "| Factor |" in understood["fast_answer"]
    assert "Understood:" not in understood["fast_answer"]
    assert "enough for a 2-bedroom" not in understood["fast_answer"]
    assert "Bay Square" in understood["fast_answer"]
    assert "Executive Towers" in understood["fast_answer"]


def _deira_studio_cards(entity_type="listing"):
    return [
        {
            "entity_type": entity_type,
            "listing_id": listing_id,
            "title": title,
            "price_aed": price,
            "size_sqft": size,
            "rent_frequency": "yearly",
            "beds": 0,
        }
        for listing_id, title, price, size in (
            ("11", "Large Studio Layout | High Exposure | Vacant", 22_000, 258),
            ("12", "Affordable Studio Apartment in Deira for Executives", 23_000, 130),
            ("13", "BEST DEAL - STUDIO ONLY 23K IN 12 PAYMENTS", 23_000, 152),
            ("14", "Large Studio Layout | High Exposure | Vacant", 26_000, 224),
            ("15", "One Month Rent Free-Studio Flat Available", 27_000, 355),
        )
    ]


def test_compare_all_apartments_ranks_on_screen_listings():
    class _Memory:
        active_candidates = _deira_studio_cards("property")

    question = "compare all apartments and tell me the best"
    turn = _compare_shown_turn(question, _Memory())
    assert turn is not None
    listings = turn["structured_summary"]["listings"]
    assert len(listings) == 5
    assert turn["structured_summary"]["projects"] == []
    assert turn["structured_summary"]["comparison"] is True
    assert turn["use_fast"] is True
    assert "| Factor |" in turn["fast_answer"]
    assert "**Acot Recommend:**" in turn["fast_answer"]
    assert "22,000" in turn["fast_answer"]
    assert "27,000" in turn["fast_answer"]
    assert "No relevant evidence" not in turn["fast_answer"]
    # Duplicate advert titles stay as two rows instead of collapsing.
    assert turn["fast_answer"].count("Large Studio Layout") >= 2
    assert _shortlist_recommend_turn(question, _Memory()) is None

    view = _detect_response_type(
        question,
        turn["question_type"],
        turn["structured_summary"],
    )
    assert view in {"listing_list", "comparison"}


def test_compare_all_and_best_to_live_uses_rental_cards():
    class _Memory:
        active_candidates = _deira_studio_cards("listing")

    turn = _compare_shown_turn(
        "compare all and tell the best to live",
        _Memory(),
    )
    assert turn is not None
    assert len(turn["structured_summary"]["listings"]) == 5
    assert "yearly rent" in turn["fast_answer"]
    assert "Recent sales" not in turn["fast_answer"]


def test_listing_compare_cards_show_baths_size_and_type():
    from app.memory.conversation_memory import ConversationMemory

    compact = ConversationMemory._compact_listing({
        "listing_id": "42",
        "title": "LARGE ONE BEDROOM FOR FAMILY ONLY 38K",
        "price_aed": 38_000,
        "size_sqft": {"min": 850, "max": 850},
        "baths": 2,
        "beds": 1,
        "property_type": "Apartment",
        "community": "International City",
        "rent_frequency": "yearly",
    })
    assert compact["baths"] == 2
    assert compact["size_sqft"] == 850
    assert compact["property_type"] == "Apartment"

    response = build_frontend_response(
        question="compare all apartments",
        standalone_question="compare all apartments",
        answer="Compared.",
        question_type="comparison",
        structured_summary={
            "listings": [compact],
            "projects": [],
            "comparison": True,
        },
        documents=[],
    )
    listing = response["data"]["listings"][0]
    assert listing["baths"] == 2
    assert listing["size_sqft"] == 850
    assert listing["property_type"] == "Apartment"
    assert not isinstance(listing["size_sqft"], dict)

    missing = ConversationMemory._compact_listing({
        "listing_id": "11",
        "title": "Large Studio Layout | High Exposure | Vacant",
        "price_aed": 22_000,
        "size_sqft": 258,
        "beds": 0,
    })
    blank = build_frontend_response(
        question="compare all apartments",
        standalone_question="compare all apartments",
        answer="Compared.",
        question_type="comparison",
        structured_summary={
            "listings": [missing],
            "projects": [],
            "comparison": True,
        },
        documents=[],
    )["data"]["listings"][0]
    assert blank["baths"] is None
    assert blank["property_type"] is None
    assert blank["size_sqft"] == 258


def test_more_transactions_uses_the_next_page():
    class _Memory:
        active_candidates = [
            {"name": "THE PINNACLE", "entity_type": "project", "community": "JVC"},
        ]
        transaction_names = None
        transaction_offset = 0

    first = _transaction_subject("Show me the recent transaction details", _Memory())
    assert first and first[0]["name"] == "THE PINNACLE"
    more_text = _format_transactions(
        ["THE PINNACLE"],
        {
            "sales": [
                {"project": "THE PINNACLE", "date": "2026-04-01", "price": 900000, "bedrooms": 1, "size_sqft": 700},
            ],
        },
        label="THE PINNACLE",
        more=True,
    )
    assert "Here are more recent transaction details for THE PINNACLE:" in more_text
    assert "**THE PINNACLE**" in more_text
    empty_more = _format_transactions(
        ["THE PINNACLE"],
        {"sales": []},
        label="THE PINNACLE",
        more=True,
    )
    assert "no more recorded sales" in empty_more.lower()


def test_a_rental_comparison_stays_off_the_project_list():
    class _Memory:
        def context(self):
            return {"active_candidates": []}

    question = "Compare the first two rentals in Dubai Marina and tell me which is best."
    assert _listing_question(question, _Memory()) is True
    assert _listing_question("Show me projects in Dubai Marina", _Memory()) is False


def test_price_followup_uses_the_projects_already_shown():
    class _Memory:
        active_candidates = [
            {
                "name": "Altia One",
                "entity_type": "project",
                "developer_name": "Yas Developers",
                "price": 1395448,
                "bedroom_min": 1,
                "bedroom_max": 3,
            }
        ]

    projects, no_match = _shown_project_followup(
        "What are their starting prices?", _Memory()
    )
    assert no_match is False
    answer = _build_fast_structured_answer(
        "What are their starting prices?",
        {"projects": projects},
    )
    assert "AED 1,395,448" in answer
    assert "exact match" not in answer.lower()
    assert _shown_project_followup("Show me projects in Dubai Marina", _Memory()) is None


def test_rera_guidance_uses_the_agent_not_hybrid_evidence_only():
    from run import _is_dubai_guidance_question, _propai_sql_browse

    assert _is_dubai_guidance_question("What is rera?")
    assert _is_dubai_guidance_question("What is freehold?")
    assert _is_dubai_guidance_question("How do I buy property in Dubai?")
    assert not _is_dubai_guidance_question("Show me off-plan projects in Dubai Marina")
    assert not _is_dubai_guidance_question("Show me projects in Dubai Marina")
    assert _propai_sql_browse("What is rera?") is False
    assert _propai_sql_browse("Show me projects in Dubai Marina") is True


def test_budget_for_each_project_is_group_attribute_not_project_name():
    from app.answers.intent_route import project_detail_subject
    from run import _propai_sql_browse

    question = "budget details about the budget for each project"

    class _Memory:
        active_candidates = [
            {
                "name": "Marina Cove",
                "entity_type": "project",
                "price": 3196888,
            },
            {
                "name": "Rove Home Dubai Marina",
                "entity_type": "project",
                "price": 1576250,
            },
        ]

    assert is_group_attribute_question(question)
    assert project_detail_subject(question, _Memory()) is None
    assert _propai_sql_browse(question, _Memory()) is False
    assert _propai_sql_browse("Show me projects in Dubai Marina", _Memory()) is True

    projects, no_match = _shown_project_followup(question, _Memory())
    assert no_match is False
    assert [item["name"] for item in projects] == [
        "Marina Cove",
        "Rove Home Dubai Marina",
    ]
    answer = _build_fast_structured_answer(question, {"projects": projects})
    assert "AED 3,196,888" in answer
    assert "AED 1,576,250" in answer
    assert "do not have a project matching" not in answer.lower()


def test_followups_skip_the_question_just_asked():
    chips = _followup_questions({
        "question": "Who are you?",
        "small_talk": "identity",
        "question_type": "general",
        "structured_summary": _summary(projects=[]),
    })
    assert "Who are you?" not in chips
    assert "What can you do?" in chips

    price_chips = _followup_questions({
        "question": "What are their starting prices?",
        "question_type": "information",
        "structured_summary": _summary(),
    })
    assert "What are their starting prices?" not in price_chips
    assert "Which of these have 2-bedroom options?" in price_chips

    class _Memory:
        turns = [
            {"user_question": "Show me the projects in Jumeirah Village Circle."},
            {"user_question": "What are their starting prices?"},
            {"user_question": "Which of these have 2-bedroom options?"},
        ]

    later_chips = _followup_questions({
        "question": "Which of these have 2-bedroom options?",
        "question_type": "information",
        "structured_summary": _summary(),
        "conversation_memory": _Memory(),
    })
    assert "What are their starting prices?" not in later_chips
    assert "Which of these have 2-bedroom options?" not in later_chips
    assert later_chips
    assert later_chips != price_chips


def test_stretched_greetings_match_plain_ones():
    assert _classify_small_talk("hiiiii") == _classify_small_talk("hi")
    assert _classify_small_talk("heyyyy") == _classify_small_talk("hey")
    assert _classify_small_talk("hiiiii") == "greeting"
    assert _classify_small_talk("hellooo") == "greeting"


def test_short_forms_become_places():
    assert EntityResolver.normalize("hyd") == "hyderabad"
    prepared = prepare_user_question("I need properties in hyd")
    assert "hyderabad" in prepared
    assert "projects" in prepared

    shape = interpret_search("I need properties in hyd")
    assert shape["kind"] == "community"
    assert shape["community"] == "hyderabad"

    slang = interpret_search("yo gimme all binghatti stuff in bb")
    assert slang["kind"] == "developer_community"
    assert slang["developer"] == "binghatti"
    assert slang["community"] == "business bay"


def test_opening_line_and_empty_or_ambiguous_names():
    one = _summary(projects=[{
        "name": "310 Riverside Crescent",
        "developer_name": "Sobha",
        "community": "Meydan",
        "price": 1500000,
        "status": "ACTIVE",
    }])
    one_answer = _build_fast_structured_answer(
        "Show me 310 Riverside Crescent",
        one,
    )
    assert "Here are 1" not in one_answer
    assert (
        "310 Riverside Crescent is an active project in Meydan."
        in one_answer
    )

    missing = _summary(projects=[])
    missing["understood_subject"] = "Hyderabad"
    missing["understood_kind"] = "place"
    missing["closest_match"] = "Al Jaddaf"
    missing_answer = _build_fast_structured_answer(
        "I need properties in hyd",
        missing,
    )
    assert "I don't have projects in Hyderabad." in missing_answer
    assert "The closest community I do have is Al Jaddaf." in missing_answer
    assert "No relevant evidence" not in missing_answer

    several = _summary(projects=[
        {"name": "Damac Bay", "community": "Dubai Harbour", "status": "ACTIVE", "price": 1000000},
        {"name": "Damac Bay 2", "community": "Dubai Harbour", "status": "ACTIVE", "price": 1200000},
        {"name": "Damac Bay Cavalli Phase 2", "community": "Dubai Harbour", "status": "ACTIVE", "price": 2000000},
    ])
    several["ambiguous_label"] = "Damac Bay"
    listed = _build_fast_structured_answer("Show me Damac Bay", several)
    assert "Damac Bay matches 3 projects. Here they are:" in listed
    assert "- **Damac Bay 2**" in listed


def test_search_shape_before_a_single_entity():
    one = interpret_search("Show me 310 Riverside Crescent")
    assert one["kind"] == "project"
    assert one["name"] == "310 riverside crescent"

    both = interpret_search("Show me Azizi projects in Meydan")
    assert both["kind"] == "developer_community"
    assert both["developer"] == "azizi"
    assert both["community"] == "meydan"

    containing = interpret_search(
        "Show me all projects containing Riverside Crescent"
    )
    assert containing["kind"] == "name_pattern"
    assert containing["stem"] == "riverside crescent"
    assert containing["starts_with"] is False

    family = interpret_search("Show me all Azizi Riviera projects")
    assert family["kind"] == "name_pattern"
    assert family["stem"] == "azizi riviera"

    compared = interpret_search(
        "Compare Azizi Riviera 49 and Binghatti Aquarise"
    )
    assert compared["kind"] == "compare"
    assert compared["names"] == [
        "azizi riviera 49",
        "binghatti aquarise",
    ]


def test_two_area_comparison_is_built_without_a_model(monkeypatch):
    from app.answers import area_comparison as ac

    monkeypatch.setattr(ac, "_names", {
        "dubai marina": "Dubai Marina", "business bay": "Business Bay",
        "jumeirah village circle": "Jumeirah Village Circle", "downtown dubai": "Downtown Dubai"})
    assert ac.parse_area_comparison("Compare Dubai Marina and Business Bay.") == ("Dubai Marina", "Business Bay")
    assert ac.parse_area_comparison("marina vs jvc") == ("Dubai Marina", "Jumeirah Village Circle")
    assert ac.parse_area_comparison("what is the difference between Business Bay and Downtown?") == (
        "Business Bay", "Downtown Dubai")
    # what speech-to-text actually writes
    assert ac.parse_area_comparison("Comparative Business Bay and Dubai, Marina.") == ("Business Bay", "Dubai Marina")
    assert ac.parse_area_comparison("Compared Business Bay and Dubai Marina") == ("Business Bay", "Dubai Marina")
    assert ac.parse_area_comparison("Uh, compare Business Bay and Dubai Marina.") == ("Business Bay", "Dubai Marina")
    assert ac.parse_area_comparison("Compare business way and Dubai Marina.") == ("Business Bay", "Dubai Marina")
    assert ac.parse_area_comparison("Compare those 2 like Dubai Marina and Business Bay.") == ("Dubai Marina", "Business Bay")
    assert ac.parse_area_comparison("which is better for families, JVC or Downtown?") == ("Jumeirah Village Circle", "Downtown Dubai")
    assert ac.parse_area_comparison("Dubai Marina and Business Bay") is None  # no comparison word
    assert ac.parse_area_comparison("Which is the best for families? Is it, JBC or downtown?") == (
        "Jumeirah Village Circle", "Downtown Dubai")
    assert ac.looks_like_comparison("JVC or Downtown?")
    assert not ac.looks_like_comparison("Should I buy in JVC or not?")
    assert ac.parse_area_comparison("Okay so, marina vs jvc") == ("Dubai Marina", "Jumeirah Village Circle")
    assert ac.parse_area_comparison("comparison between Dubai Marina and Business Bay") == ("Dubai Marina", "Business Bay")
    # "Tell me about JVC": exactly one known area named
    assert ac.mentioned_area("Tell me about JVC") == "Jumeirah Village Circle"
    assert ac.mentioned_area("how is the marina for families?") == "Dubai Marina"
    assert ac.mentioned_area("marina vs jvc") is None
    assert ac.mentioned_area("Tell me about Sobha One") is None
    # developers, projects and the same area twice are not area comparisons
    assert ac.parse_area_comparison("Compare Sobha and Damac") is None
    assert ac.parse_area_comparison("Compare Dubai Marina and Dubai Marina") is None
    assert ac.parse_area_comparison("show me projects in Dubai Marina") is None

    def fake_facts(area):
        big = area == "Business Bay"
        return {"projects": 15 if big else 1, "min_price": 650000 if big else 21132340,
                "median_price": 2828000, "beds": [0, 8] if big else [3, 4],
                "top_year": {"year": 2027, "projects": 8} if big else {"year": 2026, "projects": 1},
                "listings": 143 if big else 122, "rent_median": 90000 if big else 115000,
                "rent_range": [75000, 120000], "rent_psf": 124.4}

    monkeypatch.setattr(ac, "_facts", fake_facts)
    answer = ac.build_area_comparison("Dubai Marina", "Business Bay")
    assert "| Factor | Dubai Marina | Business Bay |" in answer
    assert "| Starting prices | from AED 21,132,340 | from AED 650,000 (median AED 2,828,000) |" in answer
    assert "Business Bay has more projects for sale (15 against 1)" in answer
    assert "no yield" in answer
    assert "Summary:" in answer


def _broker_rows():
    return [
        {
            "id": "1001",
            "brn": "1001",
            "name": "Amal Khan",
            "agency_name": "Horizon Realty",
            "card_rank": "Gold",
            "license_number": "L-1",
            "card_expiry_date": "2027-06-01",
            "awards_count": 4,
        },
        {
            "id": "1002",
            "brn": "1002",
            "name": "Sara Nair",
            "agency_name": "Creek Homes",
            "card_rank": "Silver",
            "license_number": "L-2",
            "card_expiry_date": "2026-12-01",
            "awards_count": 1,
        },
    ]


def test_public_broker_omits_contact_fields():
    from app.retrieval.propai_catalog import _BROKER_COLUMNS, _BROKER_PUBLIC, _public_broker

    assert "email" not in _BROKER_COLUMNS
    assert "mobile" not in _BROKER_COLUMNS
    assert "landline" not in _BROKER_COLUMNS
    shown = _public_broker({
        "brn": "689",
        "name_en": "Test Broker",
        "agency_name": "Acme",
        "email": "hidden@example.com",
        "mobile": "+971500000000",
        "landline": "+97140000000",
        "card_rank": "Gold",
        "card_expiry_date": "2027-06-01",
        "awards_count": 2,
    })
    blob = str(shown).lower()
    assert "email" not in shown
    assert "mobile" not in shown
    assert "landline" not in shown
    assert "hidden@" not in blob
    assert shown["name"] == "Test Broker"
    assert shown["brn"] == "689"
    assert set(shown) == set(_BROKER_PUBLIC)


def test_broker_turn_dubai_with_budget_still_lists_agents(monkeypatch):
    """'Top agents for Dubai starting above AED 5M' must not be treated as outside Dubai."""
    called = []

    def fake_list(**kwargs):
        called.append(kwargs)
        return _broker_rows()[: kwargs.get("limit", 8)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    question = "Tell me the top agents for Dubai starting above AED 5 million"
    assert _broker_place(question, None) in {"Dubai", "dubai"} or str(
        _broker_place(question, None)
    ).lower() == "dubai"
    turn = _broker_turn(question, None)
    assert turn is not None
    assert turn["structured_summary"]["brokers"]
    assert "only show licensed agents in dubai" not in turn["fast_answer"].lower()
    assert called and called[0].get("band") == "top"
    assert str(called[0].get("place") or "").lower() in {"dubai", ""}


def test_broker_turn_lists_top_brokers(monkeypatch):
    rows = _broker_rows()
    monkeypatch.setattr(
        "app.retrieval.propai_catalog.list_brokers",
        lambda **kwargs: rows[: kwargs.get("limit", 8)],
    )
    turn = _broker_turn("Show me brokers", None)
    assert turn is not None
    assert turn["use_fast"] is True
    # The agents go to the app as licence cards; the text is only the lead line, so they are not listed twice.
    names = [row["name"] for row in turn["structured_summary"]["brokers"]]
    assert "Amal Khan" in names and "Sara Nair" in names
    assert "Amal Khan" not in turn["fast_answer"]
    assert "broker" not in turn["fast_answer"].lower()
    assert "licensed agents" in turn["fast_answer"].lower()
    assert 1 <= len(turn["structured_summary"]["brokers"]) <= 8
    assert "email" not in str(turn["structured_summary"]["brokers"]).lower()
    assert _broker_turn("Show me projects in Business Bay", None) is None
    assert _broker_turn("Show me agents", None) is not None
    assert _broker_turn("show me in dubai marina", None) is None


def test_broker_followup_keeps_agent_intent(monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    class _Memory:
        last_user_question = "show me agents details in mumbai"
        last_standalone_question = "show me agents details in mumbai"
        turns = [{"user_question": "show me agents details in mumbai"}]
        active_candidates = []

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    turn = _broker_turn("show me in dubai marina", _Memory())
    assert turn is not None
    assert turn["structured_summary"]["brokers"]
    assert captured.get("place") == "Dubai Marina"
    assert "licensed agents" in turn["fast_answer"].lower()
    assert _broker_turn("show me apartments in dubai marina", _Memory()) is None
    assert _broker_turn("show me projects in dubai marina", _Memory()) is None


def test_bedroom_followup_keeps_on_screen_place():
    from run import _inherit_place_for_home_browse, _propai_sql_browse
    from app.answers.intent_route import is_incomplete_home_browse

    class _Memory:
        last_user_question = "Show me projects in Jumeirah Village Circle."
        last_standalone_question = "Show me projects in Jumeirah Village Circle."
        turns = [{"user_question": "Show me projects in Jumeirah Village Circle."}]
        active_candidates = [
            {
                "name": "Serenz by Danube",
                "entity_type": "project",
                "community": "Jumeirah Village Circle",
            }
        ]

    follow = "show me two-bedroom apartments"
    assert is_incomplete_home_browse(follow)
    rewritten = _inherit_place_for_home_browse(
        follow,
        _Memory(),
        "Show me projects in Jumeirah Village Circle.",
    )
    assert "jumeirah village circle" in rewritten.lower()
    assert "2 bedroom" in rewritten.lower()
    assert _propai_sql_browse(rewritten, _Memory()) is True
    assert _inherit_place_for_home_browse(
        "show me two-bedroom apartments", None, None
    ) == "show me two-bedroom apartments"


def test_top_agents_for_these_projects_uses_on_screen_community(monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    class _Memory:
        last_user_question = "show me 2 bedrooms apartments in Dubai Marina"
        last_standalone_question = "show me 2 bedrooms apartments in Dubai Marina"
        turns = []
        active_candidates = [
            {
                "name": "Pier 8",
                "entity_type": "project",
                "community": "Dubai Marina",
            },
            {
                "name": "Marina Living",
                "entity_type": "project",
                "community": "Dubai Marina",
            },
        ]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    turn = _broker_turn("tell me the top agents for these projects", _Memory())
    assert turn is not None
    assert captured.get("place") == "Dubai Marina"
    assert "dubai marina" in turn["fast_answer"].lower()
    assert turn["structured_summary"]["brokers"]
    assert "don't have any projects" not in turn["fast_answer"].lower()


def test_broker_followup_accepts_businessbay_spelling(monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    class _Memory:
        last_user_question = "i want agents in anantapur"
        last_standalone_question = "i want agents in anantapur"
        turns = [{"user_question": "i want agents in anantapur"}]
        active_candidates = []

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    refused = _broker_turn("i want agents in anantapur", None)
    assert refused is not None
    assert refused["structured_summary"]["brokers"] == []
    assert "dubai" in refused["fast_answer"].lower()

    turn = _broker_turn("show me in businessbay", _Memory())
    assert turn is not None
    assert turn["structured_summary"]["brokers"]
    assert captured.get("place") == "Business Bay"
    assert "licensed agents" in turn["fast_answer"].lower()
    assert "only show licensed agents in dubai" not in turn["fast_answer"].lower()


def test_broker_turn_reads_mistyped_agent_sentence(monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    assert "agents" in repair_typed_question("show me agnets in busines bay")
    assert "agents" in prepare_user_question("show me agnets in busines bay")
    turn = _broker_turn("show me agnets in busines bay", None)
    assert turn is not None
    assert turn["structured_summary"]["brokers"]
    assert captured.get("place") == "Business Bay"
    assert "licensed agents" in turn["fast_answer"].lower()

    captured.clear()
    marina = _broker_turn("show me agnets in marinna", None)
    assert marina is not None
    assert captured.get("place") == "Dubai Marina"

    assert _ambiguous_place_question("show me in marinna", None) == "Dubai Marina"
    assert _ambiguous_place_question("show me in jumera", None) is None
    assert _broker_turn("show me agnets in chennai", None)["structured_summary"]["brokers"] == []


def test_what_is_dubai_is_definitional_not_place_choice():
    from run import _is_definitional_question, _propai_sql_browse

    assert _is_definitional_question("what is dubai")
    assert _is_definitional_question("What is Marina?")
    assert _is_definitional_question("tell me about Dubai Marina")
    assert not _is_definitional_question("What is the price of 23 Marina?")
    assert not _is_definitional_question("show me in dubai marina")
    assert _ambiguous_place_question("what is dubai", None) is None
    assert _ambiguous_place_question("What is Marina?", None) is None
    assert _propai_sql_browse("what is dubai") is False
    assert _propai_sql_browse("What is the price of 23 Marina?") is True
    assert _propai_sql_browse("show me 2 bedrooms apartments in Dubai Marina") is True
    assert _propai_sql_browse("Hi, I am looking for the two bedroom apartments.") is False
    assert _is_broker_question("jukam suggest plot in India", None) is False
    assert _is_broker_question("show me indian agents", None) is True


def test_first_place_only_question_asks_properties_or_agents():
    assert _ambiguous_place_question("show me in dubai marina", None) == "Dubai Marina"
    assert _ambiguous_place_question("in jvc", None) == "Jumeirah Village Circle"
    assert _ambiguous_place_question("show me agents in dubai marina", None) is None
    assert _ambiguous_place_question("show me projects in dubai marina", None) is None
    assert _ambiguous_place_question("show me apartments in dubai marina", None) is None
    assert _ambiguous_place_question("show me in mumbai", None) is False

    class _AgentMemory:
        last_user_question = "show me agents details in mumbai"
        last_standalone_question = "show me agents details in mumbai"
        turns = [{"user_question": "show me agents details in mumbai"}]
        active_candidates = []

    assert _ambiguous_place_question("show me in dubai marina", _AgentMemory()) is None

    class _PropertyMemory:
        last_user_question = "show me projects in Business Bay"
        last_standalone_question = "show me projects in Business Bay"
        turns = [{"user_question": "show me projects in Business Bay"}]
        active_candidates = [{"name": "Canal Crown", "entity_type": "project"}]

    assert _ambiguous_place_question("show me in dubai marina", _PropertyMemory()) is None

    turn = _place_choice_turn("show me in dubai marina", "Dubai Marina", None)
    assert "properties or agents" in turn["fast_answer"].lower()
    assert turn["structured_summary"]["place_choice"] is True
    chips = _followup_questions(turn)
    assert chips == [
        "Show me properties in Dubai Marina",
        "Show me agents in Dubai Marina",
    ]

    class _Pending:
        pending_place_choice = "Dubai Marina"

    assert _apply_pending_place_choice("agents", _Pending()) == "Show me agents in Dubai Marina"
    pending = _Pending()
    assert _apply_pending_place_choice("properties", pending) == "Show me projects in Dubai Marina"
    assert pending.pending_place_choice is None


def test_broker_turn_does_not_fake_nationality_or_gender(monkeypatch):
    called = []

    def fake_list(**kwargs):
        called.append(kwargs)
        return _broker_rows()[: kwargs.get("limit", 8)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    indian = _broker_turn("Show me indian brokers", None)
    assert indian is not None
    assert indian["use_fast"] is True
    assert called == []
    assert indian["structured_summary"]["brokers"] == []
    assert "nationality" in indian["fast_answer"].lower()
    assert "broker" not in indian["fast_answer"].lower()
    assert "Indian" in indian["fast_answer"]

    mixed = _broker_turn("indian female and male brokers", None)
    assert mixed is not None
    assert called == []
    assert mixed["structured_summary"]["brokers"] == []
    assert "broker" not in mixed["fast_answer"].lower()

    follow = _broker_turn("only indian female", None)
    assert follow is not None
    assert called == []
    assert "female" in follow["fast_answer"].lower()


def test_broker_turn_refuses_places_outside_dubai(monkeypatch):
    called = []

    def fake_list(**kwargs):
        called.append(kwargs)
        return _broker_rows()

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    turn = _broker_turn("show me agents details in chennai", None)
    assert turn is not None
    assert called == []
    assert turn["structured_summary"]["brokers"] == []
    text = turn["fast_answer"].lower()
    assert "dubai" in text
    assert "dld" not in text
    assert "isn't a dubai community" not in text
    assert "licensed agents for chennai" not in text
    assert "broker" not in text
    assert "layla" not in text

    mumbai = _broker_turn("show me agents in mumbai", None)
    assert mumbai["structured_summary"]["brokers"] == []
    assert "dubai" in mumbai["fast_answer"].lower()
    assert "dld" not in mumbai["fast_answer"].lower()
    assert called == []

    leftover = _broker_turn("chennai agents", None)
    assert leftover["structured_summary"]["brokers"] == []
    assert called == []

    mixed = _broker_turn("indian brokers in chennai", None)
    assert mixed["structured_summary"]["brokers"] == []
    assert "dld" not in mixed["fast_answer"].lower()
    assert called == []

    class _Memory:
        active_candidates = [
            {
                "name": "Layla Hadi Alsayedmousa Alsayedabdulla Almousawi",
                "entity_type": "project",
                "community": "chennai",
            }
        ]

    prior = _broker_turn("show me agents details in chennai", _Memory())
    assert prior["structured_summary"]["brokers"] == []
    assert called == []


def test_broker_turn_still_lists_dubai_places(monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[:5]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    turn = _broker_turn("show me agents details in palm jumeirah", None)
    assert turn is not None
    assert "Palm Jumeirah" in turn["fast_answer"]
    assert captured.get("place") == "Palm Jumeirah"
    assert captured.get("name") in (None, "")
    assert turn["structured_summary"]["brokers"]

    captured.clear()
    dubai = _broker_turn("show me agents in dubai", None)
    assert dubai["structured_summary"]["brokers"]
    assert captured.get("place") == "Dubai"


def test_broker_followup_uses_the_community_on_screen(monkeypatch):
    class _Memory:
        active_candidates = [
            {
                "name": "Serenz",
                "entity_type": "project",
                "community": "Jumeirah Village Circle",
            },
            {
                "name": "Azizi Ruby",
                "entity_type": "project",
                "community": "Jumeirah Village Circle",
            },
        ]

    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    memory = _Memory()
    turn = _broker_turn(
        "Tell me the top brokers for Jumeirah Village Circle",
        memory,
    )
    assert turn is not None
    assert "Jumeirah Village Circle" in turn["fast_answer"]
    assert "broker" not in turn["fast_answer"].lower()
    assert "Here are licensed agents for Jumeirah Village Circle." in turn["fast_answer"]
    assert captured.get("place") == "Jumeirah Village Circle"
    # An explicit "top brokers" ask gets the top (Five Star) band.
    assert captured.get("band") == "top"
    assert memory.active_candidates[0]["name"] == "Serenz"
    assert turn["structured_summary"]["projects"] == []
    text = _format_brokers(_broker_rows(), label="Jumeirah Village Circle")
    assert "**Amal Khan**" in text
    assert "BRN 1001" in text
    assert "not exclusive" not in text.lower()


def test_broker_followup_after_yield_areas_uses_first_area(monkeypatch):
    class _Memory:
        active_candidates = [
            {
                "name": "Marina Gate",
                "entity_type": "project",
                "community": "Dubai Marina",
            },
            {
                "name": "Bay Square",
                "entity_type": "project",
                "community": "Business Bay",
            },
        ]

    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    question = "Tell me the top agents for these projects"
    assert _broker_place(question, _Memory()) == "Dubai Marina"
    turn = _broker_turn(question, _Memory())
    assert turn is not None
    assert "I can only show licensed agents in Dubai" not in turn["fast_answer"]
    assert "Here are licensed agents for Dubai Marina." in turn["fast_answer"]
    assert captured.get("place") == "Dubai Marina"

    chips = _followup_questions({
        "question": "Which Dubai areas have the best rental yield?",
        "question_type": "search",
        "structured_summary": {
            "community": [
                {
                    "name": "Dubai Marina",
                    "official_area": "MARSA DUBAI",
                    "signals": {"rent_contracts": 4200},
                },
                {
                    "name": "Business Bay",
                    "official_area": "BUSINESS BAY",
                    "signals": {"rent_contracts": 3100},
                },
            ],
            "projects": [
                {"name": "Marina Gate", "community": "Dubai Marina"},
                {"name": "Bay Square", "community": "Business Bay"},
            ],
            "listings": [],
        },
    })
    assert chips[0] == "Tell me the top agents for Dubai Marina"


def test_project_search_offers_broker_followup():
    chips = _followup_questions({
        "question": "Show me projects in Jumeirah Village Circle",
        "question_type": "search",
        "structured_summary": _summary(),
    })
    assert chips[0] == "Tell me the top agents for Jumeirah Village Circle"

    one = _followup_questions({
        "question": "Tell me about Serenz",
        "question_type": "information",
        "structured_summary": _summary(projects=[_projects()[0]]),
    })
    assert one[0] == "Tell me the top agents for Serenz"


def test_broker_list_response_omits_contact_fields():
    response = build_frontend_response(
        question="Show me brokers",
        standalone_question="Show me brokers",
        answer="Here are the top-ranked licensed brokers in our DLD data:",
        question_type="general",
        structured_summary={
            "brokers": [{
                "brn": "689",
                "name": "Amal Khan",
                "agency_name": "Horizon Realty",
                "card_rank": "Gold",
                "email": "hidden@example.com",
                "mobile": "+971500000000",
            }],
            "projects": [],
            "listings": [],
        },
        documents=[],
    )
    assert response["response_type"] == "broker_list"
    broker = response["data"]["brokers"][0]
    assert broker["name"] == "Amal Khan"
    assert "email" not in broker
    assert "mobile" not in broker


def test_full_project_data_uses_top_broker_band(monkeypatch):
    class _Memory:
        active_candidates = [{
            "name": "Canal Crown",
            "entity_type": "project",
            "community": "Business Bay",
            "price": 1_150_000,
            "bedroom_min": 1,
            "bedroom_max": 2,
            "size_min": 700,
            "size_max": 1100,
            "handover_time": "2027-06-01",
            "photos": ["https://example.com/p.jpg"],
            "sales_count": 12,
        }]

    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    assert _project_is_full(_Memory.active_candidates[0]) is True
    assert _broker_band(_Memory(), "Business Bay") == "top"
    turn = _broker_turn("Tell me the top brokers for Business Bay", _Memory())
    assert turn is not None
    assert captured.get("band") == "top"
    assert captured.get("place") == "Business Bay"


def test_broker_rank_filter_uses_bound_params(monkeypatch):
    from app.retrieval import propai_catalog as catalog

    captured = {}

    def fake_fetch(sql, params=()):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(catalog, "fetch", fake_fetch)
    catalog._broker_rows(ranks="top", limit=5)
    assert "%five" not in captured["sql"]
    assert captured["params"] == ("%five%star%", 5)
    catalog._broker_rows(ranks="mid", limit=5)
    assert "%four" not in captured["sql"]
    assert captured["params"] == ("%four%star%", "%three%star%", 5)


def test_seeded_broker_pick_varies_by_place():
    from app.retrieval.propai_catalog import seeded_broker_pick

    pool = [{"id": str(i), "brn": str(i), "name": f"Broker {i}"} for i in range(1, 21)]
    jvc = seeded_broker_pick(pool, "Jumeirah Village Circle", 5, shuffle=False)
    bay = seeded_broker_pick(pool, "Business Bay", 5, shuffle=False)
    jvc_again = seeded_broker_pick(pool, "Jumeirah Village Circle", 5, shuffle=False)
    assert [row["brn"] for row in jvc] == [row["brn"] for row in jvc_again]
    assert [row["brn"] for row in jvc] != [row["brn"] for row in bay]
    mid_jvc = seeded_broker_pick(pool, "Jumeirah Village Circle", 5, shuffle=True)
    mid_bay = seeded_broker_pick(pool, "Business Bay", 5, shuffle=True)
    assert [row["brn"] for row in mid_jvc] != [row["brn"] for row in mid_bay]


def test_search_brokers_refuses_places_outside_dubai(monkeypatch):
    from app.listings.tools import search_brokers

    def boom(**kwargs):
        raise AssertionError("must not list Dubai agents for a city outside Dubai")

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", boom)
    result = search_brokers({"place": "chennai"})
    assert result.output["out_of_coverage"] is True
    assert result.output["brokers"] == []
    assert result.cards == []
    assert "dubai" in result.output["note"].lower()


def test_tools_for_keeps_search_brokers_on_broker_questions():
    from app.listings.prompt import tools_for

    names = {tool["function"]["name"] for tool in tools_for("Show me brokers")}
    assert names == {"search_brokers", "update_profile"}
    agents = {tool["function"]["name"] for tool in tools_for("Show me agents")}
    assert agents == {"search_brokers", "update_profile"}
    rent = {tool["function"]["name"] for tool in tools_for("apartments for rent in marina")}
    assert "search_brokers" not in rent
    # Rent asks use Prop-AI rent tools (propai_rents), not project-only tools.
    assert "search_listings" in rent


def test_property_browse_uses_project_tools_not_rentals():
    from app.listings.prompt import tools_for

    names = {tool["function"]["name"] for tool in tools_for("Show me properties in Dubai Marina")}
    assert "search_projects" in names
    assert "search_listings" not in names
    assert "find_areas" not in names


def test_tools_for_open_buy_does_not_send_market_stats_stack():
    from app.listings.prompt import tools_for

    names = {tool["function"]["name"] for tool in tools_for("Where should I live in Dubai?")}
    assert names == {"area_sales", "update_profile"}
    buy = {tool["function"]["name"] for tool in tools_for("off-plan projects to buy in JVC")}
    assert "search_projects" in buy
    assert "search_brochures" not in buy
    assert "compare_projects" not in buy
    assert "area_sales" in buy


def test_recent_sales_wording_is_a_transaction_ask():
    assert _TRANSACTION.search("Show me recent sales in Business Bay")
    assert _TRANSACTION.search("Show the last 5 sales of 1-bedroom apartments in JVC")
    assert _TRANSACTION.search("How many 1-bedroom apartments in JVC were sold in November 2026?")


def test_last_5_sales_uses_limit_five(monkeypatch):
    """Understood/sales path must honor 'last 5', not hardcode 8."""
    import app.retrieval.propai_catalog as catalog

    seen = {}

    def fake_latest_sales(**kwargs):
        seen.update(kwargs)
        return {
            "count": 1,
            "sales": [{
                "project": "Luma Park Views",
                "date": "2026-11-03",
                "price": 1_319_424,
                "bedrooms": 1,
                "size_sqft": 880,
                "property_type": "Unit",
            }],
        }

    monkeypatch.setattr(catalog, "latest_sales", fake_latest_sales)
    monkeypatch.setattr(
        catalog, "resolve_dubai_place", lambda place: "Jumeirah Village Circle"
    )
    monkeypatch.setattr(
        catalog,
        "place_named",
        lambda text: "Jumeirah Village Circle" if "jvc" in (text or "").lower() else None,
    )

    class _Memory:
        dialogue_state = {}
        sale_query = None
        transaction_names = None
        transaction_offset = 0
        active_candidates = []

        def get_dialogue_state(self):
            return self.dialogue_state

        def set_dialogue_state(self, state):
            self.dialogue_state = state
            return state

        def patch_dialogue_state(self, updates):
            from app.memory.dialogue_state import patch_dialogue_state

            self.dialogue_state = patch_dialogue_state(self.dialogue_state, updates)
            return self.dialogue_state

    memory = _Memory()
    out = _transaction_turn(
        "Show the last 5 sales of 1-bedroom apartments in JVC",
        memory,
    )
    assert out is not None
    assert seen.get("limit") == 5
    assert seen.get("bedrooms") == 1
    assert memory.dialogue_state.get("limit") == 5
    assert "last 5" in out["fast_answer"].lower()


def test_sales_followup_after_agents_is_not_sticky_broker():
    class _Memory:
        last_user_question = "Who are the top agents in Business Bay?"
        last_standalone_question = "Who are the top agents in Business Bay?"
        turns = [{"user_question": "Who are the top agents in Business Bay?"}]
        active_candidates = []

    q = "Show me recent sales in Business Bay"
    assert _is_broker_question(q, _Memory()) is False
    assert _sql_project_browse_turn(q, _Memory()) is None


def test_data_freshness_answers_today_and_max_sale_date(monkeypatch):
    from datetime import date
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "fetch",
        lambda sql, params=None: [{"latest": date(2026, 11, 3)}],
    )
    out = _data_freshness_turn(
        "What is today's date, and what is the latest sale date in your data?"
    )
    assert out is not None
    assert out["use_fast"] is True
    assert out.get("documents") == []
    assert "Today is" in out["fast_answer"]
    assert "3 Nov 2026" in out["fast_answer"] or "2026-11-03" in out["fast_answer"]
    assert "invent" in out["fast_answer"].lower()


def test_data_freshness_paraphrases_about_recent_sales_data(monkeypatch):
    """Spoken/meta freshness asks must not fall through to sticky recommend."""
    from datetime import date
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "fetch",
        lambda sql, params=None: [{"latest": date(2026, 11, 18)}],
    )
    for question in (
        "What is the recent, uh, data we have for this sales? Sales.",
        "what is themost recent data you have in the dubai sales data ?",
        "How recent is your sales data?",
    ):
        out = _data_freshness_turn(question)
        assert out is not None, question
        assert out["use_fast"] is True
        assert "documents" in out
        assert "18 Nov 2026" in out["fast_answer"] or "2026-11-18" in out["fast_answer"]


def test_recent_sales_listing_is_not_data_freshness():
    """'Show me recent sales' must stay on the sales-list path, not freshness."""
    assert _data_freshness_turn("Show me recent sales in Business Bay") is None
    assert _data_freshness_turn("Recent transactions for Hanover Square") is None


def test_transactions_for_a_named_place_ignore_the_projects_on_screen(monkeypatch):
    """'Recent sales in Dubai Marina' with JVC on screen must not use JVC projects."""
    import app.retrieval.propai_catalog as catalog

    class _Memory:
        active_candidates = [
            {"name": "HANOVER SQUARE", "entity_type": "project", "community": "Jumeirah Village Circle"},
        ]

    monkeypatch.setattr(catalog, "place_named", lambda text: "Dubai Marina" if "marina" in text.lower() else None)

    # Named place → None so _transaction_turn uses latest_sales(place=Marina).
    assert _transaction_subject("Show me the recent transactions in the dubai marina", _Memory()) is None
    # No place named: the projects on screen, as before.
    on_screen = _transaction_subject("Show me the recent transactions", _Memory())
    assert on_screen and on_screen[0]["name"] == "HANOVER SQUARE"
    # A named project on screen wins over a place.
    named = _transaction_subject("Recent transactions for Hanover Square in Dubai Marina", _Memory())
    assert named and named[0]["name"] == "HANOVER SQUARE"


def test_transaction_details_after_recommend_uses_on_screen_projects(monkeypatch):
    """After best-in-JVC cards, transaction details must not jump to random JVC sales."""
    import asyncio

    import app.retrieval.propai_catalog as catalog
    from app.memory.conversation_memory import ConversationMemory

    calls = {"recent": [], "latest": []}

    def fake_recent(names, per_project=5, offset=0):
        calls["recent"].append(list(names or []))
        return {
            "sales": [
                {
                    "project": "Hanover Square",
                    "date": "2026-10-15",
                    "price": 430_000,
                    "bedrooms": 0,
                    "size_sqft": 400,
                },
                {
                    "project": "The Square",
                    "date": "2026-10-10",
                    "price": 555_000,
                    "bedrooms": 1,
                    "size_sqft": 650,
                },
            ]
        }

    def fake_latest(**kwargs):
        calls["latest"].append(kwargs)
        return {
            "sales": [
                {
                    "project": "SQUAREX ONE",
                    "date": "2026-11-03",
                    "price": 1_291_709,
                    "bedrooms": 1,
                    "size_sqft": 764,
                }
            ]
        }

    monkeypatch.setattr(catalog, "recent_sales", fake_recent)
    monkeypatch.setattr(catalog, "latest_sales", fake_latest)

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "recommend",
        "goal": "invest",
        "place": "Jumeirah Village Circle",
        "source_question": "i am looking best projects in jvc",
    })
    memory.active_candidates = [
        {
            "name": "Hanover Square",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
        },
        {
            "name": "Q Gardens Lofts 2",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
        },
        {
            "name": "The Square",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
        },
    ]

    out = asyncio.run(
        __import__("run")._understood_turn(
            "Show me the recent transaction details",
            memory,
            {},
        )
    )
    assert out is not None
    answer = out["fast_answer"]
    assert "Hanover Square" in answer
    assert "The Square" in answer
    assert "SQUAREX ONE" not in answer
    assert calls["recent"], "expected recent_sales for on-screen projects"
    assert not calls["latest"], "must not fall back to area-wide latest_sales"


def test_cold_project_cards_force_sales_clarify():
    """After 2+ topic breaks, bare transaction details must clarify — not sticky cards."""
    from run import _project_card_context_is_live, _transaction_subject

    class _Memory:
        active_candidates = [
            {
                "name": "Hanover Square",
                "entity_type": "project",
                "community": "Jumeirah Village Circle",
            },
            {
                "name": "The Square",
                "entity_type": "project",
                "community": "Jumeirah Village Circle",
            },
        ]
        turns = [
            {
                "user_question": "best projects in jvc",
                "candidate_names": ["Hanover Square", "The Square"],
            },
            {"user_question": "show me rentals in Dubai Marina"},
            {"user_question": "top agents in Business Bay"},
        ]

        def get_dialogue_state(self):
            return {"action": "agents", "place": "Business Bay"}

    assert _project_card_context_is_live(
        "Show me the recent transaction details", _Memory()
    ) is False
    assert _transaction_subject(
        "Show me the recent transaction details", _Memory()
    ) is None
    # Deixis still unlocks the cold cards.
    assert _transaction_subject(
        "Show me recent transactions for these projects", _Memory()
    )


def test_fresh_cards_stay_sticky_after_one_soft_followup():
    from run import _transaction_subject

    class _Memory:
        active_candidates = [
            {"name": "Hanover Square", "entity_type": "project", "community": "JVC"},
            {"name": "The Square", "entity_type": "project", "community": "JVC"},
        ]
        turns = [
            {
                "user_question": "best projects in jvc",
                "candidate_names": ["Hanover Square", "The Square"],
            },
            {"user_question": "what is the handover date"},
        ]

        def get_dialogue_state(self):
            return {"action": "recommend", "place": "Jumeirah Village Circle"}

    found = _transaction_subject(
        "Show me the recent transaction details", _Memory()
    )
    assert found and found[0]["name"] == "Hanover Square"


def test_named_place_sales_wins_over_cold_or_fresh_cards(monkeypatch):
    import app.retrieval.propai_catalog as catalog
    from run import _transaction_subject

    class _Memory:
        active_candidates = [
            {"name": "Hanover Square", "entity_type": "project", "community": "JVC"},
        ]
        turns = []

        def get_dialogue_state(self):
            return {"action": "recommend", "place": "Jumeirah Village Circle"}

    monkeypatch.setattr(
        catalog,
        "place_named",
        lambda text: "Dubai Marina" if "marina" in (text or "").lower() else None,
    )
    assert (
        _transaction_subject(
            "Show me recent sales in Dubai Marina", _Memory()
        )
        is None
    )


def test_cold_cards_understood_turn_clarifies(monkeypatch):
    """Sample anytime flow: recommend → rent → agents → bare transactions → clarify."""
    import asyncio

    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "agents",
        "place": "Business Bay",
        "source_question": "top agents in Business Bay",
    })
    memory.active_candidates = [
        {
            "name": "Hanover Square",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
        },
        {
            "name": "The Square",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
        },
    ]
    memory.turns = [
        {
            "user_question": "best projects in jvc",
            "candidate_names": ["Hanover Square", "The Square"],
        },
        {"user_question": "show me rentals in Dubai Marina"},
        {"user_question": "top agents in Business Bay"},
    ]

    out = asyncio.run(
        __import__("run")._understood_turn(
            "Show me the recent transaction details",
            memory,
            {},
        )
    )
    assert out is not None
    assert out.get("structured_summary", {}).get("clarify") is True
    assert "project" in out["fast_answer"].lower() or "area" in out["fast_answer"].lower()


def test_best_project_in_marina_does_not_rerank_sticky_cards(monkeypatch):
    """'best project in Dubai Marina for family' must not use format_explain_on_screen."""
    from app.memory.conversation_memory import ConversationMemory
    from run import _shortlist_recommend_turn

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "recommend",
        "goal": "family",
        "kind": "filtered",
        "place": "Dubai Marina",
        "beds": 3,
    })
    memory.active_candidates = [
        {
            "name": "Ocean Heights",
            "entity_type": "project",
            "community": "Dubai Marina",
            "price": 1_500_000,
            "size_max": 1422,
        },
        {
            "name": "Marina Pinnacle",
            "entity_type": "project",
            "community": "Dubai Marina",
            "price": 1_900_000,
        },
        {
            "name": "Sulafa Tower",
            "entity_type": "project",
            "community": "Dubai Marina",
            "price": 1_899_000,
        },
    ]

    question = "suggest me the best project in Dubai Marina for living family"
    assert _shortlist_recommend_turn(question, memory) is None

    # Deixis shortlist still works on the same cards.
    which = _shortlist_recommend_turn(
        "which of these is the best for a family?",
        memory,
    )
    assert which is not None
    assert "on-screen" in which["fast_answer"].lower() or "acot recommend" in which["fast_answer"].lower()


def test_developer_chip_after_jvc_browse_lists_developers():
    """Live bug: developer chip clarified / re-browsed instead of on-screen developers."""
    import asyncio

    from app.answers.turn_resolve import offline_resolve_turn
    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "search",
        "place": "Jumeirah Village Circle",
        "source_question": "show me projects in JVC",
    })
    memory.active_candidates = [
        {
            "name": "Laya Crest",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
            "developer_name": "LAYA DEVELOPERS L.L.C.",
        },
        {
            "name": "Serenz",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
            "developer_name": "Danube",
        },
    ]

    chip = "Who is the developer of each of these projects?"
    offline = offline_resolve_turn(chip, memory)
    assert offline and offline["action"] == "attribute"

    out = asyncio.run(__import__("run")._understood_turn(chip, memory, {}))
    assert out is not None
    assert out.get("structured_summary", {}).get("clarify") is not True
    answer = out["fast_answer"].lower()
    assert "laya" in answer or "developer" in answer
    assert "serenz" in answer or "danube" in answer
    assert "i found" not in answer
    assert "200 projects" not in answer

    follow = "i want the developer details in JVC"
    out2 = asyncio.run(__import__("run")._understood_turn(follow, memory, {}))
    assert out2 is not None
    assert out2.get("structured_summary", {}).get("clarify") is not True
    answer2 = out2["fast_answer"].lower()
    assert "laya" in answer2 and "danube" in answer2
    # Must answer the shortlist developers — not a fresh 200-project browse.
    assert "200 projects" not in answer2
    projects = out2.get("structured_summary", {}).get("projects") or []
    assert len(projects) == 2


def test_top_agents_for_jvc_chip_does_not_list_projects(monkeypatch):
    """Live bug: agents chip after JVC browse returned 200 projects, not agents."""
    import asyncio

    from app.memory.conversation_memory import ConversationMemory
    from app.retrieval.propai_catalog import resolve_dubai_place

    assert resolve_dubai_place("Jumeirah Village Circle (JVC)") == "Jumeirah Village Circle"
    assert _broker_place(
        "Tell me the top agents for Jumeirah Village Circle (JVC)", None
    ) == "Jumeirah Village Circle"

    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "search",
        "place": "Jumeirah Village Circle",
        "source_question": "show me projects in JVC",
    })
    memory.active_candidates = [
        {
            "name": "The Weave",
            "entity_type": "project",
            "community": "Jumeirah Village Circle",
        },
    ]

    # Follow-up path must not hijack into project browse.
    follow = asyncio.run(
        __import__("run")._dialogue_followup_turn(
            "Tell me the top agents for Jumeirah Village Circle (JVC)",
            memory,
            {},
        )
    )
    assert follow is None

    out = asyncio.run(
        __import__("run")._understood_turn(
            "Tell me the top agents for Jumeirah Village Circle (JVC)",
            memory,
            {},
        )
    )
    assert out is not None
    assert out.get("structured_summary", {}).get("brokers")
    assert captured.get("place") == "Jumeirah Village Circle"
    assert "licensed agents" in out["fast_answer"].lower()
    assert "jumeirah village circle" in out["fast_answer"].lower()
    assert "1wood" not in out["fast_answer"].lower()
    assert "I found" not in out["fast_answer"]
    assert "200 projects" not in out["fast_answer"].lower()


def test_top_agents_after_fresh_motor_city_cards(monkeypatch):
    """Sample: Motor City projects on screen → bare 'i want top agents' uses Motor City."""
    import asyncio

    from app.memory.conversation_memory import ConversationMemory

    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "search",
        "place": "Motor City",
        "source_question": "show me projects in Motor City",
    })
    memory.active_candidates = [
        {
            "name": "Ananda Residences",
            "entity_type": "project",
            "community": "Motor City",
        },
        {
            "name": "Regent House",
            "entity_type": "project",
            "community": "Motor City",
        },
    ]

    out = asyncio.run(
        __import__("run")._understood_turn(
            "i want top agents",
            memory,
            {},
        )
    )
    assert out is not None
    assert out.get("structured_summary", {}).get("clarify") is not True
    assert out.get("structured_summary", {}).get("brokers")
    assert captured.get("place") == "Motor City"
    assert "motor city" in out["fast_answer"].lower()
    assert "licensed agents" in out["fast_answer"].lower()
    # Must not fall back to a project browse answer.
    assert "I found" not in out["fast_answer"]
    assert "35 projects" not in out["fast_answer"].lower()


def test_top_agents_cold_cards_clarifies():
    """After 2+ topic breaks, bare top agents clarifies — does not invent Motor City."""
    import asyncio

    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "recommend",
        "place": "Dubai Marina",
        "source_question": "best family areas",
    })
    memory.active_candidates = [
        {
            "name": "Ananda Residences",
            "entity_type": "project",
            "community": "Motor City",
        },
        {
            "name": "Regent House",
            "entity_type": "project",
            "community": "Motor City",
        },
    ]
    memory.turns = [
        {
            "user_question": "show me projects in Motor City",
            "candidate_names": ["Ananda Residences", "Regent House"],
        },
        {"user_question": "show me rentals in Dubai Marina"},
        {"user_question": "best communities for a family with kids"},
    ]

    out = asyncio.run(
        __import__("run")._understood_turn(
            "i want top agents",
            memory,
            {},
        )
    )
    assert out is not None
    assert out.get("structured_summary", {}).get("clarify") is True
    assert "agent" in out["fast_answer"].lower()
    # Must not invent sticky Motor City in the Understood line.
    assert "in motor city" not in out["fast_answer"].lower()


def test_top_agents_named_place_ignores_motor_city_cards(monkeypatch):
    import asyncio

    from app.memory.conversation_memory import ConversationMemory

    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return _broker_rows()[: kwargs.get("limit", 5)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "search",
        "place": "Motor City",
        "source_question": "show me projects in Motor City",
    })
    memory.active_candidates = [
        {
            "name": "Ananda Residences",
            "entity_type": "project",
            "community": "Motor City",
        },
    ]

    out = asyncio.run(
        __import__("run")._understood_turn(
            "top agents in Business Bay",
            memory,
            {},
        )
    )
    assert out is not None
    assert captured.get("place") == "Business Bay"
    assert "business bay" in out["fast_answer"].lower()


def test_full_community_list_is_answered_from_the_data(monkeypatch):
    """'Full list of communities' gets every community (not a top 10), even with a typo."""
    import app.database.postgres as postgres
    from run import _community_list_turn

    def fake_fetch(sql, *args):
        # Prop-AI only — never the retired listings table.
        assert "from listings" not in sql
        if "from propai_places" in sql:
            return [
                {"name": "Jumeirah Village Circle", "n": 100},
                {"name": "Dubai Marina", "n": 80},
            ]
        return [
            {"name": "Jumeirah Village Circle", "n": 5},
            {"name": "Dubai Creek Harbour", "n": 4},
        ]

    monkeypatch.setattr(postgres, "fetch", fake_fetch)
    answer = _community_list_turn("i need full list of the communintes")["fast_answer"]
    assert answer.startswith("We have 3 communities")
    assert "| Jumeirah Village Circle | 5 | 100 |" in answer
    assert "| Dubai Creek Harbour | 4 |" in answer
    assert _community_list_turn("what are the communties we have") is not None
    # Questions that narrow the list go to the normal answer.
    assert _community_list_turn("best communities for families") is None
    assert _community_list_turn("Tell me about Dubai Marina") is None



def test_broker_question_with_a_place_is_not_read_as_a_name(monkeypatch):
    """'can you find me a broker for the jvc' asks for JVC agents, not an agent named 'you a jvc'."""
    seen = []

    def fake_list(**kwargs):
        seen.append(kwargs)
        return _broker_rows()[: kwargs.get("limit", 8)]

    monkeypatch.setattr("app.retrieval.propai_catalog.list_brokers", fake_list)
    turn = _broker_turn("can you find me a broker for the jvc", None)
    assert seen[0].get("name") is None
    assert seen[0].get("place") == "Jumeirah Village Circle"
    assert "Here are licensed agents for Jumeirah Village Circle." in turn["fast_answer"]
    assert "Understood:" in turn["fast_answer"]
    assert "licence" in turn["fast_answer"].lower() or "licensed" in turn["fast_answer"].lower()


def test_agency_question_goes_to_the_broker_cards():
    # "tell me top agency JVC" used to fall through to the model, which listed every agent as text above the cards.
    import run

    assert run._BROKER.search("tell me top agency JVC")
    assert run._BROKER.search("best real estate agencies in Dubai Marina")


def test_repair_does_not_rewrite_hanover_into_handover():
    fixed = repair_typed_question(
        "What is the cheapest 1-bedroom sale in Hanover Square JVC?"
    )
    assert "Hanover Square" in fixed
    assert "handover Square" not in fixed


def test_sale_count_turn_answers_how_many_sold(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "count_sales",
        lambda **kwargs: {"count": 47, "first_sale": None, "last_sale": None},
    )
    monkeypatch.setattr(catalog, "resolve_dubai_place", lambda place: "Jumeirah Village Circle")
    monkeypatch.setattr(catalog, "place_named", lambda text: "Jumeirah Village Circle")
    memory = type("M", (), {"sale_query": None, "transaction_names": None, "transaction_offset": 0})()
    out = _transaction_turn(
        "How many 1-bedroom apartments in JVC were sold in November 2026?",
        memory,
    )
    assert out is not None
    assert "47" in out["fast_answer"]
    assert "November 2026" in out["fast_answer"]
    assert "Recent Transactions" not in out["fast_answer"]
    assert memory.sale_query["place"] == "Jumeirah Village Circle"
    assert memory.sale_query["bedrooms"] == 1
    assert memory.sale_query["month"] == 11
    assert memory.sale_query["year"] == 2026


def test_ok_show_me_after_sale_count_lists_those_sales(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    class _Memory:
        sale_query = {
            "place": "Jumeirah Village Circle",
            "bedrooms": 1,
            "year": 2026,
            "month": 11,
            "count": 47,
        }
        transaction_names = "COUNT|JUMEIRAH VILLAGE CIRCLE|B1|2026-11"
        transaction_offset = 0
        last_user_question = (
            "How many 1-bedroom apartments in JVC were sold in November 2026?"
        )
        active_candidates = []

    seen = {}

    def fake_latest_sales(**kwargs):
        seen.update(kwargs)
        return {
            "count": 2,
            "sales": [
                {
                    "project": "HANOVER SQUARE",
                    "date": "2026-11-03",
                    "price": 700000,
                    "bedrooms": 1,
                    "size_sqft": 660,
                    "property_type": "Unit",
                }
            ],
        }

    monkeypatch.setattr(catalog, "latest_sales", fake_latest_sales)
    out = _transaction_turn("ok show me", _Memory())
    assert out is not None
    assert seen.get("place") == "Jumeirah Village Circle"
    assert seen.get("bedrooms") == 1
    assert seen.get("year") == 2026
    assert seen.get("month") == 11
    assert "HANOVER SQUARE" in out["fast_answer"]
    assert "2026-11-03" in out["fast_answer"]

    yes = _transaction_turn("yes show", _Memory())
    assert yes is not None
    assert "HANOVER SQUARE" in yes["fast_answer"]


def test_show_me_more_after_sales_pages_from_dialogue_state(monkeypatch):
    import app.retrieval.propai_catalog as catalog
    from app.memory.conversation_memory import ConversationMemory

    calls = []

    def fake_latest_sales(**kwargs):
        calls.append(kwargs)
        offset = int(kwargs.get("offset") or 0)
        if offset > 0:
            return {
                "count": 1,
                "sales": [{
                    "project": "NEXT PROJECT",
                    "date": "2026-10-01",
                    "price": 900000,
                    "bedrooms": 1,
                    "size_sqft": 700,
                    "property_type": "Unit",
                }],
            }
        return {
            "count": 1,
            "sales": [{
                "project": "FIRST PROJECT",
                "date": "2026-11-03",
                "price": 1000000,
                "bedrooms": 1,
                "size_sqft": 800,
                "property_type": "Unit",
            }],
        }

    monkeypatch.setattr(catalog, "latest_sales", fake_latest_sales)
    memory = ConversationMemory(llm_client=None)
    first = _transaction_turn("Show me recent sales in Business Bay", memory)
    assert first is not None
    assert "FIRST PROJECT" in first["fast_answer"]
    assert memory.get_dialogue_state()["action"] == "sales_list"
    assert memory.get_dialogue_state()["place"] == "Business Bay"

    more = _transaction_turn("show me more", memory)
    assert more is not None
    assert "NEXT PROJECT" in more["fast_answer"]
    assert calls[-1].get("offset", 0) > 0
    assert "I don't have a project called" not in more["fast_answer"]


def test_cheapest_sale_turn_uses_named_project(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "project_named_in_question",
        lambda text: "HANOVER SQUARE",
    )
    monkeypatch.setattr(
        catalog,
        "cheapest_sale",
        lambda **kwargs: {
            "project": "HANOVER SQUARE",
            "date": "2025-12-22",
            "price": 530000,
            "bedrooms": 1,
            "size_sqft": 660,
            "property_type": "Unit",
        },
    )
    monkeypatch.setattr(
        catalog,
        "sale_window_stats",
        lambda **kwargs: {"count": 42, "low": 530000, "high": 910000, "median": 700000},
    )
    out = _transaction_turn(
        "What is the cheapest 1-bedroom sale in Hanover Square JVC in the last 12 months?",
        None,
    )
    assert out is not None
    answer = out["fast_answer"]
    assert "530,000" in answer
    assert "HANOVER SQUARE" in answer
    assert out["structured_summary"]["transactions"][0]["price"] == 530000
    # Explain-first: restates the ask, then gives evidence for the number.
    assert answer.startswith("Understood:")
    assert "lowest" in answer.split("\n", 1)[0]
    assert "lowest of 42 recorded" in answer
    assert "median AED 700,000" in answer
    assert "Dubai Land Department" in answer
    # Never the generic sales list.
    assert "recent transaction details" not in answer


def test_which_one_has_cheapest_transactions_picks_on_screen_winner(monkeypatch):
    """After a sales dump, compare asks use context — not a transaction synonym bank."""
    import asyncio

    import app.retrieval.propai_catalog as catalog
    from app.answers.turn_resolve import offline_resolve_turn
    from app.memory.conversation_memory import ConversationMemory
    from app.retrieval.hybrid.question_text import repair_typed_question
    from run import _cheapest_sale_turn, _sale_stat_kind, _understood_turn

    typed = "which one has chepest transcations"
    fixed = repair_typed_question(typed)
    assert "cheapest" in fixed.lower()
    # Typo repair is vocabulary edit-distance — routing uses dialogue context.
    memory_probe = ConversationMemory(llm_client=None)
    memory_probe.set_dialogue_state({
        "action": "sales_list",
        "place": "Jumeirah Village Circle",
        "limit": 5,
    })
    # Even without the word "transaction", prior sales + "which one is cheapest" → sales_stat.
    bare = offline_resolve_turn("which one is cheapest?", memory_probe)
    assert bare["action"] == "sales_stat"
    assert bare["stat"] == "cheapest"
    assert offline_resolve_turn(fixed, memory_probe)["action"] == "sales_stat"
    assert _sale_stat_kind(fixed) == "cheapest"

    prices = {
        "SERENZ BY DANUBE": 972400,
        "ALTA VIEW SKYHOMES": 1108896,
        "SKY LEVEL 1": 928857,
        "THE HUDSON": 669000,
        "SKY HARMONY BY PEACE HOMES": 650000,
    }

    def fake_cheapest(**kwargs):
        name = str(kwargs.get("project") or "").strip().upper()
        price = prices.get(name)
        if price is None:
            return None
        return {
            "project": kwargs.get("project"),
            "date": "2026-10-29",
            "price": price,
            "bedrooms": 0,
            "size_sqft": 394,
            "property_type": "Unit",
        }

    monkeypatch.setattr(catalog, "cheapest_sale", fake_cheapest)
    monkeypatch.setattr(catalog, "project_named_in_question", lambda text: None)
    monkeypatch.setattr(
        catalog,
        "sale_window_stats",
        lambda **kwargs: {"count": 25, "low": 650000, "high": 2915000, "median": 1100000},
    )

    memory = ConversationMemory(llm_client=None)
    memory.active_candidates = [
        {"name": name, "entity_type": "project", "community": "Jumeirah Village Circle"}
        for name in prices
    ]
    memory.set_dialogue_state({
        "action": "sales_list",
        "place": "Jumeirah Village Circle",
        "limit": 5,
    })

    out = _cheapest_sale_turn(
        typed,
        slots={"stat": "cheapest"},
        conversation_memory=memory,
    )
    assert out is not None
    answer = out["fast_answer"]
    assert "SKY HARMONY" in answer.upper()
    assert "650,000" in answer
    assert "recent transaction details" not in answer.lower()
    assert "Acot Recommend" in answer

    # Full understood path with typos must not re-dump the sales list.
    full = asyncio.run(_understood_turn(typed, memory, {}))
    assert full is not None
    assert "650,000" in full["fast_answer"]
    assert "recent transaction details" not in full["fast_answer"].lower()


def test_cheapest_question_is_sales_stat_not_sales_list_even_after_sales_list(monkeypatch):
    """Regression: 'cheapest ... in Hanover Square' returned a JVC sales list."""
    import asyncio

    import app.retrieval.propai_catalog as catalog
    from app.answers.followup_resolve import looks_like_state_followup, resolve_followup
    from app.answers.turn_resolve import offline_resolve_turn
    from app.memory.conversation_memory import ConversationMemory
    from run import _dialogue_followup_turn, _understood_turn

    question = (
        "What is the cheapest 1-bedroom sale in Hanover Square JVC in the last 12 months?"
    )
    resolution = offline_resolve_turn(question)
    assert resolution["action"] == "sales_stat"
    assert resolution["beds"] == 1

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "sales_list",
        "place": "Jumeirah Village Circle",
        "beds": 1,
        "limit": 5,
    })
    # The sticky sales task must not swallow a fresh superlative question.
    resolved = asyncio.run(
        resolve_followup(question, memory.get_dialogue_state(), llm_generate=None)
    )
    assert resolved is None or resolved["kind"] == "new"
    assert asyncio.run(_dialogue_followup_turn(question, memory, {})) is None

    monkeypatch.setattr(catalog, "project_named_in_question", lambda text: "HANOVER SQUARE")
    monkeypatch.setattr(
        catalog,
        "cheapest_sale",
        lambda **kwargs: {
            "project": "HANOVER SQUARE",
            "date": "2025-12-22",
            "price": 530000,
            "bedrooms": 1,
            "size_sqft": 660,
            "property_type": "Unit",
        },
    )
    monkeypatch.setattr(
        catalog,
        "sale_window_stats",
        lambda **kwargs: {"count": 42, "low": 530000, "high": 910000, "median": 700000},
    )
    out = asyncio.run(_understood_turn(question, memory, {}))
    assert out is not None
    assert "AED 530,000" in out["fast_answer"]
    assert "recent transaction details" not in out["fast_answer"]
    assert "BINGHATTI" not in out["fast_answer"]


def test_sales_stat_follows_llm_slots_not_phrase_regexes(monkeypatch):
    """Paraphrase with none of cheapest/lowest/sale: only the LLM slots can route it."""
    import asyncio
    import json

    import app.retrieval.propai_catalog as catalog
    from app.memory.conversation_memory import ConversationMemory
    from app.answers.turn_resolve import _SALES_STAT_SIGNAL, offline_resolve_turn
    from run import _sale_stat_kind, _understood_turn

    from app.answers.turn_resolve import resolve_turn

    question = "What's the least I'd have to pay for a one-bed at Hanover Square this past year?"
    # Offline regex must NOT understand this paraphrase...
    assert _sale_stat_kind(question) is None
    assert not _SALES_STAT_SIGNAL.search(question)
    assert (offline_resolve_turn(question) or {}).get("action") != "sales_stat"

    class _FakeLLM:
        async def generate(self, prompt):
            return json.dumps({
                "action": "sales_stat",
                "stat": "cheapest",
                "project": "Hanover Square",
                "place": None,
                "beds": 1,
                "months": 12,
                "confidence": 0.95,
            })

    seen = {}

    def fake_cheapest(**kwargs):
        seen.update(kwargs)
        return {
            "project": "HANOVER SQUARE",
            "date": "2025-12-22",
            "price": 530000,
            "bedrooms": 1,
            "size_sqft": 660,
            "property_type": "Unit",
        }

    monkeypatch.setattr(catalog, "project_named_in_question", lambda text: "HANOVER SQUARE")
    monkeypatch.setattr(catalog, "cheapest_sale", fake_cheapest)
    monkeypatch.setattr(
        catalog,
        "sale_window_stats",
        lambda **kwargs: {"count": 42, "low": 530000, "high": 910000, "median": 700000},
    )
    memory = ConversationMemory(llm_client=_FakeLLM())
    resolved = asyncio.run(
        resolve_turn(question, memory, llm_generate=memory.llm_client.generate)
    )
    assert resolved["action"] == "sales_stat"
    assert resolved["source"] == "llm"
    assert resolved["stat"] == "cheapest"
    assert resolved["project"] == "Hanover Square"

    out = asyncio.run(_understood_turn(question, memory, {}))
    assert out is not None
    assert seen["project"] == "HANOVER SQUARE"
    assert seen["bedrooms"] == 1
    assert seen["months"] == 12
    assert seen["highest"] is False
    answer = out["fast_answer"]
    assert answer.startswith("Understood:")
    assert "AED 530,000" in answer
    assert "recent transaction details" not in answer


def test_conflict_studio_and_villa_forces_split_not_sticky_search(monkeypatch):
    """LLM wrongly says search+JVC → still split with explain-first, no place invent."""
    import asyncio
    import json

    import app.retrieval.propai_catalog as catalog
    from app.answers.turn_resolve import resolve_turn
    from app.memory.conversation_memory import ConversationMemory
    from run import _understood_turn

    question = (
        "The budget is AED 900,000. I need a 3-bedroom villa and, "
        "at the same time, only a studio apartment."
    )

    class _FakeLLM:
        async def generate(self, prompt):
            # Deliberately wrong: single search with sticky community.
            return json.dumps({
                "action": "search",
                "place": "Jumeirah Village Circle",
                "beds": 3,
                "budget_aed": 900000,
                "confidence": 0.95,
            })

    monkeypatch.setattr(
        catalog,
        "agent_projects",
        lambda filters, limit=8, sort="cheapest", skip_names=None: (
            [
                {
                    "id": "P1",
                    "name": "Studio One",
                    "price": 700000,
                    "developer_name": "Dev",
                    "community": "Marina",
                    "bedroom_min": 0,
                    "bedroom_max": 0,
                }
            ] * 2,
            2,
        ),
    )

    memory = ConversationMemory(llm_client=_FakeLLM())
    memory.set_dialogue_state({
        "action": "search",
        "place": "Jumeirah Village Circle",
        "beds": 2,
    })
    resolved = asyncio.run(
        resolve_turn(question, memory, llm_generate=memory.llm_client.generate)
    )
    assert resolved["action"] == "recommend"
    assert resolved["kind"] == "split"
    assert resolved.get("place") in (None, "")
    assert resolved.get("beds") is None

    out = asyncio.run(_understood_turn(question, memory, {}))
    assert out is not None
    answer = out["fast_answer"]
    assert answer.startswith("Understood:")
    assert "Acot Recommend" in answer
    assert "studio" in answer.lower()
    assert "villa" in answer.lower() or "3-bedroom" in answer.lower()
    assert "prop-ai" not in answer.lower() and "propai" not in answer.lower()
    assert "cannot" not in answer.lower()
    assert "Jumeirah Village Circle" not in answer.split("\n")[0]
    assert "I found 136 projects" not in answer


def test_highest_and_average_sale_asks_use_exact_aggregates(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    seen = {}

    def fake_extreme(**kwargs):
        seen.update(kwargs)
        return {
            "project": "BURJ VISTA",
            "date": "2026-10-01",
            "price": 9_000_000,
            "bedrooms": 2,
            "size_sqft": 1800,
            "property_type": "Unit",
        }

    monkeypatch.setattr(catalog, "project_named_in_question", lambda text: None)
    monkeypatch.setattr(catalog, "place_named", lambda text: "Downtown Dubai")
    monkeypatch.setattr(catalog, "resolve_dubai_place", lambda place: "Downtown Dubai")
    monkeypatch.setattr(catalog, "cheapest_sale", fake_extreme)
    monkeypatch.setattr(
        catalog,
        "sale_window_stats",
        lambda **kwargs: {"count": 10, "low": 900_000, "high": 9_000_000, "median": 2_000_000},
    )
    high = _transaction_turn("What is the highest sale price in Downtown?", None)
    assert high is not None
    assert seen.get("highest") is True
    assert "AED 9,000,000" in high["fast_answer"]
    assert high["fast_answer"].startswith("Understood:")
    assert "highest of 10 recorded" in high["fast_answer"]

    avg = _cheapest_sale_turn("What is the average sale price of 2-bedroom in Downtown?")
    assert avg is not None
    assert "median price is **AED 2,000,000**" in avg["fast_answer"]
    assert "AED 900,000 to AED 9,000,000" in avg["fast_answer"]
    assert avg["fast_answer"].startswith("Understood:")


def test_project_count_turn_answers_from_propai(monkeypatch):
    import app.retrieval.propai_catalog as catalog
    from app.memory.conversation_memory import ConversationMemory

    monkeypatch.setattr(catalog, "count_projects", lambda: 4110)
    memory = ConversationMemory(llm_client=None)
    out = _project_count_turn(
        "How many residential projects are there in Dubai?",
        memory,
    )
    assert out is not None
    assert "4,110" in out["fast_answer"]
    assert "show me" in out["fast_answer"].lower()
    state = memory.get_dialogue_state()
    assert state["action"] == "project_count"
    assert state["place"] == "Dubai"
    assert state["last_result"]["count"] == 4110


def test_developer_rank_turn_lists_top_developers(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(
        catalog,
        "top_developers_by_value",
        lambda limit=5, months=12: [
            {"developer": "EMAAR", "sales": 100, "value": 50_000_000_000},
            {"developer": "DAMAC PROPERTIES", "sales": 80, "value": 20_000_000_000},
        ],
    )
    out = _developer_rank_turn(
        "Which 5 developers had the highest sales value in Dubai in the last 12 months?"
    )
    assert out is not None
    assert "EMAAR" in out["fast_answer"]
    assert "DAMAC PROPERTIES" in out["fast_answer"]


def test_sql_browse_passes_min_price_and_ready(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    seen = {}

    def fake_agent_projects(filters, limit=10, sort="latest_handover", skip_names=None):
        seen["filters"] = filters
        seen["sort"] = sort
        return (
            [{
                "name": "Lunaya",
                "developer_name": "Zaya",
                "community": "Dubai",
                "price": 6440000,
                "status": "Secondary Ready",
                "project_status": "Secondary Ready",
            }],
            1,
        )

    monkeypatch.setattr(catalog, "agent_projects", fake_agent_projects)
    monkeypatch.setattr(catalog, "resolve_dubai_place", lambda place: place)

    above = _sql_project_browse_turn(
        "Show me projects in Dubai starting above AED 5 million", None
    )
    assert above is not None
    assert seen["filters"].get("min_price") == 5_000_000
    assert seen["sort"] == "priciest"
    assert not seen["filters"].get("communities")
    assert "Jumeirah Village Circle" not in (above.get("fast_answer") or "")


def test_latest_sales_keeps_named_project_building_deals(monkeypatch):
    """AC3-style project asks must not be emptied by the unit sqft outlier band."""
    import app.retrieval.propai_catalog as catalog

    calls = {"sql": []}

    def fake_fetch(sql, params=None):
        calls["sql"].append((sql, params))
        text = " ".join(str(sql).lower().split())
        if "from propai_projects" in text and "upper(btrim(name))" in text:
            return [{"name": "AC3", "master_project": "Motor City"}]
        if "from propai_sales" in text:
            assert "procedure_area_sqft between 50 and 100000" not in text
            assert params and "AC3" in params
            return [{
                "project": "AC3",
                "sold_on": "2025-12-18",
                "price": 182_100_000,
                "bedrooms": None,
                "size_sqft": 145669,
                "property_type": "Building",
            }]
        return []

    monkeypatch.setattr(catalog, "fetch", fake_fetch)
    out = catalog.latest_sales(place="AC3", limit=8)
    assert out["count"] == 1
    assert out["sales"][0]["project"] == "AC3"
    assert out["sales"][0]["price"] == 182_100_000


def test_latest_sales_matches_short_sale_alias_for_woods(monkeypatch):
    """Catalog 'The Woods at Sobha Sanctuary' must find sale-table 'THE WOODS'."""
    import app.retrieval.propai_catalog as catalog

    def fake_fetch(sql, params=None):
        text = " ".join(str(sql).lower().split())
        if "from propai_projects" in text:
            return [{
                "name": "The Woods at Sobha Sanctuary",
                "master_project": "SOBHA SANCTUARY",
            }]
        if "from propai_sales" in text:
            # Short alias + master must both be in the filter params.
            flat = [str(p) for p in (params or ())]
            assert any("Woods at Sobha Sanctuary" in p for p in flat)
            assert any(p.upper() == "THE WOODS" or p == "The Woods" for p in flat)
            assert any("SOBHA SANCTUARY" in p.upper() for p in flat)
            return [{
                "project": "THE WOODS",
                "sold_on": "2026-11-18",
                "price": 2_500_000,
                "bedrooms": 3,
                "size_sqft": 2200,
                "property_type": "Villa",
            }]
        return []

    monkeypatch.setattr(catalog, "fetch", fake_fetch)
    out = catalog.latest_sales(place="The Woods at Sobha Sanctuary", limit=5)
    assert out["count"] == 1
    assert out["sales"][0]["project"] == "THE WOODS"


def test_dubai_above_5m_after_jvc_state_stays_citywide(monkeypatch):
    """Sticky JVC must not hijack an explicit Dubai min-price browse."""
    import asyncio

    import app.retrieval.propai_catalog as catalog
    from app.memory.conversation_memory import ConversationMemory

    seen = {}

    def fake_browse(filters, limit=10, sort="latest_handover", skip_names=None):
        seen["filters"] = dict(filters or {})
        seen["sort"] = sort
        return (
            [{
                "name": "Lunaya",
                "developer_name": "Zaya",
                "community": "Palm Jumeirah",
                "price": 6_440_000,
            }],
            1,
            None,
        )

    monkeypatch.setattr(catalog, "browse_projects_near_budget", fake_browse)
    monkeypatch.setattr(catalog, "resolve_dubai_place", lambda place: place)

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "search",
        "place": "Jumeirah Village Circle",
        "beds": 1,
        "budget_aed": 700_000,
        "ready_only": True,
        "source_question": "Show only ready 1-bedroom apartments in JVC under AED 700,000",
    })

    question = "Show me projects in Dubai starting above AED 5 million"
    out = asyncio.run(
        __import__("run")._understood_turn(question, memory, {})
    )
    assert out is not None
    assert seen.get("filters", {}).get("min_price") == 5_000_000
    assert not seen.get("filters", {}).get("communities")
    assert seen.get("filters", {}).get("bedrooms") is None
    assert not seen.get("filters", {}).get("ready_only")
    answer = (out.get("fast_answer") or "").lower()
    assert "jumeirah village circle" not in answer
    assert "lunaya" in answer or "project" in answer


def test_sql_browse_ready_jvc_answer_explains_filters(monkeypatch):
    import app.retrieval.propai_catalog as catalog

    seen = {}

    def fake_ready(filters, limit=10, sort="latest_handover", skip_names=None):
        seen["filters"] = filters
        seen["sort"] = sort
        return (
            [{
                "name": "Chaimaa Premiere",
                "developer_name": "X",
                "community": "Jumeirah Village Circle",
                "price": 600000,
                "bedroom_min": 1,
                "bedroom_max": 1,
                "status": "Secondary Ready",
                "project_status": "Secondary Ready",
            }],
            1,
        )

    monkeypatch.setattr(catalog, "agent_projects", fake_ready)
    question = "Show only ready 1-bedroom apartments in JVC under AED 700,000"
    ready = _sql_project_browse_turn(question, None)
    assert ready is not None
    assert seen["filters"].get("ready_only") is True
    assert seen["filters"].get("max_price") == 700_000
    assert seen["filters"].get("bedrooms") == 1
    assert seen["filters"].get("communities") == ["Jumeirah Village Circle"]

    # Answer text must explain the filters — not a bare "Bedrooms: 1" stub list.
    answer = _build_fast_structured_answer(question, ready["structured_summary"])
    assert "Bedrooms:" not in answer or "Recent sale" in answer or "ready" in answer.lower()
    assert "ready" in answer.lower()
    assert "1-bedroom" in answer.lower() or "1 bedroom" in answer.lower()
    assert "700,000" in answer or "700000" in answer.replace(",", "")
    assert "Jumeirah Village Circle" in answer or "JVC" in answer
    assert "Chaimaa Premiere" in answer
    assert "Found 10 project(s) matching your request" not in answer


def test_other_city_is_declined_not_mapped_onto_dubai():
    from run import _outside_dubai_market_reply

    reply = _outside_dubai_market_reply('Best family areas in Abu Dhabi.')
    assert reply and 'only covers Dubai' in reply and 'Abu Dhabi' in reply
    assert _outside_dubai_market_reply('Best family areas in Dubai') is None
    assert _outside_dubai_market_reply('Top agents in Abu Dhabi') is None


def test_arabian_ranches_3_stays_on_the_numbered_master():
    from app.retrieval.propai_catalog import place_named

    assert place_named('I want a 3-bedroom in Arabian Ranches 3 around AED 2.8M.') == 'Arabian Ranches III'
    assert place_named('typical sale in Arabian Ranches only') == 'Arabian Ranches'

