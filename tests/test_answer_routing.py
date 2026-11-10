from app.retrieval.hybrid.entity_resolver import EntityResolver
from app.retrieval.hybrid.hybrid_retriever import HybridRetriever
from app.retrieval.hybrid.query_planner import QueryPlanner
from app.retrieval.hybrid.question_text import (
    interpret_search,
    is_group_attribute_question,
    prepare_user_question,
)
from run import (
    _build_comparison,
    _build_fast_structured_answer,
    _classify_small_talk,
    _detect_response_type,
    _followup_questions,
    _is_fast_structured_question,
    _format_transactions,
    _listing_question,
    _compare_shown_turn,
    _shortlist_recommend_turn,
    _shown_project_followup,
    _transaction_subject,
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
    assert answer.startswith("I found 5")
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


def test_investment_answer_with_projects_shows_cards():
    question = "Find me a good investment property in Business Bay"
    assert _detect_response_type(question, "investment", _summary()) == "project_list"
    assert _detect_response_type(
        question, "investment", _summary(projects=[])
    ) == "investment_analysis"


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
    assert not _is_fast_structured_question(
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
    assert not _is_fast_structured_question(
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


def test_bedroom_follow_up_skips_fast_path():
    question = "Which of these have 2-bedroom options?"
    plan = _planner()._rule_plan(question)

    assert plan.intent == "information"
    assert not _is_fast_structured_question(
        question,
        plan.intent,
        _summary(),
        [],
    )


def test_comparison_skips_fast_path():
    question = (
        "Compare the first and second ones based on price, "
        "bedroom range, and handover date."
    )
    summary = _summary()
    plan = _planner()._rule_plan(question)

    assert plan.intent == "comparison"
    assert not _is_fast_structured_question(
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
    assert "lower yearly rent" in comparison["answer"]
    assert "more space" in comparison["answer"]


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
    ):
        assert _classify_small_talk(text) == "greeting", text
    assert _classify_small_talk("good night") == "bye"
    for text in (
        "good projects in Marina", "good morning show me projects",
        "how much is a 2 bedroom", "morning view apartments",
    ):
        assert _classify_small_talk(text) is None, text


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
    assert text.index("Mayfair Tower") < text.index("Mayfair Residency")


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
    assert "AED 1395448" in answer
    assert "exact match" not in answer.lower()
    assert _shown_project_followup("Show me projects in Dubai Marina", _Memory()) is None


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
