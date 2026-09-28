from app.retrieval.hybrid.entity_resolver import EntityResolver
from app.retrieval.hybrid.hybrid_retriever import HybridRetriever
from app.retrieval.hybrid.query_planner import QueryPlanner
from app.retrieval.hybrid.question_text import (
    interpret_search,
    is_group_attribute_question,
    prepare_user_question,
)
from run import (
    _build_fast_structured_answer,
    _classify_small_talk,
    _detect_response_type,
    _followup_questions,
    _is_fast_structured_question,
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
