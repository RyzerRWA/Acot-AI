from app.answers.recommend import (
    _budget_fit_scores,
    format_compare_table,
    format_footer,
    format_ranked_picks,
    format_shortlist,
    rank_projects,
)


def test_named_compare_picks_clearer_value_and_activity():
    projects = [
        {
            "id": "P1",
            "name": "Alpha Tower",
            "price": 2_000_000,
            "bedroom_min": 1,
            "bedroom_max": 2,
            "size_sqft": 700,
            "handover_time": "2028-01-01",
            "status": "On Sale",
            "sales_count": 3,
            "rent_count": 2,
        },
        {
            "id": "P2",
            "name": "Beta Residences",
            "price": 1_400_000,
            "bedroom_min": 1,
            "bedroom_max": 3,
            "size_sqft": 950,
            "handover_time": "2027-01-01",
            "status": "Ready",
            "sales_count": 40,
            "rent_count": 25,
        },
    ]
    result = rank_projects(projects, attach_sales=False)
    assert result["can_rank"] is True
    assert result["winner"]["name"] == "Beta Residences"
    footer = format_footer(result)
    assert "**Acot Recommend:** Beta Residences" in footer
    assert "Why this is the strongest match" in footer
    assert "Watch-outs" not in footer
    assert "yield" not in footer.lower()
    assert "roi" not in footer.lower()
    assert "AED 1,400,000" in footer or "sales" in footer.lower() or "rental" in footer.lower()


def test_thin_data_stays_positive():
    projects = [
        {"name": "Only Name A"},
        {"name": "Only Name B"},
    ]
    result = rank_projects(projects, attach_sales=False)
    footer = format_footer(result)
    assert "**Acot Recommend:**" in footer
    assert result["winner"] is None
    assert "Only Name A" not in footer
    assert "Only Name B" not in footer
    assert "can't" not in footer.lower()
    assert "no clear" not in footer.lower()
    assert "missing" not in footer.lower()
    assert "yield" not in footer.lower()
    assert "evidence" in footer.lower() or "transaction" in footer.lower() or "thin" in footer.lower()


def test_rank_projects_drops_handover_only_for_any_goal():
    pool = [
        {"name": "Forest City Tower 3", "handover_time": "2032-01-01", "status": "On Sale"},
        {
            "name": "Sale Backed Court",
            "price": 1_200_000,
            "price_kind": "recent_sale",
            "sales_count": 22,
            "rent_count": 9,
            "last_sold_on": "2026-06-01",
            "bedroom_min": 2,
            "bedroom_max": 3,
            "status": "Ready",
        },
    ]
    for goal in ("family", "ready", "compare", "budget", "invest"):
        result = rank_projects(pool, attach_sales=False, goal=goal)
        assert result["winner"]["name"] == "Sale Backed Court", goal
        names = [item["name"] for item in result["ranked"]]
        assert "Forest City Tower 3" not in names, goal


def test_shortlist_format_includes_ranked_list():
    projects = [
        {"name": "One", "price": 900_000, "bedroom_min": 1, "bedroom_max": 1, "size_sqft": 650, "status": "Ready", "sales_count": 10, "rent_count": 8},
        {"name": "Two", "price": 1_500_000, "bedroom_min": 2, "bedroom_max": 2, "size_sqft": 900, "status": "On Sale", "sales_count": 2, "rent_count": 1},
        {"name": "Three", "price": 1_200_000, "bedroom_min": 1, "bedroom_max": 2, "size_sqft": 800, "status": "Ready", "sales_count": 8, "rent_count": 6},
    ]
    result = rank_projects(projects, attach_sales=False)
    text = format_shortlist(result, question="Which of these is best?")
    assert "1. **" in text
    assert "**Acot Recommend:**" in text
    assert "sales" in text.lower() or "rental" in text.lower() or "sqft" in text.lower()


def test_open_recommend_uses_positive_market_language():
    projects = [
        {"name": "Budget Fit", "price": 1_000_000, "bedroom_min": 1, "bedroom_max": 2, "size_sqft": 750, "status": "Ready", "sales_count": 12, "rent_count": 15, "handover_time": "2026-06-01"},
        {"name": "Premium", "price": 3_000_000, "bedroom_min": 3, "bedroom_max": 4, "size_sqft": 1600, "status": "On Sale", "sales_count": 4, "rent_count": 2, "handover_time": "2029-06-01"},
    ]
    result = rank_projects(projects, wanted_beds=2, attach_sales=False)
    text = format_shortlist(
        result,
        question="Where should I invest 1.5M?",
        assumptions="Ranked on price, size, sales/transactions and rental activity.",
    )
    assert "price, size, sales/transactions and rental activity" in text.lower()
    assert "**Acot Recommend:**" in text
    assert "no yield" not in text.lower()
    assert "roi" not in text.lower()


def test_budget_fit_beats_a_much_cheaper_project():
    projects = [
        {"name": "Cheap Tower", "price": 430_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 40, "status": "Ready"},
        {"name": "Near Budget", "price": 1_720_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 14, "status": "Ready"},
        {"name": "Over Budget", "price": 2_600_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 20, "status": "On Sale"},
    ]
    result = rank_projects(projects, wanted_beds=3, budget_aed=1_800_000, attach_sales=False)
    assert result["winner"]["name"] == "Near Budget"
    text = format_footer(result)
    assert "1,720,000" in text
    assert "430,000" not in text


def test_ready_stock_quotes_latest_sale_not_old_handover():
    from app.answers.recommend import format_ranked_picks

    projects = [
        {
            "name": "Manchester Tower",
            "price": 1_200_000,
            "bedroom_min": 1,
            "bedroom_max": 1,
            "sales_count": 20,
            "community": "Dubai Marina",
            "price_kind": "recent_sale",
            "handover_time": "2007-10-01",
            "last_sold_on": "2026-10-15",
        },
        {
            "name": "Cascades",
            "price": 1_100_000,
            "bedroom_min": 1,
            "bedroom_max": 1,
            "sales_count": 5,
            "community": "Dubai Marina",
            "price_kind": "recent_sale",
            "handover_time": "2006-02-01",
            "last_sold_on": "2026-02-05",
        },
    ]
    text = format_ranked_picks(
        rank_projects(projects, wanted_beds=1, budget_aed=1_200_000, attach_sales=False),
        intro="For a 1-bedroom in Dubai Marina at up to AED 1,200,000,",
    )
    assert "2007" not in text
    assert "2006" not in text
    assert "latest recorded sale Oct 2026" in text
    assert "sales in the last 12 months" in text


def test_old_completion_does_not_beat_a_more_recent_sale():
    projects = [
        {
            "name": "Vintage Tower",
            "price": 1_200_000,
            "bedroom_min": 1,
            "bedroom_max": 1,
            "sales_count": 10,
            "handover_time": "2006-01-01",
            "last_sold_on": "2026-01-01",
        },
        {
            "name": "Newer Ready",
            "price": 1_200_000,
            "bedroom_min": 1,
            "bedroom_max": 1,
            "sales_count": 10,
            "handover_time": "2024-01-01",
            "last_sold_on": "2026-10-01",
        },
    ]
    result = rank_projects(projects, wanted_beds=1, budget_aed=1_200_000, attach_sales=False)
    assert result["winner"]["name"] == "Newer Ready"


def test_offplan_still_shows_future_handover():
    from app.answers.recommend import format_ranked_picks

    projects = [
        {
            "name": "Future A",
            "price": 1_200_000,
            "sales_count": 8,
            "price_kind": "recent_sale",
            "handover_time": "2028-06-01",
            "last_sold_on": "2026-09-01",
        },
        {
            "name": "Future B",
            "price": 1_150_000,
            "sales_count": 4,
            "price_kind": "recent_sale",
            "handover_time": "2029-06-01",
            "last_sold_on": "2026-08-01",
        },
    ]
    text = format_ranked_picks(
        rank_projects(projects, budget_aed=1_200_000, attach_sales=False),
        intro="For these options,",
    )
    assert "handover Jun 2028" in text
    assert "latest recorded sale Sep 2026" in text


def test_compare_table_mentions_acot_recommend():
    projects = [
        {"name": "Bayshore", "price": 1_600_000, "size_sqft": 694, "sales_count": 53, "rent_count": 249, "handover_time": "2022-10-01"},
        {"name": "Orchid", "price": 1_850_000, "size_sqft": 811, "sales_count": 80, "rent_count": 151, "handover_time": "2025-09-01"},
    ]
    result = rank_projects(projects, attach_sales=False)
    text = format_compare_table(projects, result)
    assert "| Factor |" in text
    assert "| Recent sales |" in text
    assert "| Rental contracts |" in text
    assert "| Score |" in text
    assert "**Acot Recommend:**" in text
    assert "Why this is the strongest match" in text or "**Why:**" in text


def test_budget_floor_is_sixty_percent():
    scores = _budget_fit_scores([430_000, 1_080_000, 1_800_000, 2_400_000], 1_800_000)
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1
    assert scores[3] < scores[2]


def test_jvc_1_8m_pool_does_not_lead_with_430k():
    projects = [
        {"name": "Hanover Square", "price": 430_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 80, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Cheap Studio Block", "price": 520_000, "bedroom_min": 0, "bedroom_max": 1, "sales_count": 40, "community": "JVC"},
        {"name": "Mid Tower", "price": 900_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 30, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Binghatti Emerald", "price": 1_800_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 22, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Aveline", "price": 1_786_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 18, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Allura", "price": 1_781_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 12, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Over Budget Bay", "price": 2_600_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 20, "community": "JVC"},
        {"name": "Park Gate", "price": 1_650_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 9, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Diamond Views 1", "price": 1_720_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 14, "community": "JVC", "price_kind": "recent_sale"},
        {"name": "Elz", "price": 1_740_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 7, "community": "JVC", "price_kind": "recent_sale"},
    ]
    result = rank_projects(projects, wanted_beds=3, budget_aed=1_800_000, attach_sales=False)
    top = [item["name"] for item in result["ranked"][:3]]
    assert result["winner"]["name"] != "Hanover Square"
    assert "Hanover Square" not in top
    assert result["ranked"][0]["price"] >= 1_800_000 * 0.6
    text = format_ranked_picks(result, intro="For a 3-bedroom in Jumeirah Village Circle at up to AED 1,800,000,")
    assert "430,000" not in text.split("Share a bedroom")[0]


def test_wanted_beds_inside_min_max_counts_as_a_fit():
    projects = [
        {"name": "Wide Range", "price": 1_800_000, "bedroom_min": 1, "bedroom_max": 4, "sales_count": 10},
        {"name": "Wrong Beds", "price": 1_800_000, "bedroom_min": 0, "bedroom_max": 1, "sales_count": 10},
    ]
    result = rank_projects(projects, wanted_beds=3, budget_aed=1_800_000, attach_sales=False)
    assert result["winner"]["name"] == "Wide Range"


def test_single_factor_says_limited_data_and_omits_score():
    projects = [
        {"name": "A", "price": 1_000_000},
        {"name": "B", "price": 2_000_000},
    ]
    # Exercise the single-factor ranking path; production recommend uses require_evidence=True.
    result = rank_projects(projects, attach_sales=False, require_evidence=False)
    assert result["can_rank"] is False
    assert result["used_factors"] == ["lower price"]
    footer = format_footer(result)
    assert "limited data" in footer.lower()
    assert "score" not in footer.lower()
    table = format_compare_table(projects, result)
    assert "| Score |" in table


def test_afford_turn_caps_at_budget_and_lists_nearby(monkeypatch):
    from app.answers import intent_turns

    over = [{"name": "Marina Over", "price": 1_400_000, "bedroom_min": 1, "bedroom_max": 1}]
    monkeypatch.setattr(intent_turns, "fetch_recommend_projects", lambda *args, **kwargs: ([], over))
    monkeypatch.setattr(intent_turns, "fetch_bedroom_median", lambda *args, **kwargs: {"median": 1_400_000})
    monkeypatch.setattr(intent_turns, "fetch_areas_near_budget", lambda *args, **kwargs: [{"area": "JUMEIRAH VILLAGE CIRCLE"}])
    monkeypatch.setattr(intent_turns, "_search", lambda *args, **kwargs: ([], []))
    turn = intent_turns.recommend_data_turn("Is AED 900,000 enough for a 1-bedroom in Dubai Marina?")
    names = [row["name"] for row in turn["structured_summary"]["projects"]]
    assert "Marina Over" not in names
    assert "Jumeirah Village Circle" in turn["fast_answer"] or "JUMEIRAH" in turn["fast_answer"]
    assert "couldn’t find recorded" in turn["fast_answer"].lower() or "couldn't find recorded" in turn["fast_answer"].lower()


def test_afford_turn_keeps_under_budget_sales_and_drops_weak_old_handover(monkeypatch):
    from app.answers import intent_turns

    matches = [
        {
            "name": "Damac Voleo",
            "price": 1_400_000,
            "sales_count": 1,
            "price_kind": "recent_sale",
            "handover_time": "2014-06-05",
            "last_sold_on": "2026-10-08",
        },
        {
            "name": "Aykon City Tower C",
            "price": 1_480_000,
            "sales_count": 42,
            "price_kind": "recent_sale",
            "handover_time": "2023-10-12",
            "last_sold_on": "2026-11-03",
        },
        {
            "name": "Merano Tower",
            "price": 1_390_000,
            "sales_count": 12,
            "price_kind": "recent_sale",
            "handover_time": "2020-02-27",
            "last_sold_on": "2026-10-15",
        },
        {
            "name": "Over Budget Bay",
            "price": 2_200_000,
            "sales_count": 20,
            "price_kind": "recent_sale",
        },
    ]
    monkeypatch.setattr(intent_turns, "fetch_recommend_projects", lambda *a, **k: (matches, []))
    monkeypatch.setattr(intent_turns, "fetch_bedroom_median", lambda *a, **k: {"median": 3_093_457})
    monkeypatch.setattr(
        intent_turns,
        "fetch_areas_near_budget",
        lambda *a, **k: [{"area": "AL SATWA"}],
    )
    turn = intent_turns.recommend_data_turn(
        "Is AED 1.5M enough for a 2-bedroom in Business Bay?"
    )
    names = [row["name"] for row in turn["structured_summary"]["projects"]]
    answer = turn["fast_answer"].lower()
    assert "Over Budget Bay" not in names
    assert "Damac Voleo" not in names
    assert "Aykon City Tower C" in names
    assert "Merano Tower" in names
    assert "still exist there" in answer
    assert "nearby areas can be a better fit" not in answer
    assert "closest options we have" not in answer
    for row in turn["structured_summary"]["projects"]:
        assert row.get("handover_time"), row.get("name")
        assert float(row["price"]) <= 1_500_000
    # Compare must still see completion dates (cards hide past handover in the UI).
    from app.answers.recommend import format_compare_table, rank_projects

    ranked = rank_projects(
        turn["structured_summary"]["projects"],
        attach_sales=False,
    )
    table = format_compare_table(turn["structured_summary"]["projects"], ranked)
    assert "| Handover |" in table
    assert table.count("| — |") == 0 or "Oct 2023" in table or "2023" in table or "2020" in table


def test_filtered_turn_ranks_a_wide_pool_then_shows_three(monkeypatch):
    from app.answers import intent_turns
    from app.answers.recommend import RECOMMEND_POOL

    pool = []
    pool.append({"name": "Hanover Square", "price": 430_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 80, "price_kind": "recent_sale"})
    for index in range(RECOMMEND_POOL - 1):
        price = 1_200_000 + index * 15_000
        pool.append({
            "name": f"JVC Tower {index}",
            "price": min(price, 1_800_000),
            "bedroom_min": 3,
            "bedroom_max": 3,
            "sales_count": 8 + index,
            "price_kind": "recent_sale",
        })
    monkeypatch.setattr(intent_turns, "fetch_recommend_projects", lambda *args, **kwargs: (pool, []))
    turn = intent_turns.recommend_data_turn(
        "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    )
    names = [row["name"] for row in turn["structured_summary"]["projects"]]
    assert len(names) == 3
    assert names[0] != "Hanover Square"
    assert "Hanover Square" not in names
    assert "430,000" not in turn["fast_answer"]
    answer = turn["fast_answer"]
    assert "family of 4" in answer.lower()
    assert "3-bedroom" in answer
    assert "1,800,000" in answer
    assert "Jumeirah Village Circle" in answer
    assert "not far cheaper" in answer.lower()
    assert "parents" in answer.lower()
    for name in names:
        assert name in answer


_FILTERED_ASKS = [
    "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M.",
    "I want a 2-bedroom in Business Bay for AED 1.5M to buy.",
    "I want a 1-bedroom in Dubai Marina around AED 1.2M.",
    "Looking for a 4-bedroom in Arabian Ranches around AED 4M.",
    "I need a 3-bedroom in Dubai Hills Estate around AED 2.5M.",
    "I want a studio in JLT under AED 800,000.",
]


def test_filtered_asks_keep_explanation_and_never_sell_over_budget_as_match(monkeypatch):
    """Class rule for place + beds + budget: explain the ask; data must fit the ask."""
    from app.answers import intent_turns
    from app.answers.intent_route import classify_intent, parse_budget_aed

    for question in _FILTERED_ASKS:
        route = classify_intent(question)
        assert route["kind"] == "filtered", question
        budget = int(route["budget_aed"] or parse_budget_aed(question) or 0)
        assert budget > 0, question
        place = route["subject"]
        assert place, question

        over = [
            {
                "name": f"{place} Far One",
                "price": int(budget * 1.5),
                "bedroom_min": route.get("bedrooms") or 0,
                "bedroom_max": route.get("bedrooms") or 0,
                "sales_count": 12,
                "community": place,
                "price_kind": "recent_sale",
            },
            {
                "name": f"{place} Far Two",
                "price": int(budget * 1.8),
                "bedroom_min": route.get("bedrooms") or 0,
                "bedroom_max": route.get("bedrooms") or 0,
                "sales_count": 9,
                "community": place,
                "price_kind": "recent_sale",
            },
            {
                "name": f"{place} Far Three",
                "price": int(budget * 2.2),
                "bedroom_min": route.get("bedrooms") or 0,
                "bedroom_max": route.get("bedrooms") or 0,
                "sales_count": 4,
                "community": place,
                "price_kind": "recent_sale",
            },
            {
                "name": f"{place} Far Dump",
                "price": int(budget * 3.1),
                "bedroom_min": route.get("bedrooms") or 0,
                "bedroom_max": route.get("bedrooms") or 0,
                "sales_count": 2,
                "community": place,
                "price_kind": "recent_sale",
            },
        ]
        monkeypatch.setattr(
            intent_turns, "fetch_recommend_projects", lambda *a, **k: ([], over)
        )
        turn = intent_turns.recommend_data_turn(question)
        answer = turn["fast_answer"]
        cards = turn["structured_summary"]["projects"]
        assert place in answer, question
        assert f"{budget:,}" in answer, question
        assert "looked for recorded" in answer.lower() or "you asked" in answer.lower(), question
        assert "the options below are recorded" not in answer.lower(), question
        assert "above" in answer.lower(), question
        assert len(cards) <= 3, question
        assert f"{place} Far Dump" not in [row["name"] for row in cards], question
        for row in cards:
            assert float(row["price"]) > budget, question
            assert row["name"] in answer, question

        memory = type("M", (), {
            "last_user_question": question,
            "last_standalone_question": question,
            "turns": [{
                "user_question": question,
                "standalone_question": question,
                "candidate_names": [row["name"] for row in cards],
            }],
            "active_candidates": [
                {"name": row["name"], "entity_type": "project"} for row in cards
            ],
        })()
        more = intent_turns.recommend_data_turn("show me more", memory)
        assert more["structured_summary"]["projects"] == [], question
        assert "already showed" in more["fast_answer"].lower() or "requested budget" in more["fast_answer"].lower(), question


def test_filtered_in_band_sales_keep_near_budget_explanation(monkeypatch):
    from app.answers import intent_turns
    from app.answers.intent_route import classify_intent

    for question in _FILTERED_ASKS:
        route = classify_intent(question)
        budget = int(route["budget_aed"])
        place = route["subject"]
        beds = route.get("bedrooms")
        matches = [
            {
                "name": f"{place} Fit {index}",
                "price": int(budget * ratio),
                "bedroom_min": beds or 0,
                "bedroom_max": beds or 0,
                "sales_count": 10 + index,
                "community": place,
                "price_kind": "recent_sale",
            }
            for index, ratio in enumerate((0.72, 0.85, 0.94, 0.98), start=1)
        ]
        monkeypatch.setattr(
            intent_turns, "fetch_recommend_projects", lambda *a, **k: (matches, [])
        )
        turn = intent_turns.recommend_data_turn(question)
        answer = turn["fast_answer"]
        cards = turn["structured_summary"]["projects"]
        assert "sit near" in answer.lower() and "not far cheaper" in answer.lower(), question
        assert f"{budget:,}" in answer, question
        assert place in answer, question
        assert len(cards) == 3, question
        for row in cards:
            assert float(row["price"]) <= budget, question
            assert row["name"] in answer, question


def test_hyphen_bedroom_is_not_rewritten_to_bedroom_only():
    from app.retrieval.hybrid.question_text import repair_typed_question
    from app.answers.intent_route import classify_intent, parse_bedrooms
    from app.answers.intent_turns import _explain_filtered_request

    raw = "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    repaired = repair_typed_question(raw)
    assert "3-bedroom" in repaired.lower()
    assert parse_bedrooms(repaired) == [3]
    route = classify_intent(repaired)
    text = _explain_filtered_request(route, route.get("subject"), repaired)
    assert "family of 4" in text.lower()
    assert "3-bedroom" in text
    assert "1,800,000" in text


def test_filtered_explains_beds_place_and_budget_without_family():
    from app.answers.intent_route import classify_intent
    from app.answers.intent_turns import _explain_filtered_request

    for question in _FILTERED_ASKS:
        route = classify_intent(question)
        text = _explain_filtered_request(route, route.get("subject"), question)
        assert text, question
        assert route["subject"] in text, question
        assert f"{int(route['budget_aed']):,}" in text, question
        assert "sit near" in text.lower(), question

    route = classify_intent("I want a 2-bedroom in Business Bay for AED 1.5M to buy.")
    text = _explain_filtered_request(route, route.get("subject"), "I want a 2-bedroom in Business Bay for AED 1.5M to buy.")
    assert "2-bedroom" in text
    assert "Business Bay" in text
    assert "1,500,000" in text
    assert "not far cheaper" in text.lower()

    studio = classify_intent("I want a studio in JLT under AED 800,000.")
    studio_text = _explain_filtered_request(
        studio, studio.get("subject"), "I want a studio in JLT under AED 800,000."
    )
    assert "studio" in studio_text.lower()
    assert "800,000" in studio_text


def test_afford_and_split_open_with_the_ask():
    from app.answers.recommend import format_afford_answer, format_split_answer

    afford = format_afford_answer(
        "Dubai Marina",
        900_000,
        {"median": 1_400_000},
        [{"name": "Close", "price": 850_000, "sales_count": 8}],
        [{"area": "JUMEIRAH VILLAGE CIRCLE"}],
        bedrooms=1,
    )
    assert afford.startswith("Understood:")
    assert "you're asking whether aed 900,000 is enough" in afford.lower()
    assert "1-bedroom" in afford
    assert "Dubai Marina" in afford
    assert "still exist there" in afford.lower()
    assert "nearby areas can be a better fit" not in afford.lower()

    split = format_split_answer(
        [
            {
                "name": "Studio One",
                "price": 320_000,
                "price_kind": "recent_sale",
                "sales_count": 12,
                "last_sold_on": "2026-06-01",
            }
        ],
        [
            {
                "name": "Villa One",
                "price": 680_000,
                "price_kind": "recent_sale",
                "sales_count": 8,
                "rent_count": 4,
                "last_sold_on": "2026-05-01",
            }
        ],
        900_000,
    )
    assert "transaction-backed" in split.lower() or "sales + rent" in split.lower()
    assert "studio" in split.lower()
    assert "villa" in split.lower()
    assert "sales" in split.lower()


def test_transaction_evidence_rejects_handover_only():
    from app.answers.recommend import _has_transaction_evidence, _project_points

    bare = {"name": "Forest City Tower 3", "handover_time": "2032-01-01"}
    sold = {
        "name": "Ajmal Sarah",
        "price": 320_756,
        "price_kind": "recent_sale",
        "sales_count": 14,
        "rent_count": 6,
        "median_rent": 42_000,
        "last_sold_on": "2026-06-01",
    }
    assert _has_transaction_evidence(bare) is False
    assert _has_transaction_evidence(sold) is True
    points = " ".join(_project_points(sold)).lower()
    assert "sales" in points
    assert "rent" in points
    assert "handover" not in " ".join(_project_points(bare)).lower()


def test_invest_does_not_recommend_handover_only_project():
    """Thin rows (handover only) must not beat projects with recorded sales/price."""
    pool = [
        {
            "name": "One Central Dubai",
            "handover_time": "2031-09-01",
            "status": "On Sale",
        },
        {
            "name": "Grand Polo - Equestra",
            "price": 4_002_888,
            "bedroom_min": 3,
            "bedroom_max": 5,
            "size_sqft": 1601,
            "handover_time": "2031-08-01",
            "status": "On Sale",
            "sales_count": 335,
            "rent_count": 40,
            "last_sold_on": "2026-10-01",
        },
        {
            "name": "Trump Tower",
            "price": 4_081_995,
            "bedroom_min": 1,
            "bedroom_max": 3,
            "size_sqft": 883,
            "handover_time": "2031-12-01",
            "status": "On Sale",
            "sales_count": 34,
            "last_sold_on": "2026-09-01",
        },
        {
            "name": "Forest City Tower 3",
            "handover_time": "2032-01-01",
            "status": "On Sale",
        },
    ]
    result = rank_projects(pool, attach_sales=False, goal="invest")
    assert result["winner"]["name"] == "Grand Polo - Equestra"
    assert result["winner"]["name"] != "One Central Dubai"
    text = format_shortlist(result, question="which property is best to invest")
    assert "**Acot Recommend:** Grand Polo - Equestra" in text
    assert "One Central Dubai" not in text.split("**Acot Recommend:**")[-1].split("\n")[0]
    assert "335" in text or "sales" in text.lower()
    why_block = text.split("**Why")[-1]
    assert "handover" not in why_block.lower() or "sales" in why_block.lower() or "4,002,888" in why_block


def test_phase4_family_and_invest_rank_differently():
    """Same pool: family prefers beds/size/ready; invest prefers sales+rent."""
    pool = [
        {
            "name": "Family Court",
            "price": 1_900_000,
            "bedroom_min": 3,
            "bedroom_max": 4,
            "size_sqft": 1800,
            "status": "Ready",
            "sales_count": 4,
            "rent_count": 2,
        },
        {
            "name": "Investor Tower",
            "price": 1_100_000,
            "bedroom_min": 0,
            "bedroom_max": 1,
            "size_sqft": 450,
            "status": "On Sale",
            "sales_count": 80,
            "rent_count": 120,
        },
        {
            "name": "Mid Option",
            "price": 1_500_000,
            "bedroom_min": 2,
            "bedroom_max": 2,
            "size_sqft": 900,
            "status": "Ready",
            "sales_count": 20,
            "rent_count": 15,
        },
    ]
    family = rank_projects(
        pool, wanted_beds=3, budget_aed=2_000_000, attach_sales=False, goal="family"
    )
    invest = rank_projects(
        pool, wanted_beds=3, budget_aed=2_000_000, attach_sales=False, goal="invest"
    )
    assert family["goal"] == "family"
    assert invest["goal"] == "invest"
    assert family["winner"]["name"] == "Family Court"
    assert invest["winner"]["name"] == "Investor Tower"
    assert family["winner"]["name"] != invest["winner"]["name"]
    # Why should surface goal-primary factors first.
    assert family["why"][0] in {
        "bedroom fit", "more bedrooms", "more space", "readiness", "budget fit",
    }
    assert invest["why"][0] in {
        "more recent sales activity", "active rental market", "latest recorded sale",
    }


def test_phase4_budget_goal_prefers_budget_fit_over_volume():
    pool = [
        {
            "name": "Near Budget",
            "price": 1_750_000,
            "bedroom_min": 3,
            "bedroom_max": 3,
            "size_sqft": 1200,
            "status": "Ready",
            "sales_count": 5,
            "rent_count": 3,
        },
        {
            "name": "Busy Cheap",
            "price": 700_000,
            "bedroom_min": 3,
            "bedroom_max": 3,
            "size_sqft": 1100,
            "status": "Ready",
            "sales_count": 90,
            "rent_count": 70,
        },
    ]
    budget = rank_projects(
        pool, wanted_beds=3, budget_aed=1_800_000, attach_sales=False, goal="budget"
    )
    assert budget["winner"]["name"] == "Near Budget"
    assert "budget fit" in (budget["why"] or budget["used_factors"])


def test_phase4_filtered_turn_passes_family_goal(monkeypatch):
    from app.answers import intent_turns

    rows = [
        {
            "name": "Family Court",
            "price": 1_750_000,
            "bedroom_min": 3,
            "bedroom_max": 4,
            "size_sqft": 1800,
            "status": "Ready",
            "sales_count": 4,
            "rent_count": 2,
            "community": "Jumeirah Village Circle",
        },
        {
            "name": "Investor Tower",
            "price": 1_100_000,
            "bedroom_min": 0,
            "bedroom_max": 1,
            "size_sqft": 450,
            "status": "On Sale",
            "sales_count": 80,
            "rent_count": 120,
            "community": "Jumeirah Village Circle",
        },
    ]
    monkeypatch.setattr(
        intent_turns, "_recommend_pool", lambda filters, skip_names=None: (rows, [])
    )
    turn = intent_turns.recommend_data_turn(
        "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    )
    assert turn is not None
    assert "**Acot Recommend:** Family Court" in turn["fast_answer"]
    assert "Investor Tower" in turn["fast_answer"] or "Also strong" in turn["fast_answer"]
