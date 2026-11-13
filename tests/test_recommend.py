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
    assert "can't" not in footer.lower()
    assert "no clear" not in footer.lower()
    assert "missing" not in footer.lower()
    assert "yield" not in footer.lower()


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
    result = rank_projects(projects, attach_sales=False)
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


def test_filtered_turn_ranks_a_wide_pool_then_shows_six(monkeypatch):
    from app.answers import intent_turns
    from app.answers.recommend import RECOMMEND_POOL, RECOMMEND_SHOW

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
    assert len(names) == RECOMMEND_SHOW
    assert names[0] != "Hanover Square"
    assert "Hanover Square" not in names
    assert "430,000" not in turn["fast_answer"]
