from app.answers.intent_route import classify_intent, parse_budget_aed
from app.answers.predict import format_trend_answer
from app.answers.recommend import (
    format_afford_answer,
    format_open_areas,
    format_open_best,
    format_ranked_picks,
    format_split_answer,
    rank_projects,
    tag_project,
)
from app.answers import intent_turns


def _no_ids(text: str):
    assert "[[" not in text
    assert "P315" not in text


def test_classify_marina_median_is_snapshot():
    route = classify_intent(
        "What is the median sale price and the median price per sq ft in Dubai Marina "
        "for the last 12 months? Also state the latest data date"
    )
    assert route["intent"] == "predict"
    assert route["kind"] == "snapshot"
    assert route["subject"] == "Dubai Marina"


def test_snapshot_answer_quotes_marina_not_closest_areas():
    from app.answers.predict import format_snapshot_answer

    text = format_snapshot_answer(
        "Dubai Marina",
        {
            "median_sale": 2_600_000,
            "psf": 2490,
            "as_of": "2026-11-03",
        },
        [{"name": "Marina Gate"}],
    )
    assert "2,600,000" in text
    assert "2,490" in text
    assert "3 Nov 2026" in text
    assert "couldn't find" not in text.lower()
    assert "World Central" not in text
    assert "[[" not in text


def test_classify_predict_before_recommend():
    route = classify_intent("What will the price of Dubai Marina be next year?")
    assert route["intent"] == "predict"
    assert route["subject"] == "Dubai Marina"


def test_classify_predict_business_bay_in_sentence():
    route = classify_intent("Will prices in Business Bay go up?")
    assert route["intent"] == "predict"
    assert route["subject"] == "Business Bay"


def test_classify_properties_browse_is_neither():
    route = classify_intent("Properties in Business Bay.")
    assert route["intent"] == "neither"


def test_classify_open_best():
    route = classify_intent("Tell me which one is the best property in Dubai.")
    assert route["intent"] == "recommend"
    assert route["kind"] == "open_best"


def test_classify_where_should_i_buy_is_open_best():
    route = classify_intent("Where should I buy in Dubai?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "open_best"


def test_classify_filtered_jvc_buy():
    route = classify_intent("I want a 2-bedroom in JVC for AED 1.5M to buy in 3 months.")
    assert route["intent"] == "recommend"
    assert route["kind"] == "filtered"
    assert route["subject"] == "Jumeirah Village Circle"
    assert route["bedrooms"] == 2
    assert route["budget_aed"] == 1_500_000
    assert route["months"] == 3


def test_classify_afford():
    route = classify_intent("Is AED 900,000 enough for a 1-bedroom in Dubai Marina?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "afford"
    assert route["budget_aed"] == 900_000
    assert route["bedrooms"] == 1
    assert route["subject"] == "Dubai Marina"


def test_classify_split():
    route = classify_intent(
        "Budget is now AED 900,000. I need a 4-bedroom villa with a private beach "
        "and, at the same time, only a studio apartment."
    )
    assert route["intent"] == "recommend"
    assert route["kind"] == "split"
    assert route["budget_aed"] == 900_000


def test_classify_shortlist_on_screen():
    class _Memory:
        active_candidates = [
            {"name": "A", "entity_type": "project"},
            {"name": "B", "entity_type": "project"},
        ]

    route = classify_intent("Which of these is best?", _Memory())
    assert route["intent"] == "recommend"
    assert route["kind"] == "shortlist"


def test_parse_budget():
    assert parse_budget_aed("AED 1.5M") == 1_500_000
    assert parse_budget_aed("AED 900,000") == 900_000


def test_trend_answer_leads_with_history_not_a_future_price():
    text = format_trend_answer(
        "Dubai Marina",
        {
            "median_sale": 1_800_000,
            "psf": 1950,
            "change_pct": 3.2,
            "as_of": "2026-11-01",
        },
        [{"id": "P1", "name": "Marina One"}],
    )
    _no_ids(text)
    assert "1,800,000" in text
    assert "1,950" in text
    assert "3.2%" in text or "3%" in text
    assert "cards below" in text.lower()
    assert "2,100,000" not in text


def test_trend_answer_without_sales_uses_projects():
    text = format_trend_answer("Dubai Marina", None, [{"id": "P9", "name": "Tower"}])
    _no_ids(text)
    assert "cards below" in text.lower()
    assert "2,000,000" not in text


def test_open_best_format_uses_names():
    ranked = rank_projects(
        [
            {"id": "P1", "name": "Alpha", "price": 1_000_000, "sales_count": 20, "status": "Ready"},
            {"id": "P2", "name": "Beta", "price": 2_000_000, "sales_count": 2, "status": "On Sale"},
            {"id": "P3", "name": "Gamma", "price": 1_200_000, "sales_count": 8, "status": "Ready"},
        ],
        attach_sales=False,
    )
    text = format_open_best(ranked)
    _no_ids(text)
    assert "**Alpha**" in text or "Alpha" in text
    assert "budget" in text.lower()


def test_open_areas_format():
    text = format_open_areas(
        [
            {
                "area": "Business Bay",
                "sales_12m": 1200,
                "median_sale_price_aed": 1_900_000,
                "median_price_per_sqft_aed": 1800,
            },
            {
                "area": "Dubai Marina",
                "sales_12m": 900,
                "median_sale_price_aed": 2_600_000,
                "median_price_per_sqft_aed": 2490,
            },
        ]
    )
    _no_ids(text)
    assert "**Business Bay**" in text
    assert "1,900,000" in text
    assert "live in or to rent" in text.lower()


def test_afford_below_median_offers_nearby_from_data():
    text = format_afford_answer(
        "Dubai Marina",
        900_000,
        {"median": 1_400_000},
        [{"id": "P1", "name": "Close"}],
        [{"area": "JUMEIRAH VILLAGE CIRCLE"}, {"area": "ARJAN"}],
        bedrooms=1,
    )
    _no_ids(text)
    assert "1,400,000" in text
    assert "cards below" in text.lower()
    assert "Jumeirah Village Circle" in text or "Arjan" in text
    assert "not enough" not in text.lower()


def test_split_format_shows_both_lists():
    text = format_split_answer(
        [{"id": "S1", "name": "Studio One"}],
        [{"id": "V1", "name": "Villa One"}],
        900_000,
    )
    _no_ids(text)
    assert "**Studio One**" in text
    assert "**Villa One**" in text
    assert "conflict" not in text.lower()
    assert "cannot" not in text.lower()


def test_filtered_closest_uses_names():
    ranked = rank_projects(
        [
            {
                "id": "P4",
                "name": "Later Handover",
                "price": 1_200_000,
                "bedroom_min": 2,
                "handover_time": "2027-06-01",
                "sales_count": 10,
            },
            {
                "id": "P5",
                "name": "Also Close",
                "price": 1_400_000,
                "bedroom_min": 2,
                "handover_time": "2027-09-01",
                "sales_count": 4,
            },
        ],
        wanted_beds=2,
        attach_sales=False,
    )
    text = format_ranked_picks(
        ranked,
        intro="For a 2-bedroom in Jumeirah Village Circle at up to AED 1,500,000,",
        relaxed="I can keep the 3-month window and widen the area if you want.",
    )
    _no_ids(text)
    assert "Later Handover" in text


def test_tag_project_is_name_only():
    assert tag_project({"id": "P0001", "name": "Tower"}) == "**Tower**"


def test_predict_turn_uses_mocked_trend(monkeypatch):
    monkeypatch.setattr(
        intent_turns,
        "fetch_place_trend",
        lambda place: {
            "median_sale": 1_800_000,
            "psf": 1950,
            "change_pct": 3.2,
            "as_of": "2026-11-01",
            "sales_12m": 100,
        },
    )
    monkeypatch.setattr(
        intent_turns,
        "_search",
        lambda filters, limit=6: ([{"id": "P1", "name": "Marina One", "price": 1_800_000}], []),
    )
    turn = intent_turns.predict_turn("What will the price of Dubai Marina be next year?")
    assert turn["use_fast"] is True
    _no_ids(turn["fast_answer"])
    assert "1,800,000" in turn["fast_answer"]
    assert "2,500,000" not in turn["fast_answer"]
    assert turn["structured_summary"]["projects"][0]["name"] == "Marina One"


def test_recommend_open_best_turn(monkeypatch):
    areas = [
        {
            "area": "Business Bay",
            "official_area": "BUSINESS BAY",
            "sales_12m": 1000,
            "median_sale_price_aed": 1_800_000,
            "median_price_per_sqft_aed": 1700,
        }
    ]
    rows = [{"id": "A", "name": "Bay Tower", "price": 900_000, "sales_count": 12, "status": "Ready"}]
    monkeypatch.setattr(
        "app.listings.tools.top_areas_by_sales",
        lambda limit=5: areas,
    )
    monkeypatch.setattr(intent_turns, "_search", lambda filters, limit=6: (rows, []))
    turn = intent_turns.recommend_data_turn("Tell me which one is the best property in Dubai.")
    assert turn is not None
    _no_ids(turn["fast_answer"])
    assert "Business Bay" in turn["fast_answer"]
    assert "cards below" in turn["fast_answer"].lower()
    assert turn["structured_summary"]["projects"][0]["name"] == "Bay Tower"
