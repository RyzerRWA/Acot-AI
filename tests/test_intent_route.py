from app.answers.intent_route import classify_intent, is_show_more, parse_bedrooms, parse_budget_aed
from app.answers.predict import format_snapshot_answer, format_trend_answer
from app.answers.recommend import (
    format_afford_answer,
    format_family_areas,
    format_open_areas,
    format_open_best,
    format_ranked_picks,
    format_ready_offplan,
    format_split_answer,
    format_yield_areas,
    rank_family_areas,
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


def test_classify_arabian_ranches_typical_beds_is_snapshot():
    route = classify_intent(
        "what about Arabian Ranches only — typical 3-bed and 4-bed sale?"
    )
    assert route["intent"] == "predict"
    assert route["kind"] == "snapshot"
    assert route["subject"] == "Arabian Ranches"
    assert route["bedrooms"] == [3, 4]
    assert parse_bedrooms(
        "what about Arabian Ranches only — typical 3-bed and 4-bed sale?"
    ) == [3, 4]


def test_classify_show_properties_in_arabian_ranches_is_browse():
    route = classify_intent("Show me properties in Arabian Ranches")
    assert route["intent"] == "neither"
    assert route["subject"] == "Arabian Ranches"


def test_snapshot_answer_quotes_separate_bedroom_medians():
    text = format_snapshot_answer(
        "Arabian Ranches",
        {"median_sale": 1_051_650},
        [{"name": "Alma"}],
        [
            {"bedrooms": 3, "median": 2_400_000, "sales": 40, "as_of": "2026-10-01"},
            {"bedrooms": 4, "median": 3_100_000, "sales": 22, "as_of": "2026-10-01"},
        ],
    )
    _no_ids(text)
    assert "2,400,000" in text
    assert "3,100,000" in text
    assert "1,051,650" not in text
    assert "couldn't find" not in text.lower()
    assert "yield" not in text.lower()
    assert "cards below" in text.lower()


def test_predict_turn_uses_bedroom_medians(monkeypatch):
    monkeypatch.setattr(intent_turns, "fetch_place_trend", lambda place: {"median_sale": 1_051_650})
    monkeypatch.setattr(
        intent_turns,
        "fetch_bedroom_median",
        lambda place, beds: {
            3: {"bedrooms": 3, "median": 2_400_000, "sales": 40, "as_of": "2026-10-01"},
            4: {"bedrooms": 4, "median": 3_100_000, "sales": 22, "as_of": "2026-10-01"},
        }.get(beds),
    )
    monkeypatch.setattr(
        intent_turns,
        "_search",
        lambda filters, limit=6, skip_names=None: (
            [{"id": "P1", "name": "Alma", "community": "Arabian Ranches"}],
            [],
        ),
    )
    turn = intent_turns.predict_turn(
        "what about Arabian Ranches only — typical 3-bed and 4-bed sale?"
    )
    assert turn["use_fast"] is True
    _no_ids(turn["fast_answer"])
    assert "2,400,000" in turn["fast_answer"]
    assert "3,100,000" in turn["fast_answer"]
    assert "1,051,650" not in turn["fast_answer"]
    assert "couldn't find" not in turn["fast_answer"].lower()
    assert turn["structured_summary"]["projects"][0]["name"] == "Alma"
    assert turn["question_type"] == "search"


def test_classify_family_areas():
    route = classify_intent("Which Dubai areas are best for families?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "family_areas"
    assert route["focus"]["avoid_towers"] is False


def test_classify_communities_suit_families():
    route = classify_intent("Which communities suit families?")
    assert route["kind"] == "family_areas"


def test_classify_family_of_four_in_jvc_is_not_city_family_areas():
    route = classify_intent("I want a 3-bedroom in JVC for a family of 4 to buy.")
    assert route["kind"] != "family_areas"


def test_family_questions_read_different_intents():
    general = classify_intent("Which Dubai areas are best for families?")
    towers = classify_intent(
        "Best communities for a family with kids, not just the busiest towers."
    )
    villas = classify_intent(
        "Where should a family live in Dubai if they want more 3-bedroom and villa stock?"
    )
    assert general["focus"] != towers["focus"]
    assert towers["focus"]["kids"] is True
    assert towers["focus"]["avoid_towers"] is True
    assert villas["focus"]["want_3bed"] is True
    assert villas["focus"]["want_villas"] is True
    general_text = format_family_areas(_family_sample(), general["focus"])
    towers_text = format_family_areas(_family_sample(), towers["focus"])
    assert general_text.split("\n")[0] != towers_text.split("\n")[0]
    assert "does not tag" not in general_text.lower()
    assert "towers" in towers_text.lower()


def _family_sample():
    return [
        {
            "area": "Mudon",
            "existing_villas": 133,
            "existing_villa_share": 0.88,
            "existing_family_sales": 151,
            "median_family_sale": 2_500_000,
        }
    ]


def test_family_areas_format_uses_sales_not_lifestyle():
    text = format_family_areas(
        [
            {
                "area": "Arabian Ranches",
                "existing_villas": 96,
                "existing_villa_share": 1.0,
                "existing_family_sales": 42,
                "median_family_sale": 3_400_000,
            },
            {
                "area": "Dubai Hills Estate",
                "existing_villas": 62,
                "existing_villa_share": 0.18,
                "existing_family_sales": 123,
                "median_family_sale": 2_900_000,
            },
        ]
    )
    _no_ids(text)
    assert "couldn't find" not in text.lower()
    assert "does not tag" not in text.lower()
    assert "Arabian Ranches" in text
    assert "96" in text
    assert "3,400,000" in text
    assert "park" not in text.lower()
    assert "school" not in text.lower()
    assert "cards below" in text.lower()


def test_family_rank_skips_busy_towers_when_asked():
    rows = [
        {
            "area": "Jumeirah Village Circle",
            "sales_12m": 10545,
            "existing_sales": 4187,
            "villa_sales": 76,
            "existing_villas": 73,
            "family_sales": 242,
            "existing_family_sales": 177,
        },
        {
            "area": "Mudon",
            "sales_12m": 206,
            "existing_sales": 151,
            "villa_sales": 188,
            "existing_villas": 133,
            "family_sales": 181,
            "existing_family_sales": 151,
        },
        {
            "area": "Damac Islands",
            "sales_12m": 4059,
            "existing_sales": 10,
            "villa_sales": 1706,
            "existing_villas": 5,
            "family_sales": 4045,
            "existing_family_sales": 8,
        },
        {
            "area": "The Springs",
            "sales_12m": 114,
            "existing_sales": 114,
            "villa_sales": 114,
            "existing_villas": 114,
            "family_sales": 51,
            "existing_family_sales": 51,
        },
    ]
    general = [row["area"] for row in rank_family_areas(rows, focus={}, limit=5)]
    towers = [
        row["area"]
        for row in rank_family_areas(
            rows,
            focus={"avoid_towers": True, "kids": True, "want_villas": True},
            limit=5,
        )
    ]
    stock = [
        row["area"]
        for row in rank_family_areas(
            rows,
            focus={"want_villas": True, "want_3bed": True},
            limit=5,
        )
    ]
    assert "Mudon" in general
    assert "Damac Islands" not in general
    assert "Jumeirah Village Circle" not in towers
    assert towers[0] in {"The Springs", "Mudon"}
    assert stock[0] == "Damac Islands"


def test_classify_open_best():
    route = classify_intent("Tell me which one is the best property in Dubai.")
    assert route["intent"] == "recommend"
    assert route["kind"] == "open_best"


def test_classify_where_should_i_buy_is_open_best():
    route = classify_intent("Where should I buy in Dubai?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "open_best"


def test_classify_open_buy_styles_stay_on_sql_recommend():
    for question in (
        "Where should I buy in Dubai if I have no area yet and no budget?",
        "Where should I live in Dubai?",
        "Best areas to buy in Dubai",
        "Which area has the most sales?",
        "Is Dubai a good investment based on the available data?",
        "Where are people buying?",
    ):
        route = classify_intent(question)
        assert route["intent"] == "recommend", question
        assert route["kind"] == "open_best", question


def test_classify_best_properties_in_a_place_stays_filtered():
    route = classify_intent("show me the best properties in Jumeirah Village Circle?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "filtered"
    assert route["subject"] == "Jumeirah Village Circle"


def test_classify_where_should_i_buy_stays_open_best_with_cards_on_screen():
    class _Memory:
        active_candidates = [
            {"name": "A", "entity_type": "project"},
            {"name": "B", "entity_type": "project"},
        ]

    route = classify_intent(
        "Where should I buy in Dubai if I have no area yet and no budget?",
        _Memory(),
    )
    assert route["intent"] == "recommend"
    assert route["kind"] == "open_best"


def test_classify_ready_or_offplan_is_not_open_best():
    route = classify_intent("Should I buy ready or off-plan in Dubai?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "ready_offplan"
    assert route["subject"] is None


def test_apartment_for_two_members_is_a_one_bedroom_recommend():
    route = classify_intent("Which apartment is suitable for 2 members")
    assert route["intent"] == "recommend"
    assert route["kind"] == "filtered"
    assert route["bedrooms"] == 1
    assert route["household"] == 2
    assert route["subject"] is None
    persons = classify_intent("Show me the best apartments for 2 persons")
    assert persons["intent"] == "recommend"
    assert persons["kind"] == "filtered"
    assert persons["bedrooms"] == 1
    assert persons["household"] == 2


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
    text = format_open_best(ranked, question="Tell me which one is the best property in Dubai.")
    _no_ids(text)
    assert "**Alpha**" in text or "Alpha" in text
    assert "budget" in text.lower()
    assert "strong starting point" not in text.lower()
    assert "look strong" not in text.lower()
    assert "there isn’t a single best property" in text.lower() or "there isn't a single best property" in text.lower()


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
        ],
        question="Where should I buy in Dubai if I have no area yet and no budget?",
    )
    _no_ids(text)
    assert "**Business Bay**" in text
    assert "1,900,000" in text
    assert "live in the home" in text.lower() or "rent it out" in text.lower()
    assert "strong starting point" not in text.lower()
    assert "look strong" not in text.lower()
    assert "no area or budget" in text.lower()


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
        lambda filters, limit=6, skip_names=None: ([{"id": "P1", "name": "Marina One", "price": 1_800_000}], []),
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
    monkeypatch.setattr(intent_turns, "_search", lambda filters, limit=6, skip_names=None: (rows, []))
    turn = intent_turns.recommend_data_turn("Tell me which one is the best property in Dubai.")
    assert turn is not None
    _no_ids(turn["fast_answer"])
    assert "Business Bay" in turn["fast_answer"]
    assert "cards below" in turn["fast_answer"].lower()
    assert "strong starting point" not in turn["fast_answer"].lower()
    assert "look strong" not in turn["fast_answer"].lower()
    assert turn["structured_summary"]["projects"][0]["name"] == "Bay Tower"


class _Memory:
    def __init__(self, question, names=None):
        self.last_user_question = question
        self.last_standalone_question = question
        self.turns = [{
            "user_question": question,
            "standalone_question": question,
            "candidate_names": names or [],
        }]
        self.active_candidates = [
            {"name": name, "entity_type": "project"}
            for name in (names or [])
        ]


def test_show_more_is_detected():
    assert is_show_more("show me more")
    assert is_show_more("Show more")
    assert is_show_more("more options")
    assert is_show_more("more please")
    assert not is_show_more("show me more in Dubai Marina")
    assert not is_show_more("I want a 3-bedroom in JVC")


def test_show_more_keeps_jvc_filters():
    prior = "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    route = classify_intent("show me more", _Memory(prior, ["Hanover Square"]))
    assert route["intent"] == "recommend"
    assert route["kind"] == "filtered"
    assert route["more"] is True
    assert route["bedrooms"] == 3
    assert route["budget_aed"] == 1_800_000
    assert "Jumeirah Village Circle" in (route["subject"] or "")
    assert route["source_question"] == prior


def test_show_more_without_history_is_not_a_new_search():
    route = classify_intent("show me more")
    assert route["intent"] == "neither"
    assert route["kind"] == "show_more"


def test_second_show_more_still_uses_first_question():
    prior = "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    memory = _Memory("show me more", ["Hanover Square"])
    memory.last_standalone_question = prior
    memory.turns = [
        {"user_question": prior, "standalone_question": prior, "candidate_names": ["Hanover Square"]},
        {"user_question": "show me more", "standalone_question": prior},
    ]
    route = classify_intent("show me more", memory)
    assert route["kind"] == "filtered"
    assert route["more"] is True
    assert route["source_question"] == prior


def test_show_more_after_family_areas_stays_family_areas():
    prior = "Which Dubai areas are best for families?"
    route = classify_intent("show me more", _Memory(prior))
    assert route["kind"] == "family_areas"
    assert route["more"] is True


def test_show_more_skips_projects_already_on_screen(monkeypatch):
    prior = "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    memory = _Memory(prior, ["Hanover Square", "Diamond Views 1"])

    def fake_pool(filters, skip_names=None, limit=6):
        skip = {str(name).lower() for name in (skip_names or set())}
        assert filters.get("communities")
        assert filters.get("bedrooms") == 3
        assert filters.get("max_price") == 1_800_000
        rows = [
            {"id": "P1", "name": "Hanover Square", "price": 430_000, "bedroom_min": 3, "bedroom_max": 3},
            {"id": "P2", "name": "Diamond Views 1", "price": 555_000, "bedroom_min": 3, "bedroom_max": 3},
            {"id": "P3", "name": "Later Match", "price": 1_650_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 8},
            {"id": "P4", "name": "Another Match", "price": 1_720_000, "bedroom_min": 3, "bedroom_max": 3, "sales_count": 14},
        ]
        kept = [row for row in rows if row["name"].lower() not in skip]
        return kept, []

    monkeypatch.setattr(intent_turns, "_recommend_pool", fake_pool)
    turn = intent_turns.recommend_data_turn("show me more", memory)
    assert turn is not None
    names = [row["name"] for row in turn["structured_summary"]["projects"]]
    assert "Hanover Square" not in names
    assert "Diamond Views 1" not in names
    assert "Later Match" in names
    assert names[0] == "Another Match"
    assert "specify" not in turn["fast_answer"].lower()
    assert "Jumeirah Village Circle" in turn["fast_answer"]
    assert turn["standalone_question"] == prior
    _no_ids(turn["fast_answer"])


def test_recommend_family_areas_turn(monkeypatch):
    areas = [
        {
            "area": "Arabian Ranches",
            "official_area": "ARABIAN RANCHES",
            "family_sales": 1240,
            "villa_townhouse_sales": 680,
            "median_family_sale": 3_400_000,
        }
    ]
    rows = [{"id": "R1", "name": "Ranches Villa", "price": 3_200_000, "bedroom_min": 3}]
    monkeypatch.setattr(
        intent_turns, "fetch_family_areas", lambda limit=5, focus=None: areas
    )
    monkeypatch.setattr(intent_turns, "_search", lambda filters, limit=6, skip_names=None: (rows, []))
    turn = intent_turns.recommend_data_turn("Which Dubai areas are best for families?")
    assert turn is not None
    _no_ids(turn["fast_answer"])
    assert "couldn't find" not in turn["fast_answer"].lower()
    assert "does not tag" not in turn["fast_answer"].lower()
    assert "Arabian Ranches" in turn["fast_answer"]
    assert "park" not in turn["fast_answer"].lower()
    assert turn["structured_summary"]["projects"][0]["name"] == "Ranches Villa"


def test_classify_rental_yield_is_yield_areas():
    route = classify_intent("Which Dubai areas have the best rental yield?")
    assert route["intent"] == "recommend"
    assert route["kind"] == "yield_areas"
    assert route["subject"] is None


def test_yield_format_quotes_rents_not_a_percentage():
    text = format_yield_areas(
        [
            {
                "area": "Dubai Marina",
                "rent_contracts": 4200,
                "median_yearly_rent_aed": 120_000,
                "median_sale_price_aed": 2_600_000,
                "sales_12m": 900,
            }
        ]
    )
    _no_ids(text)
    assert "does not have rental yield" in text.lower() or "does not have" in text.lower()
    assert "Dubai Marina" in text
    assert "120,000" in text
    assert "2,600,000" in text
    assert "would you like" not in text.lower()
    assert "%" not in text


def test_yield_areas_turn_lists_rents(monkeypatch):
    areas = [
        {
            "area": "Dubai Marina",
            "official_area": "MARSA DUBAI",
            "rent_contracts": 4200,
            "median_yearly_rent_aed": 120_000,
            "median_sale_price_aed": 2_600_000,
            "sales_12m": 900,
        }
    ]
    rows = [{"id": "M1", "name": "Marina Gate", "price": 2_400_000}]
    monkeypatch.setattr(
        "app.listings.tools.top_areas_by_rent",
        lambda limit=5: areas,
    )
    monkeypatch.setattr(intent_turns, "_search", lambda filters, limit=6, skip_names=None: (rows, []))
    turn = intent_turns.recommend_data_turn("Which Dubai areas have the best rental yield?")
    assert turn is not None
    _no_ids(turn["fast_answer"])
    assert "would you like" not in turn["fast_answer"].lower()
    assert "Dubai Marina" in turn["fast_answer"]
    assert "120,000" in turn["fast_answer"]
    assert turn["structured_summary"]["projects"][0]["name"] == "Marina Gate"


def test_family_yield_question_still_uses_family_areas():
    route = classify_intent("Show me yield and ROI for the best family areas.")
    assert route["kind"] == "family_areas"
    assert route["focus"]["no_yield"] is True
    text = format_family_areas(_family_sample(), route["focus"])
    assert "yield" in text.lower() or "roi" in text.lower()
    assert "would you like" not in text.lower()


def test_ancient_handover_does_not_crash_on_windows():
    from datetime import datetime
    from app.answers.recommend import _handover_stamp, format_ready_offplan

    assert _handover_stamp(datetime(1, 1, 1)) is None
    text = format_ready_offplan(
        {"ready": {"sales_12m": 1, "median_price": 1_000_000}, "offplan": {}},
        [{"name": "Old", "handover_time": datetime(1, 1, 1)}],
        [{"name": "New", "handover_time": datetime(2029, 12, 1)}],
    )
    assert "Old" in text
    assert "New" in text


def test_ready_offplan_format_compares_both_sides():
    text = format_ready_offplan(
        {
            "ready": {"sales_12m": 5000, "median_price": 1_200_000, "median_ppsf": 1400},
            "offplan": {"sales_12m": 8000, "median_price": 1_050_000, "median_ppsf": 1500},
        },
        [{"name": "Hanover Square", "community": "JVC", "price": 430_000, "handover_time": "2015-12-01"}],
        [{"name": "Waada Altura", "community": "Dubai South", "price": 475_000, "handover_time": "2029-12-01"}],
    )
    _no_ids(text)
    assert "Ready" in text
    assert "Off-plan" in text
    assert "Hanover Square" in text
    assert "Waada Altura" in text
    assert "look strong" not in text.lower()
    assert "World Central" not in text
    assert "would you like" not in text.lower()
    assert "budget" in text.lower()


def test_ready_offplan_turn(monkeypatch):
    stats = {
        "ready": {"sales_12m": 5000, "median_price": 1_200_000, "median_ppsf": 1400},
        "offplan": {"sales_12m": 8000, "median_price": 1_050_000, "median_ppsf": 1500},
    }
    ready = [{"name": "Desert Sun Tower", "handover_time": "2015-06-01", "price": 390_000}]
    offplan = [{"name": "Waada - Altura 1", "handover_time": "2029-12-01", "price": 475_000}]

    def fake_pool(filters, limit=8, sort="cheapest"):
        return ready if sort == "earliest_handover" else offplan

    monkeypatch.setattr(intent_turns, "fetch_ready_offplan_stats", lambda place=None: stats)
    monkeypatch.setattr(intent_turns, "_project_pool", fake_pool)
    turn = intent_turns.recommend_data_turn("Should I buy ready or off-plan in Dubai?")
    assert turn is not None
    _no_ids(turn["fast_answer"])
    assert "Ready" in turn["fast_answer"]
    assert "Off-plan" in turn["fast_answer"]
    assert "Desert Sun Tower" in turn["fast_answer"]
    assert "Waada - Altura 1" in turn["fast_answer"]
    assert "these areas currently look strong" not in turn["fast_answer"].lower()
    names = [row["name"] for row in turn["structured_summary"]["projects"]]
    assert "Desert Sun Tower" in names
    assert "Waada - Altura 1" in names
