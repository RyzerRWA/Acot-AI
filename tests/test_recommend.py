from app.answers.recommend import format_footer, format_shortlist, rank_projects


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


def test_compare_table_mentions_acot_recommend():
    from app.answers.recommend import format_compare_table

    projects = [
        {"name": "Bayshore", "price": 1_600_000, "size_sqft": 694, "sales_count": 53, "rent_count": 249, "handover_time": "2022-10-01"},
        {"name": "Orchid", "price": 1_850_000, "size_sqft": 811, "sales_count": 80, "rent_count": 151, "handover_time": "2025-09-01"},
    ]
    result = rank_projects(projects, attach_sales=False)
    text = format_compare_table(projects, result)
    assert "| Factor |" in text
    assert "| Recent sales |" in text
    assert "**Acot Recommend:**" in text
    assert "Why this is the strongest match" in text
