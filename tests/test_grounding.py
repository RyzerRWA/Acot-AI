"""Phase 6: cards and answer text only cite grounded sale facts."""

from datetime import date

from app.answers.grounding import ground_project_row, ground_project_rows
from app.answers.structured_answer import (
    _build_fast_structured_answer,
    _format_project_card,
    _recent_sale_row,
    build_frontend_response,
)


def test_ground_clears_recent_sale_without_price():
    row = ground_project_row(
        {
            "name": "Thin Tower",
            "price": None,
            "price_kind": "recent_sale",
            "source": "propai",
            "handover_time": "2028-01-01",
        }
    )
    assert row["price"] is None
    assert row["price_kind"] is None
    assert row["last_sold_on"] is None

    card = _format_project_card(row)
    assert card["price"] is None
    assert card["price_kind"] is None
    assert card["price_label"] is None
    assert card["last_sold_on"] is None


def test_propai_source_alone_is_not_recent_sale():
    assert not _recent_sale_row({"source": "propai", "price": None})
    assert not _recent_sale_row(
        {"source": "propai", "price_kind": "recent_sale", "price": None}
    )
    assert _recent_sale_row(
        {
            "source": "propai",
            "price": 1_000_000,
            "price_kind": "recent_sale",
        }
    )


def test_ground_syncs_last_sold_on_with_last_sale():
    row = ground_project_row(
        {
            "name": "Sold Tower",
            "price": 1_250_000,
            "price_kind": "recent_sale",
            "source": "propai",
            "last_sale": date(2026, 10, 5),
        }
    )
    assert row["last_sold_on"] == "2026-10-05"
    card = _format_project_card(row)
    assert card["last_sold_on"] == "2026-10-05"
    assert card["price_kind"] == "recent_sale"
    assert "median recorded sale" in (card["price_label"] or "").lower()


def test_answer_omits_sale_wording_without_price():
    summary = {
        "projects": [
            {
                "name": "No Sale Yet",
                "developer_name": "Demo Dev",
                "price": None,
                "price_kind": "recent_sale",
                "source": "propai",
                "community": "Business Bay",
                "handover_time": "2029-06-01",
            }
        ]
    }
    answer = _build_fast_structured_answer("Show me No Sale Yet", summary)
    assert "Recent sale" not in answer
    assert "Starting price" not in answer
    assert "No Sale Yet" in answer

    response = build_frontend_response(
        question="Show me No Sale Yet",
        standalone_question="Show me No Sale Yet",
        answer=answer,
        question_type="search",
        structured_summary=summary,
        documents=[],
    )
    card = response["data"]["projects"][0]
    assert card["price"] is None
    assert card["price_kind"] is None
    assert card["price_label"] is None


def test_answer_and_card_share_recent_sale_price():
    summary = {
        "projects": [
            {
                "name": "Marina Tower",
                "developer_name": "Emaar",
                "price": 1_500_000,
                "price_kind": "recent_sale",
                "source": "propai",
                "community": "Dubai Marina",
                "last_sold_on": "2026-09-12",
                "bedroom_min": 1,
                "bedroom_max": 2,
                "handover_time": "2027-03-01",
            }
        ]
    }
    answer = _build_fast_structured_answer("Show me Marina Tower", summary)
    assert "Median recorded sale AED 1,500,000" in answer

    response = build_frontend_response(
        question="Show me Marina Tower",
        standalone_question="Show me Marina Tower",
        answer=answer,
        question_type="search",
        structured_summary=summary,
        documents=[],
    )
    card = response["data"]["projects"][0]
    assert card["price"] == 1_500_000
    assert card["price_kind"] == "recent_sale"
    assert card["last_sold_on"] == "2026-09-12"
    assert "median recorded sale" in (card["price_label"] or "").lower()


def test_filter_no_match_keeps_cards_empty_and_is_honest():
    evidence = [
        {
            "name": "Serenz",
            "bedroom_min": 0,
            "bedroom_max": 1,
            "price": 840_000,
            "price_kind": "recent_sale",
        }
    ]
    summary = {
        "projects": evidence,  # stale matched rows must not become cards
        "filter_no_match": True,
        "filter_evidence_projects": evidence,
    }
    answer = _build_fast_structured_answer(
        "Which of these have 3 bedroom options?",
        summary,
    )
    assert "No retrieved projects have 3-bedroom options" in answer
    assert "Serenz offers" in answer
    assert summary["projects"] == []

    response = build_frontend_response(
        question="Which of these have 3 bedroom options?",
        standalone_question="Which of these have 3 bedroom options?",
        answer=answer,
        question_type="search",
        structured_summary=summary,
        documents=[],
    )
    assert response["data"]["projects"] == []


def test_filter_no_match_without_evidence_is_honest():
    summary = {"projects": [], "filter_no_match": True}
    answer = _build_fast_structured_answer(
        "Only villas under AED 1",
        summary,
    )
    assert "None of the projects I have matched that filter" in answer


def test_ground_project_rows_skips_non_dicts():
    rows = ground_project_rows(
        [
            {"name": "A", "price": 100_000, "price_kind": "recent_sale", "source": "propai"},
            "skip",
            None,
        ]
    )
    assert len(rows) == 1
    assert rows[0]["price"] == 100_000
