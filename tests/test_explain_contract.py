"""Explain-first answer contract across recommend / browse / clarify."""
from app.answers.clarify import clarify_question_text
from app.answers.explain_contract import build_understood, prepend_understood
from app.answers.recommend import format_ranked_picks, rank_projects


def test_build_understood_browse_ready_budget():
    line = build_understood(
        place="Jumeirah Village Circle",
        beds=1,
        budget_aed=700_000,
        ready_only=True,
        intent="browse",
        under_budget=True,
    )
    assert line.startswith("Understood:")
    assert "looking for" in line.lower()
    assert "ready" in line.lower()
    assert "1-bedroom" in line
    assert "700,000" in line
    assert "Jumeirah Village Circle" in line
    assert len(line) > 60


def test_agents_and_browse_narratives_are_fact_rich():
    from app.answers.explain_contract import (
        build_agents_narrative,
        build_browse_narrative,
        build_family_motive,
        build_motive,
    )

    agents = build_agents_narrative(place="Dubai Hills Estate", count=5)
    assert "Understood:" in agents
    assert "Dubai Hills Estate" in agents
    assert "Here are licensed agents for Dubai Hills Estate." in agents
    assert "licence" in agents.lower() or "licensed" in agents.lower()
    assert "5" in agents
    assert len(agents.split()) > 40

    browse = build_browse_narrative(
        place="Dubai Marina", total=133, shown=10, beds=2, more=False
    )
    assert "133" in browse or "Dubai Marina" in browse
    assert "recent" in browse.lower() or "sale" in browse.lower()
    assert len(browse.split()) > 40

    motive = build_motive("recommend")
    assert "Acot Recommend" in motive
    assert "sales" in motive.lower()

    towers = build_family_motive({"kids": True, "avoid_towers": True, "want_villas": True})
    general = build_family_motive({})
    assert towers != general
    assert "tower" in towers.lower() or "low-rise" in towers.lower()
    assert "villa" in towers.lower()


def test_format_ranked_picks_leads_with_understood():
    projects = [
        {
            "name": "Tower A",
            "price": 1_100_000,
            "bedroom_min": 1,
            "bedroom_max": 1,
            "sales_count": 12,
            "community": "Dubai Marina",
            "price_kind": "recent_sale",
        },
        {
            "name": "Tower B",
            "price": 1_050_000,
            "bedroom_min": 1,
            "bedroom_max": 1,
            "sales_count": 8,
            "community": "Dubai Marina",
            "price_kind": "recent_sale",
        },
    ]
    text = format_ranked_picks(
        rank_projects(projects, wanted_beds=1, budget_aed=1_200_000, attach_sales=False),
        intro="For a 1-bedroom in Dubai Marina at up to AED 1,200,000,",
        question="Best 1 bedroom in Dubai Marina under 1.2M",
        place="Dubai Marina",
        beds=1,
        budget_aed=1_200_000,
        goal="budget",
    )
    assert text.startswith("Understood:")
    assert "Dubai Marina" in text.split("\n", 1)[0]
    assert "Tower A" in text or "Tower B" in text
    assert "**Acot Recommend:**" in text


def test_clarify_leads_with_understood():
    ask = clarify_question_text(
        {"action": "search", "beds": 2, "place": None, "budget_aed": None},
        "looking for 2 bedroom apartments",
    )
    assert ask.startswith("Understood:")
    assert "community" in ask.lower() or "area" in ask.lower()


def test_prepend_does_not_double():
    body = "Understood: ready homes in JVC.\n\nHere are 3 options:"
    assert prepend_understood(body, "Understood: ready homes in JVC.") == body
