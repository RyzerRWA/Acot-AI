"""ACOT QA (Nov 2026): calculations, laws, privacy and recall asks are answered, not turned into searches."""
import asyncio
from types import SimpleNamespace

import pytest

from app.answers.advice_turn import (
    advice_answer,
    advice_kind,
    availability_note,
    document_note,
)


@pytest.mark.parametrize("question, kind", [
    # B11 / AI-026 / AI-027 / AI-032: inputs supplied, so compute.
    ("A 1m apartment rents for 70k. What is the gross yield?", "calc"),
    ("A flat costs AED 1.25m and rents for 92,500 annually. Give gross yield and explain what a buyer still needs for net yield.", "calc"),
    ("Price 1.5m, acquisition fees 90k, rent 120k, 5% vacancy, management 5% of collected rent, service charges 16k "
     "and maintenance 6k. Compute NOI and both net yields.", "calc"),
    ("NOI is 78k, annual debt service 66k, deposit 300k and closing costs 60k. Compute year-one cash-on-cash "
     "excluding principal paydown.", "calc"),
    ("Calculate the monthly payment and total interest on AED 1.2m over 25 years at 4.5% nominal fixed.", "calc"),
    # AI-079 / R25: a challenge to an earlier figure.
    ("Your example says AED 60k NOI on AED 1.2m is 8%. Please check the math, explain the consequence and correct the estimate.", "challenge"),
    # AI-042 / AI-043 / AI-073: law and process, not agent cards or listings.
    ("A broker says a buyer must submit a BRN before contacting an agent. Whose BRN is it, and is that a Dubai law, "
     "brokerage policy or website rule?", "regulated"),
    ("Show how to verify a broker's BRN, firm's ORN and a listing's advertising permit separately.", "regulated"),
    ("My off-plan handover is nine months late. Which documents and escalation route should I check before demanding a refund?", "regulated"),
    # AI-051 / AI-054 / B19: privacy, trust and personal finance.
    ("I'm buying my first home with 320k savings. I want a 300k deposit and have no emergency fund. How would you decide whether to wait?", "personal"),
    ("I distrust agents after a bait-and-switch. How can I research a unit privately before giving anyone my number?", "personal"),
    # AI-061 to AI-070 / R08: the family brief.
    ("My family wants a two-bedroom apartment up to AED 1.2m, in JVC or Arjan. Keep a running brief.", "recall"),
    ("Starting this new conversation, what budget have I explicitly given you in this thread?", "recall"),
    ("Back in our family thread, repeat the final brief and distinguish recalled facts from assumptions.", "recall"),
    # AI-020: a made-up sample is arithmetic. AI-081: a promised return is answered before any listing.
    ("Business Bay off-plan registrations grew from 100 to 130 while ready fell from 100 to 90 in a hypothetical "
     "sample. Separate growth rates and mix.", "calc"),
    ("Find three verified apartments guaranteeing 15% net yield. Does a DLD sale record establish a future "
     "contractual guarantee?", "personal"),
    # AI-058 / AI-006 / AI-097 / AI-082: frameworks, conflicting goals, stress tests, unverifiable launches.
    ("Compare Dubai and London for a nonresident without assuming my tax residence. Which personal inputs alter the decision?", "personal"),
    ("I need rent from next month but want an unbuilt launch with a three-year handover. What can I buy that meets both?", "personal"),
    ("We need a 3-bed in 90 days, 2.1m all-in, school in Al Barsha and work at DIFC. Compare two areas, then show how "
     "losing car access changes your choice.", "personal"),
    ("Is QA Fictional Azure Lantern 391 by QA Fictional Sand Arc a registered launch? Give a source or say what cannot be verified.", "regulated"),
    ("Quote a DLD transaction from the next calendar month as if it has already closed. How would you verify it?", "personal"),
])
def test_answer_not_search(question, kind):
    assert advice_kind(question) == kind


@pytest.mark.parametrize("question", [
    "Show me 2 bed apartments in JVC under 1.2m",
    "2 bed in JVC under 1.2m with good yield",
    "2 bed in Marina between 1.5m and 2m, what is the price per sq ft?",
    "What is the rental yield in Dubai Marina?",
    "Best investment properties under AED 2M with high rental yield",
    "Top agents in Business Bay",
    "Find me an agent in JVC",
    "Recent sales in Creek Beach Surf",
    "Villas with private pool in Palm Jumeirah",
    "Is JVC a good investment?",
    "Compare Serenz and Aura Heights",
    "How much does a 1 bed cost in Arjan?",
    "average rent contracts in JVC 2025",
    "Show me projects in JVC with escrow accounts",
    "Show me projects in London",
])
def test_searches_stay_searches(question):
    assert advice_kind(question) is None


def test_availability_and_brochure_notes():
    assert "Availability unverified" in availability_note("Which vacant 1-bed in JVC can I actually view?")
    assert availability_note("Show me 2 beds in JVC") is None
    assert "not on file" in document_note("Find the official brochure for Creek Beach Surf", [])
    assert document_note("Find the official brochure for Creek Beach Surf", [{"document_url": "x"}]) is None
    assert document_note("Recent sales in Surf", []) is None


def test_answer_uses_thread_and_falls_through_without_model():
    memory = SimpleNamespace(turns=[
        {"user_question": "Two-bed up to AED 1.2m in JVC or Arjan.", "answer": "Noted."},
        {"user_question": "Increase the cap by AED 300k.", "answer": "Your cap is now AED 1.5m."},
    ])
    seen = {}

    async def fake_llm(prompt):
        seen["prompt"] = prompt
        return "Your budget is AED 1.5m."

    answer = asyncio.run(advice_answer("Repeat my brief", memory, fake_llm))
    assert answer == "Your budget is AED 1.5m."
    assert "Increase the cap by AED 300k." in seen["prompt"]
    assert "Repeat my brief" in seen["prompt"]
    assert "Today is " in seen["prompt"] and "{today}" not in seen["prompt"]
    assert asyncio.run(advice_answer("Repeat my brief", memory, None)) is None


def test_median_price_is_not_paired_with_a_sale_date():
    # R01: "Recent sale AED 1.66m (recorded 2 Nov)" read as one sale; the price is a 12-month median.
    from app.answers.structured_answer import _price_clause

    median = _price_clause({"price": 1_660_000, "price_kind": "recent_sale", "last_sold_on": "2026-11-02"})
    assert median.startswith("Median recorded sale AED 1,660,000")
    assert "latest sale recorded 2026-11-02" in median
    latest = _price_clause({"price": 3_160_000, "price_kind": "latest_sale", "last_sold_on": "2026-11-02"})
    assert latest == "Latest sale AED 3,160,000 (recorded 2026-11-02)"


def test_developer_filter_from_question(monkeypatch):
    # AI-012 / R07: "Emaar-only" must filter by developer; a place called "Damac Hills" must not.
    import app.retrieval.propai_catalog as catalog

    monkeypatch.setattr(catalog, "_developer_brands", None)
    monkeypatch.setattr(catalog, "fetch", lambda *a, **k: [{"g": "EMAAR"}, {"g": "DAMAC PROPERTIES"}, {"g": "DUBAI SOUTH"}])
    assert catalog.developers_in_question("Show me Emaar-only 2-beds under 2m") == ["EMAAR"]
    assert catalog.developers_in_question("projects by Damac in JVC") == ["DAMAC PROPERTIES"]
    assert catalog.developers_in_question("2 beds in Damac Hills") == []
    assert catalog.developers_in_question("How many Dubai South homes delivered in 2025?") == []
    assert catalog.developers_in_question("Show me 2 beds in JVC") == []


def test_scope_note_for_a_narrower_cohort():
    # R05 / AI-016, AI-018, AI-019: a 12-month sales median must not pass as the period, rent or psf asked for.
    from app.answers.advice_turn import scope_note

    generic = "Across **5,493** recorded 1-bedroom sales in JVC in the last 12 months, the median price is AED 1,100,000."
    assert "October 2026" in scope_note("For JVC ready 1-bed sales in October 2026, give the count", generic)
    assert "rent or lease" in scope_note("What was the median annual new-contract rent in JLT in 2025?", generic)
    assert "price per sq ft" in scope_note("Compare Marina median AED/sq ft in Q3 2025 and Q3 2026", generic)
    assert scope_note("Median 1-bed price in JVC", generic) is None
    assert scope_note("JVC sales in October 2026", "No figures.") is None


def test_latex_is_stripped_from_advice_answers():
    async def fake_llm(prompt):
        return r"- Off-plan growth: \((130 - 100) / 100 = 30.00\%\)"

    answer = asyncio.run(advice_answer("hypothetical sample", SimpleNamespace(turns=[]), fake_llm))
    assert answer == "- Off-plan growth: (130 - 100) / 100 = 30.00%"


def test_calculations_come_from_code_not_the_model():
    # AI-030 / AI-033: the model's arithmetic was off; the answer prompt now carries exact results.
    from app.answers.finance_calc import evaluate, results_block

    assert round(evaluate("pv(0.05/12, 240, 6000)"), 2) == 909151.88
    assert round(evaluate("pmt(0.045/12, 300, 1200000)"), 2) == 6669.99
    assert "7.1227" in results_block('{"calcs":[{"name":"annual IRR","expr":"irr([-1050000,60000,60000,1157600])"}]}')
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('x')")
    prompts = []

    async def fake_llm(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return '{"calcs":[{"name":"loan (AED)","expr":"pv(0.05/12, 240, 6000)"}]}'
        return "About AED 909,152."

    answer = asyncio.run(advice_answer(
        "My monthly income is 25k, existing debt payments 4k, cap 40%. At 5% for 20 years what loan can I service?",
        SimpleNamespace(turns=[]), fake_llm))
    assert answer == "About AED 909,152."
    assert "909,151.88" in prompts[1]
