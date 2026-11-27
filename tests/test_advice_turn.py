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
    assert asyncio.run(advice_answer("Repeat my brief", memory, None)) is None


def test_median_price_is_not_paired_with_a_sale_date():
    # R01: "Recent sale AED 1.66m (recorded 2 Nov)" read as one sale; the price is a 12-month median.
    from app.answers.structured_answer import _price_clause

    median = _price_clause({"price": 1_660_000, "price_kind": "recent_sale", "last_sold_on": "2026-11-02"})
    assert median.startswith("Median recorded sale AED 1,660,000")
    assert "latest sale recorded 2026-11-02" in median
    latest = _price_clause({"price": 3_160_000, "price_kind": "latest_sale", "last_sold_on": "2026-11-02"})
    assert latest == "Latest sale AED 3,160,000 (recorded 2026-11-02)"
