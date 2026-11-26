"""Phase 5: turn understanding without demo-sentence routes in run.py."""
import asyncio

from app.answers.turn_resolve import offline_resolve_turn, resolve_turn


def test_offline_project_count_any_wording():
    a = offline_resolve_turn("How many residential projects are there in Dubai?")
    b = offline_resolve_turn("What's the number of projects across Dubai?")
    assert a["action"] == "project_count"
    assert b["action"] == "project_count"
    assert (a["place"] or "Dubai") == "Dubai"


def test_offline_sales_count_and_list():
    count = offline_resolve_turn(
        "How many 1-bedroom apartments in JVC were sold in November 2026?"
    )
    assert count["action"] == "sales_count"
    assert count["beds"] == 1
    assert "Jumeirah" in (count["place"] or "") or count["place"] == "JVC"
    assert count["month"] == 11
    assert count["year"] == 2026

    listed = offline_resolve_turn("Show me recent sales in Business Bay")
    assert listed["action"] == "sales_list"
    assert listed["place"] == "Business Bay"


def test_offline_recommend_family_buy():
    route = offline_resolve_turn(
        "I want a 3-bedroom in JVC for a family of 4 to buy around AED 1.8M."
    )
    assert route["action"] == "recommend"
    assert route["goal"] == "family"
    assert route["beds"] == 3
    assert route["budget_aed"] == 1_800_000


def test_offline_search_browse():
    route = offline_resolve_turn("Show me 2 bedroom apartments in Dubai Marina")
    assert route["action"] == "search"
    assert route["beds"] == 2
    assert route["place"] == "Dubai Marina"


def test_offline_incomplete_browse_is_clarify():
    route = offline_resolve_turn("looking for 2 bedroom apartments")
    assert route["action"] == "clarify"


def test_llm_resolve_unusual_wording_without_new_regex():
    async def fake_llm(_prompt):
        return """{
          "action": "sales_count",
          "goal": null,
          "place": "Jumeirah Village Circle",
          "beds": 1,
          "budget_aed": null,
          "month": 11,
          "year": 2026,
          "ready_only": false,
          "prefer_cheaper": false,
          "kind": null
        }"""

    resolution = asyncio.run(
        resolve_turn(
            "Could you check how many one-beds changed hands in JVC last November?",
            None,
            llm_generate=fake_llm,
        )
    )
    assert resolution is not None
    assert resolution["source"] == "llm"
    assert resolution["action"] == "sales_count"
    assert resolution["place"] == "Jumeirah Village Circle"
    assert resolution["beds"] == 1


def test_top_agents_for_place_is_agents_not_projects():
    """Regression: 'top agents for Dubai Hills' must not list projects."""
    from app.answers.turn_resolve import _apply_agents_override

    q = "Tell me the top agents for Dubai Hills Estate"
    offline = offline_resolve_turn(q)
    assert offline is not None
    assert offline["action"] == "agents"
    assert "Hills" in (offline.get("place") or "")

    # Even if a model wrongly chooses search, force agents.
    fixed = _apply_agents_override(
        {"action": "search", "place": "Dubai Hills Estate", "confidence": 0.9},
        q,
    )
    assert fixed["action"] == "agents"
    assert "Hills" in (fixed.get("place") or "")

    called = {"n": 0}

    async def boom(_prompt):
        called["n"] += 1
        raise AssertionError("clear agents ask must skip LLM")

    resolution = asyncio.run(resolve_turn(q, None, llm_generate=boom))
    assert called["n"] == 0
    assert resolution["action"] == "agents"


def test_llm_failure_falls_back_to_offline_understanding():
    """Keyword rules below 0.9 no longer pre-empt the LLM (rewordings were misrouted),
    but when the LLM call fails the offline understanding still answers."""
    called = {"n": 0}

    async def boom(_prompt):
        called["n"] += 1
        raise RuntimeError("model unavailable")

    resolution = asyncio.run(
        resolve_turn(
            "Which Dubai areas are best for families?",
            None,
            llm_generate=boom,
        )
    )
    assert called["n"] == 1
    assert resolution is not None
    assert resolution["source"] == "offline"
    assert resolution["action"] == "recommend"
    assert resolution["kind"] == "family_areas"


def test_understood_turn_executes_search_from_resolve(monkeypatch):
    import run as run_mod
    from app.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(llm_client=None)
    calls = {}

    def fake_agent_projects(filters, limit=10, sort="latest_handover", skip_names=None):
        calls["filters"] = filters
        return (
            [{
                "name": "Marina Gate",
                "community": "Dubai Marina",
                "price": 1_200_000,
                "entity_type": "project",
            }],
            1,
        )

    monkeypatch.setattr(
        "app.retrieval.propai_catalog.agent_projects",
        fake_agent_projects,
    )
    out = asyncio.run(
        run_mod._understood_turn(
            "Show me 2 bedroom apartments in Dubai Marina",
            memory,
            {},
        )
    )
    assert out is not None
    assert out["structured_summary"]["projects"]
    assert memory.get_dialogue_state()["action"] == "search"
    assert calls["filters"].get("bedrooms") == 2


def test_near_certain_offline_skips_llm():
    """Explicit stat words (0.92) still answer without an LLM hop."""
    async def boom(_prompt):
        raise AssertionError("LLM should not run")

    resolution = asyncio.run(
        resolve_turn("What is the cheapest sale in JVC?", None, llm_generate=boom)
    )
    assert resolution["source"] == "offline"
    assert resolution["action"] == "sales_stat" and resolution["stat"] == "cheapest"


def test_offline_parses_relative_month():
    from datetime import date
    from app.answers.turn_resolve import _parse_month_year

    today = date.today()
    prev = (today.month - 2) % 12 + 1
    assert _parse_month_year("sales in marina last month") == (prev, today.year - (today.month == 1))
    assert _parse_month_year("this month") == (today.month, today.year)


def test_handover_rankings_resolve_offline():
    from app.answers.turn_resolve import offline_resolve_turn

    for question in ("which project have the recent handover to users ?", "recently completed projects in JVC"):
        assert offline_resolve_turn(question, None)["action"] == "handovers"
        assert offline_resolve_turn(question, None)["kind"] == "recent"
    assert offline_resolve_turn("upcoming handovers", None)["kind"] == "upcoming"
    # Pointing at the cards on screen stays an attribute ask.
    assert (offline_resolve_turn("their handover dates?", None) or {}).get("action") != "handovers"


def test_typo_repair_keeps_real_words():
    from app.retrieval.hybrid.question_text import repair_typed_question as repair

    # Real words were being "corrected": rents/parents -> agents, central -> rental, dates -> latest.
    assert repair("cheapest rents in Dubai") == "cheapest rents in Dubai"
    assert repair("parents with kids") == "parents with kids"
    assert repair("compare sobha central and trump tower") == "compare sobha central and trump tower"
    assert repair("handovr dates") == "handover dates"
    # "nearest hotel" became "newest hotel" and answered with a handover list.
    for word in ("nearest", "closest", "biggest", "largest", "nicest", "newer", "cheaper", "visas"):
        assert repair(f"show me {word} in dubai") == f"show me {word} in dubai"
    # Genuine typos are still fixed.
    assert repair("agnets in marina") == "agents in marina"
    assert repair("chepest 1 bed") == "cheapest 1 bed"


def test_new_question_types_are_in_the_schema():
    from app.answers.turn_resolve import _ACTIONS, _clean_resolution

    for action in ("rent_stat", "area_market", "area_rank", "top_projects", "project_detail", "guidance", "handovers"):
        assert action in _ACTIONS
    cleaned = _clean_resolution({"action": "project_count", "developer": "Danube"}, "How many projects does Danube have?")
    assert cleaned["developer"] == "Danube"


def test_roi_adds_price_growth_to_yield(monkeypatch):
    """ROI and rental yield are different answers: ROI = yield + yearly price growth."""
    import run
    import app.answers.area_comparison as area
    import app.retrieval.propai_live as live

    monkeypatch.setattr(area, "official_area", lambda name: "AREA")
    monkeypatch.setattr(area, "market_signals", lambda official: {"price_psf": 1500, "price_psf_change_pct": 2.9, "sales_12m": 100})
    monkeypatch.setattr(live, "area_returns", lambda official: {"rental_yield_pct": 6.7, "growth_3y_pct": 9.3, "growth_years": (2023, 2026)})
    roi = run._area_market_turn("what is the ROI of jvc", {"place": "JVC", "kind": "roi"})["fast_answer"]
    yld = run._area_market_turn("rental yield in jvc", {"place": "JVC", "kind": "yield"})["fast_answer"]
    assert "**16.0% a year**" in roi
    assert "about **6.7%** a year" in yld
    assert roi != yld


def test_too_few_sales_means_no_trend(monkeypatch):
    """One sale must not become "+47%": below the minimum, predict says there is not enough data."""
    import run
    import app.answers.area_comparison as area
    import app.retrieval.propai_live as live

    monkeypatch.setattr(area, "official_area", lambda name: "AREA")
    monkeypatch.setattr(area, "market_signals", lambda official: {"price_psf": 1891, "price_psf_change_pct": 47.4, "sales_12m": 1})
    monkeypatch.setattr(live, "area_returns", lambda official: None)
    text = run._predict_turn("will prices go up in Al Qusais?", {"place": "Al Qusais", "action": "predict"})["fast_answer"]
    assert "don't have enough recorded sales" in text and "47.4" not in text


def test_propai_failure_reuses_last_good_figure(monkeypatch, tmp_path):
    """A Prop-AI timeout must not flip the yield back to our DB figure (6.7% one minute, 6.4% the next)."""
    import app.retrieval.propai_live as live

    monkeypatch.setattr(live, "_CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(live, "_CACHE", {"JVC": [0, {"rental_yield_pct": 6.7}]})  # stale entry
    monkeypatch.setattr(live, "_client", lambda: (_ for _ in ()).throw(TimeoutError("slow")))
    assert live.area_returns("jvc") == {"rental_yield_pct": 6.7}
    assert live.area_returns("Unknown Area") is None


def test_area_profile_mixes_web_text_with_our_figures(monkeypatch):
    """"Tell me about JVC": web profile + our key figures + sources; no web -> the investment view."""
    import run
    import app.answers.area_comparison as area
    import app.retrieval.propai_live as live
    import app.retrieval.web_context as web

    monkeypatch.setattr(area, "official_area", lambda name: "AREA")
    monkeypatch.setattr(area, "market_signals", lambda official: {"price_psf": 1492, "price_psf_change_pct": 2.9, "sales_12m": 11108})
    monkeypatch.setattr(live, "area_returns", lambda official: {"rental_yield_pct": 6.7, "growth_3y_pct": 9.3})
    monkeypatch.setattr(web, "area_profile", lambda place, key=None: {
        "text": "- **Transport:** No metro station.", "sources": [["Area Guide", "https://example.com/jvc"]]})
    text = run._area_market_turn("tell me about jvc", {"place": "JVC", "kind": "profile"})["fast_answer"]
    assert "No metro station" in text and "AED 1,492 per sq ft" in text and "6.7%" in text
    assert "example.com" not in text  # sources are not printed (the UI showed raw markdown links)

    monkeypatch.setattr(web, "area_profile", lambda place, key=None: None)
    text = run._area_market_turn("tell me about jvc", {"place": "JVC", "kind": "profile"})["fast_answer"]
    assert "as an investment" in text


def test_web_profile_parse_moves_citations_to_sources():
    from types import SimpleNamespace as NS
    from app.retrieval.web_context import _parse

    url = "https://www.propertyfinder.ae/en/area/jvc?utm_source=openai"
    response = NS(
        output_text="- **Transport:** No metro. ([propertyfinder.ae](%s))" % url,
        output=[NS(content=[NS(annotations=[NS(type="url_citation", url=url, title="JVC Guide")])])],
    )
    out = _parse(response)
    assert out["text"] == "- **Transport:** No metro."
    assert out["sources"] == [["JVC Guide", "https://www.propertyfinder.ae/en/area/jvc"]]
    assert _parse(NS(output_text="UNKNOWN", output=[])) is None
    plain = _parse(NS(output_text="Try [Pitfire Pizza](https://x.com/p) at Circle Mall.", output=[]))
    assert plain["text"] == "Try Pitfire Pizza at Circle Mall."


def test_area_profile_puts_the_asked_facet_first(monkeypatch):
    import run
    import app.answers.area_comparison as area
    import app.retrieval.propai_live as live
    import app.retrieval.web_context as web

    monkeypatch.setattr(area, "official_area", lambda name: "AREA")
    monkeypatch.setattr(area, "market_signals", lambda official: {"sales_12m": 100})
    monkeypatch.setattr(live, "area_returns", lambda official: None)
    monkeypatch.setattr(web, "area_profile", lambda place, key=None: {"sources": [], "text":
        "- **What it's like:** Suburban.\n- **Getting around:** No metro station.\n- **Schools:** JSS."})
    text = run._area_market_turn("is there a metro in jvc?", {"place": "JVC", "kind": "profile"})["fast_answer"]
    assert text.index("No metro station") < text.index("Suburban")


def test_guidance_is_answered_from_the_web(monkeypatch):
    """"Best restaurant near JVC?" was refused ("our data doesn't cover restaurants"); the web answers it."""
    import run
    import app.retrieval.web_context as web

    seen = {}

    def fake(question, place=None):
        seen["place"] = place
        return {"text": "Popular picks include Circle Mall restaurants.", "sources": [["Guide", "https://example.com"]]}

    monkeypatch.setattr(web, "web_answer", fake)
    out = run._web_answer_turn("best restaurant near JVC?", {"action": "guidance", "place": "JVC"})["fast_answer"]
    assert "Circle Mall" in out and "example.com" not in out and seen["place"] == "JVC"
    monkeypatch.setattr(web, "web_answer", lambda question, place=None: None)
    assert run._web_answer_turn("golden visa rules", {"action": "guidance"}) is None  # agent fallback


def test_pdf_sources_show_the_site_not_pdf_text():
    from app.retrieval.web_context import _source_title

    assert _source_title("DUBAI\nSUBSCRIBE\n41°C", "https://palm.fivehotels.com/media/x.pdf") == "palm.fivehotels.com"
    assert _source_title("JVC Guide", "https://www.bayut.com/jvc") == "JVC Guide"
    assert _source_title("", "https://www.bayut.com/jvc") == "bayut.com"
