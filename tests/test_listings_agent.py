"""Free checks for the listings agent: no database, no language model.  Run: python -m unittest tests.test_listings_agent"""
import unittest

from app.listings.agent import AgentResult, ListingAgent, strip_links
from app.listings.router import is_rent_question
from app.listings.tools import ToolInputError, parse_filters, run_tool


class FilterTests(unittest.TestCase):
    def test_monthly_budgets_become_yearly(self):
        f = parse_filters({"max_monthly_rent": 9000, "min_monthly_rent": 5000})
        self.assertEqual((f["max_annual_rent"], f["min_annual_rent"]), (108000, 60000))
        self.assertEqual(parse_filters({"max_annual_rent": 50000})["max_annual_rent"], 50000)

    def test_bad_arguments_are_rejected(self):
        bad = [{"beds_min": "x"}, {"beds_min": 5, "beds_max": 2}, {"furnishing": "half"}, {"max_annual_rent": -1},
               {"min_annual_rent": 9, "max_annual_rent": 1}, {"areas": ["a"] * 9}, {"areas": ["x" * 81]}, [1, 2]]
        for arguments in bad:
            with self.assertRaises(ToolInputError, msg=str(arguments)):
                parse_filters(arguments)
        self.assertEqual(parse_filters(None), parse_filters({}))
        self.assertEqual(parse_filters({"areas": "Marina"})["areas"], ["Marina"])

    def test_unknown_tool_and_sql_injection_shaped_input(self):
        with self.assertRaises(ToolInputError):
            run_tool("drop_everything", {})
        with self.assertRaises(ToolInputError):
            run_tool("area_stats", {"group_by": "community; drop table listings"})
        with self.assertRaises(ToolInputError):
            run_tool("get_listing", {"listing_id": "1; drop table listings"})


class LinkTests(unittest.TestCase):
    def test_links_and_images_never_reach_the_user(self):
        self.assertEqual(" ".join(strip_links("a ![x](https://a.b/c.jpg) b").split()), "a b")
        self.assertEqual(strip_links("[Nice flat](https://www.bayut.com/x) is [[123]]"), "Nice flat is [[123]]")
        self.assertEqual(" ".join(strip_links("go to https://evil.com now").split()), "go to now")
        self.assertEqual(strip_links("[[16793835]] AED 90,000"), "[[16793835]] AED 90,000")

    def test_invented_listing_ids_are_dropped(self):
        seen = {"111": {"listing_id": "111", "title": "real"}}
        result: AgentResult = ListingAgent._finish("See [[111]] and [[999]].", seen, [], "m", 0, 0.0)
        self.assertEqual([c["listing_id"] for c in result.cards], ["111"])
        self.assertNotIn("999", result.answer)
        self.assertIn("[[111]]", result.answer)


class RouterTests(unittest.TestCase):
    def test_rent_questions_go_to_the_listings_agent(self):
        for q in ["What's the average rent for a 2 bedroom in Dubai Marina?", "Cheapest studio to rent in JVC",
                  "Furnished 3 bedroom villas under 250,000 a year", "2 bedrooms in Marina under AED 9,000 a month",
                  "Show me rental listings in Business Bay"]:
            self.assertTrue(is_rent_question(q), q)

    def test_buying_and_project_questions_stay_with_the_existing_pipeline(self):
        for q in ["Which developers have off-plan projects in JVC?", "What is the starting price of Sky Edition?",
                  "Should I buy or rent in Marina?", "rental yield in Business Bay for investors",
                  "What are the handover dates?", "Tell me about Dubai Hills"]:
            self.assertFalse(is_rent_question(q), q)

    def test_follow_ups_stay_in_the_rental_conversation_only(self):
        self.assertTrue(is_rent_question("and in JLT?", "listings"))
        self.assertFalse(is_rent_question("and in JLT?", None))
        self.assertFalse(is_rent_question("What are the handover dates?", "listings"))
        self.assertFalse(is_rent_question("", "listings"))


class ProjectToolTests(unittest.TestCase):
    """Project tools and the user profile. These checks need no database and no language model."""

    def test_unknown_price_is_never_shown_as_zero_and_bad_sizes_are_blanked(self):
        from app.listings.tools import _project_for_model
        row = {"id": 1, "name": "X", "price": 0, "size_min": 0, "size_max": 225, "bedroom_min": 0,
               "bedroom_max": 2, "handover_time": None, "developer_name": " Dev "}
        shown = _project_for_model(row)
        self.assertIsNone(shown["starting_price_aed"])
        self.assertIsNone(shown["size_sqft_min"])
        self.assertEqual(shown["size_sqft_max"], 225.0)
        self.assertEqual(shown["developer"], "Dev")

    def test_budget_filters_exclude_projects_with_no_price(self):
        from app.listings.tools import _project_filters, _project_where
        where, params = _project_where(_project_filters({"max_price_aed": 1_500_000}))
        self.assertIn("price > 0", where)
        self.assertEqual(params, [1_500_000])

    def test_project_size_filter(self):
        from app.listings.tools import _project_filters, _project_where
        where, params = _project_where(_project_filters({"min_size_sqft": 1000}))
        self.assertIn("size_max >= %s", where)
        self.assertEqual(params, [1000])

    def test_bad_project_arguments_are_rejected(self):
        from app.listings.tools import search_projects, compare_projects, get_project
        for tool, arguments in ((search_projects, {"sort": "x; drop table projects"}),
                                (search_projects, {"min_price_aed": 9, "max_price_aed": 1}),
                                (search_projects, {"communities": ["a"] * 9}),
                                (compare_projects, {"projects": ["only one"]}),
                                (compare_projects, {"projects": "not a list"}),
                                (get_project, {})):
            with self.assertRaises(ToolInputError, msg=str(arguments)):
                tool(arguments)

    def test_profile_is_validated_and_kept_in_the_session_dict(self):
        from app.listings.tools import clean_profile, update_profile
        self.assertEqual(clean_profile({"budget_max_aed": "1500000", "goal": "invest", "family_size": 4}),
                         {"budget_max_aed": 1500000, "goal": "invest", "family_size": 4})
        for bad in ({"goal": "flip"}, {"budget_min_aed": 5, "budget_max_aed": 1}, {"family_size": 0}, [1]):
            with self.assertRaises(ToolInputError, msg=str(bad)):
                clean_profile(bad)
        saved: dict = {}
        update_profile({"budget_max_aed": 900000}, profile=saved)
        update_profile({"goal": "buy_to_live"}, profile=saved)
        self.assertEqual(saved, {"budget_max_aed": 900000, "goal": "buy_to_live"})
        update_profile({"clear": True}, profile=saved)
        self.assertEqual(saved, {})

    def test_project_cards_are_kept_only_for_ids_a_tool_returned(self):
        seen = {"233000800": {"listing_id": "233000800", "kind": "project", "row": {"name": "Sky"}}}
        result = ListingAgent._finish("Try [[233000800]] or [[999]].", seen, [], "m", 0, 0.0)
        self.assertEqual([c["kind"] for c in result.cards], ["project"])
        self.assertNotIn("999", result.answer)

    def test_a_project_named_without_a_tag_still_gets_its_card(self):
        seen = {"1": {"listing_id": "1", "kind": "project", "row": {"name": "Sky Edition at Seahaven"}},
                "2": {"listing_id": "2", "kind": "project", "row": {"name": "Other Tower"}}}
        result = ListingAgent._finish("Sky Edition at Seahaven starts from AED 21,132,340.", seen, [], "m", 0, 0.0)
        self.assertEqual([c["listing_id"] for c in result.cards], ["1"])

    def test_stray_name_and_id_lines_are_removed(self):
        from run import _drop_repeated_name_lines
        text = "1. **Samana Waves**\n   - Starting: AED 483,277\n   - **ID:** Samana Waves\n2. Azizi Ruby\n   - Azizi Ruby"
        out = _drop_repeated_name_lines(text, {"Samana Waves", "Azizi Ruby"})
        self.assertNotIn("ID:", out)
        self.assertEqual(out.count("Samana Waves"), 1)
        self.assertEqual(out.count("Azizi Ruby"), 1)
        self.assertIn("AED 483,277", out)


class GraphTests(unittest.TestCase):
    """The LangGraph agent, driven by a scripted model and a fake tool: no database, no credits."""

    @staticmethod
    def _graph(script, max_rounds=5):
        from types import SimpleNamespace as NS
        from app.graph.graph import build_agent_graph
        calls = {"chat": 0}

        def chat(messages):
            reply = script[min(calls["chat"], len(script) - 1)]
            calls["chat"] += 1
            if isinstance(reply, str):
                message = NS(content=reply, tool_calls=None)
            else:
                message = NS(content=None, tool_calls=[NS(id="c1", function=NS(name=reply[0], arguments=reply[1]))])
            return "fake", NS(usage=NS(total_tokens=10), choices=[NS(message=message)])

        def run(name, args):
            return NS(output={"price": 21132340}, cards=[{"listing_id": "1", "kind": "project", "row": {"name": "Sky"}}])

        graph = build_agent_graph(chat, run, ToolInputError, max_rounds=max_rounds)
        start = {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "price of Sky?"}],
                 "seen": {}, "used": [], "tokens": 0, "rounds": 0}
        return graph.invoke(start), calls

    def test_tool_then_answer(self):
        state, calls = self._graph([("get_project", "{}"), "From AED 21,132,340. [[1]]"])
        self.assertEqual(calls["chat"], 2)
        self.assertEqual(state["used"], [{"name": "get_project", "ok": True}])
        self.assertIn("1", state["seen"])
        self.assertEqual(state["tokens"], 20)

    def test_a_figure_no_tool_returned_is_sent_back_once(self):
        state, calls = self._graph([("get_project", "{}"), "It costs AED 19,500,000.", "It costs AED 21,132,340."])
        self.assertEqual(calls["chat"], 3)
        self.assertEqual(state["text"], "It costs AED 21,132,340.")

    def test_only_one_retry_even_if_the_model_repeats_itself(self):
        state, calls = self._graph([("get_project", "{}"), "AED 19,500,000."])
        self.assertEqual(calls["chat"], 3)
        self.assertEqual(state["text"], "AED 19,500,000.")

    def test_figures_from_the_question_years_and_small_numbers_are_not_flagged(self):
        from app.graph.graph import unsupported_numbers
        messages = [{"role": "system", "content": "AED 90,000 example"},
                    {"role": "user", "content": "under 1.5 million please"}]
        self.assertEqual(unsupported_numbers("Under AED 1,500,000, handover 2028, 3 bedrooms, 1.5 million.", messages), [])
        self.assertEqual(unsupported_numbers("AED 90,000", messages), [90000.0])

    def test_a_question_about_a_known_place_answered_from_memory_is_sent_back_for_data(self):
        from types import SimpleNamespace as NS
        from app.graph.graph import build_agent_graph
        replies = ["Dubai Marina is a vibrant waterfront community.", ("market_stats", "{}"), "It has 1 project. [[1]]"]
        seen_calls = {"n": 0}

        def chat(messages):
            reply = replies[seen_calls["n"]]
            seen_calls["n"] += 1
            message = NS(content=reply, tool_calls=None) if isinstance(reply, str) else NS(
                content=None, tool_calls=[NS(id="c", function=NS(name=reply[0], arguments=reply[1]))])
            return "fake", NS(usage=NS(total_tokens=1), choices=[NS(message=message)])

        graph = build_agent_graph(chat, lambda n, a: NS(output={"projects": 1}, cards=[]), ToolInputError,
                                  needs_grounding=lambda q: "marina" in q.lower())
        state = graph.invoke({"messages": [{"role": "system", "content": "rules"},
                                           {"role": "user", "content": "What is Dubai Marina?"}],
                              "seen": {}, "used": [], "tokens": 0, "rounds": 0})
        self.assertEqual(seen_calls["n"], 3)
        self.assertEqual(state["used"], [{"name": "market_stats", "ok": True}])
        self.assertIn("1 project", state["text"])

    def test_general_guidance_or_unrelated_questions_are_not_forced_to_use_a_tool(self):
        state, calls = self._graph(["General guidance: escrow protects buyers."])
        self.assertEqual(calls["chat"], 1)

    def test_doubled_names_after_a_tag_are_collapsed(self):
        from run import _collapse_doubled_names
        text = 'Samana Waves "Samana Waves" from AED 483,277, and Azizi Ruby Azizi Ruby from AED 618,000.'
        self.assertEqual(_collapse_doubled_names(text, {"Samana Waves", "Azizi Ruby"}),
                         "Samana Waves from AED 483,277, and Azizi Ruby from AED 618,000.")

    def test_endless_tool_calls_stop_at_the_round_limit(self):
        state, calls = self._graph([("get_project", "{}")], max_rounds=3)
        self.assertEqual(calls["chat"], 3)
        self.assertIn("couldn't pin that down", state["text"])


class QuestionUnderstandingTests(unittest.TestCase):
    """Regressions: "I want 10 1-bedroom apartments" was read as a reference to projects number 10 and 1."""

    @classmethod
    def setUpClass(cls):
        from app.memory.conversation_memory import ConversationMemory
        cls.memory = ConversationMemory(llm_client=None)
        cls.memory.active_candidates = [{"name": f"Project {i}"} for i in range(1, 21)]

    def test_quantities_are_not_references_to_earlier_results(self):
        m = self.memory
        self.assertIsNone(m._extract_numeric_pair("I want 10 1-bedroom apartments?"))
        self.assertIsNone(m._extract_numeric_pair("show 3 and 4 bedroom villas"))
        self.assertEqual(m._extract_numeric_pair("compare 1 and 2"), [1, 2])
        self.assertFalse(m._looks_like_follow_up("I want 10 1-bedroom apartments?"))
        self.assertFalse(m._looks_like_follow_up("one bedroom apartments under 2 million"))
        self.assertTrue(m._looks_like_follow_up("compare 1 and 2"))
        self.assertTrue(m._looks_like_follow_up("what is their price"))

    def test_a_single_ordinal_is_resolved_without_the_model(self):
        m = self.memory
        self.assertEqual(m._rewrite_candidate_reference("what is the price of the first one"),
                         'what is the price of "Project 1"')
        self.assertEqual(m._rewrite_candidate_reference("tell me about the third"), 'tell me about "Project 3"')
        self.assertIsNone(m._rewrite_candidate_reference("is the first time buyer discount available"))

    def test_number_words_for_bedrooms_read_like_digits(self):
        from app.retrieval.hybrid.question_text import prepare_user_question
        from app.retrieval.hybrid.query_planner import QueryPlanner
        text = prepare_user_question("I want 10 one-bedroom apartments.")
        self.assertEqual(QueryPlanner._extract_filters(text), {"bedrooms": 1, "property_types": ["apartment"]})

    def test_wanting_a_kind_of_property_is_a_search_not_a_field_lookup(self):
        from app.retrieval.hybrid.query_planner import QueryPlanner
        planner = QueryPlanner(llm_client=None, use_llm=False)
        for question in ("I want 1-bedroom apartments?", "looking for 3 bedroom villas", "1 bedroom apartments"):
            self.assertEqual(planner._rule_plan(question).intent, "search", question)
        self.assertEqual(planner._rule_plan("what is the handover of Azizi Ruby").intent, "information")

    def test_a_browse_shows_the_requested_number_of_projects(self):
        from app.answers.structured_answer import _limit_browse_projects
        projects = [{"name": f"P{i}"} for i in range(50)]
        summary = {"projects": projects}
        shown, summary, browsing = _limit_browse_projects("i want 10 1-bedroom apartments.", projects, summary)
        self.assertTrue(browsing)
        self.assertEqual((len(shown), len(summary["projects"]), summary["project_total"]), (10, 10, 50))
        shown, _, _ = _limit_browse_projects("i want 1-bedroom apartments", list(projects), {"projects": projects})
        self.assertEqual(len(shown), 10)                       # no count asked: a short default list
        _, _, browsing = _limit_browse_projects("what are the bedrooms of these", projects, {})
        self.assertFalse(browsing)


class SessionStoreTests(unittest.TestCase):
    """Saved chat memory, against a fake in-memory database (nothing is written to the real one)."""

    def setUp(self):
        import json, os
        from unittest import mock
        from app.listings import session_store
        self.store, self.rows = session_store, {}
        session_store._ready = False

        def execute(sql, params=()):
            if sql.startswith("insert"):
                self.rows[params[0]] = {"history": json.loads(params[1]), "profile": json.loads(params[2])}
            elif sql.startswith("delete"):
                self.rows.pop(params[0], None)

        def fetch(sql, params=()):
            return [self.rows[params[0]]] if params[0] in self.rows else []

        patches = [mock.patch.object(session_store, "execute", execute), mock.patch.object(session_store, "fetch", fetch),
                   mock.patch.dict(os.environ, {"ACOT_PERSIST_SESSIONS": "1"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_memory_survives_a_restart_and_clear_forgets_it(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        self.store.save("s1", history, {"budget_max_aed": 900000})
        self.assertEqual(len(self.rows["s1"]["history"]), 12)           # only the last turns are kept
        loaded_history, loaded_profile = self.store.load("s1")
        self.assertEqual(loaded_history[-1]["content"], "q19")
        self.assertEqual(loaded_profile, {"budget_max_aed": 900000})
        self.store.delete("s1")
        self.assertEqual(self.store.load("s1"), ([], {}))

    def test_it_does_nothing_unless_switched_on(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"ACOT_PERSIST_SESSIONS": "0"}):
            self.store.save("s2", [{"role": "user", "content": "hi"}], {})
            self.assertNotIn("s2", self.rows)
            self.assertEqual(self.store.load("s2"), ([], {}))

    def test_a_database_error_never_breaks_the_chat(self):
        from unittest import mock
        with mock.patch.object(self.store, "execute", side_effect=RuntimeError("db down")):
            self.store.save("s3", [], {})                     # must not raise
            self.assertEqual(self.store.load("s3"), ([], {}))


if __name__ == "__main__":
    unittest.main()
