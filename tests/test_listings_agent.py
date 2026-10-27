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


if __name__ == "__main__":
    unittest.main()
