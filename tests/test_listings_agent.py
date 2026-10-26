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


if __name__ == "__main__":
    unittest.main()
