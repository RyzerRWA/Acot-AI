import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.Viedo_generation import collector
from app.Viedo_generation.pack import _dedupe_photos


class CollectorTests(unittest.TestCase):
    def run_collect(self, found, work):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "k", "VIDEO_WEB_COLLECT": "true"}),
            patch.object(collector, "_search", return_value=found) as search,
            patch.object(collector, "_page_images", return_value=["https://dev.example/page.jpg"]),
            patch.object(collector, "_usable", side_effect=lambda urls: urls),
        ):
            result = collector.collect("KEY", "Tower", "Creek", "Dev", cache_dir=Path(work))
            again = collector.collect("KEY", "Tower", "Creek", "Dev", cache_dir=Path(work))
        return result, again, search

    def test_portals_blocked_amenities_need_official_source_and_results_cached(self):
        found = {
            "official_url": "https://dev.example/tower",
            "amenities": ["Pool", "Gym"],
            "image_urls": ["https://www.bayut.com/x.jpg", "http://dev.example/insecure.jpg", "https://dev.example/a.jpg"],
        }
        with tempfile.TemporaryDirectory() as work:
            result, again, search = self.run_collect(found, work)
        self.assertEqual(result["photos"], ["https://dev.example/a.jpg", "https://dev.example/page.jpg"])
        self.assertEqual(result["amenities"], ["Pool", "Gym"])
        self.assertEqual(again, result)
        search.assert_called_once()  # second call is served from the cache: no second paid search

        portal = {**found, "official_url": "https://www.propertyfinder.ae/en/tower"}
        with tempfile.TemporaryDirectory() as work:
            result, _, _ = self.run_collect(portal, work)
        self.assertIsNone(result["official_url"])
        self.assertEqual(result["amenities"], [])

    def test_same_image_on_two_hosts_is_kept_once(self):
        urls = ["https://www.emaar.com/u/hero.jpg", "https://cms.emaar.com/u/hero.jpg?w=2", "https://www.emaar.com/u/pool.jpg"]
        self.assertEqual(_dedupe_photos(urls), [urls[0], urls[2]])


if __name__ == "__main__":
    unittest.main()


class InsidePhotosTests(unittest.TestCase):
    def test_rooms_get_their_own_part_and_plans_are_dropped(self):
        from app.Viedo_generation.pack import _split_inside
        from app.Viedo_generation.script import build_script

        labels = [
            {"url": "pool", "kind": "amenity", "room": None},
            {"url": "bath", "kind": "interior", "room": "bathroom"},
            {"url": "tower", "kind": "exterior", "room": None},
            {"url": "bed1", "kind": "interior", "room": "bedroom"},
            {"url": "plan", "kind": "other", "room": None},
            {"url": "living", "kind": "interior", "room": "living room"},
            {"url": "bed2", "kind": "interior", "room": "bedroom"},
        ]
        outside, rooms = _split_inside([x["url"] for x in labels] + ["extra"], labels)
        self.assertEqual(outside, ["tower", "pool", "extra"])            # building first, plan dropped
        self.assertEqual([x["url"] for x in rooms], ["living", "bed1", "bath", "bed2"])  # one of each room first
        self.assertEqual(_split_inside(["a", "b"], []), (["a", "b"], []))  # no labels: nothing changes

        project = {"name": "Tower", "community": "Creek", "photos": outside + [x["url"] for x in rooms],
                   "interior_photos": rooms, "handover": "ready",
                   "recent_sale": {"price": 1_000_000}, "rent": {"annual": 90_000}}
        for language, phrase in (("en", "Now let's step inside. Here's a look at the living room, bedroom and bathroom."),
                                 ("hi", "चलिए, अब घर के अंदर चलते हैं। देखिए लिविंग रूम, बेडरूम और बाथरूम।")):
            script = build_script({"ok": True, "projects": [project]}, language=language)
            kinds = [s["kind"] for s in script["slides"]]
            self.assertEqual(kinds, ["intro", "project", "inside", "outro"])
            inside = script["slides"][2]
            self.assertEqual(inside["narration"], phrase)
            self.assertEqual(inside["photo_urls"], ["living", "bed1", "bath", "bed2"])
            # Room shots never appear in the outside parts.
            for slide in script["slides"]:
                if slide["kind"] != "inside":
                    self.assertFalse(set(slide["photo_urls"]) & {"living", "bed1", "bath", "bed2"})
