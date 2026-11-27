import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.Viedo_generation import catalog_prompt, jobs, pack, router1


class VideoApiTests(unittest.TestCase):
    def test_partial_comparison_offers_only_the_qualified_project(self):
        request = {"user_id": "buyer", "prompt_text": "SURF_KEY, SPRING_KEY"}
        candidates = [{"key": "SURF_KEY", "explicit_project": True}, {"key": "SPRING_KEY", "explicit_project": True}]
        qualified = {"key": "SURF_KEY", "name": "Creek Beach - Surf"}
        with (
            patch.object(jobs, "get_job", return_value={"request": request}),
            patch.object(jobs, "_update") as update,
            patch.object(catalog_prompt, "resolve_catalog_selection", return_value=candidates),
            patch.object(pack, "build_pack", return_value={"ok": False, "projects": [qualified], "reason": "Spring has no photos"}),
        ):
            jobs._run_job("test")
        self.assertEqual(update.call_args.kwargs["single_project_option"], qualified)
        self.assertEqual(update.call_args.kwargs["error"], "Spring has no photos")

    def test_interior_photos_count_toward_single_property_readiness(self):
        project = {"key": "SURF_KEY", "name": "Creek Beach - Surf", "community": "Dubai Creek Harbour",
                   "developer": "Developer", "locationKey": "surf", "status": "ready"}
        photos = ["exterior", "living", "bedroom"]
        labels = [{"url": "exterior", "kind": "exterior", "room": None},
                  {"url": "living", "kind": "interior", "room": "living room"},
                  {"url": "bedroom", "kind": "interior", "room": "bedroom"}]
        today = date.today().isoformat()
        with (
            patch.object(pack, "fetch", side_effect=[[project], [{"sales12m": 1, "rents12m": 1}]]),
            patch.object(pack, "_catalog_photos", return_value=photos),
            patch.object(pack, "_latest_sale", return_value={"price": 1, "date": today}),
            patch.object(pack, "_latest_rent", return_value={"annual": 1, "date": today}),
            patch("app.Viedo_generation.collector.collect", return_value={"amenities": [], "photos": []}),
            patch("app.Viedo_generation.collector.label_photos", return_value=labels),
        ):
            result = pack.build_pack([{"key": "SURF_KEY", "name": project["name"]}], min_projects=1, web=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["projects"][0]["photos"], photos)

    def test_video_pack_names_missing_catalog_photos(self):
        project = {"name": "Spring", "community": "Dubai Silicon Oasis", "developer": "Developer",
                   "locationKey": "spring", "completionDate": "2025-09-01"}
        with (
            patch.object(pack, "fetch", return_value=[project]),
            patch.object(pack, "_catalog_photos", return_value=[]),
            patch.object(pack, "_matching_listing_photos", return_value=[]),
        ):
            result = pack.build_pack([{"key": "SPRING_DSO", "name": "Spring"}], min_projects=1)
        self.assertFalse(result["ok"])
        self.assertIn("Spring: property photos (0 available, 4 needed)", result["reason"])

    def test_listing_photo_fallback_matches_both_the_project_and_master_community(self):
        with patch.object(pack, "fetch", return_value=[{"listing_id": 123, "stored_photos": 4}]) as fetch:
            photos = pack._matching_listing_photos("Spring", "Arabian Ranches III")
        self.assertEqual(fetch.call_args.args[1], ("spring", ["arabian ranches iii", "arabian ranches 3", "arabian ranches-3"]))
        self.assertEqual([url.rsplit("/", 2)[-2:] for url in photos],
                         [["123", f"{index}.jpeg"] for index in range(4)])

    def test_ambiguous_project_choices_keep_other_projects_in_the_request(self):
        rows = [
            {"key": "SPRING_DSO", "name": "Spring", "area": "Dubai Silicon Oasis"},
            {"key": "SPRING_AR3", "name": "Spring", "area": "Wadi Al Safa 5", "master_project": "Arabian Ranches III"},
        ]
        request = {"user_id": "buyer", "prompt_text": "Spring (2 bed), Ciel"}
        with (
            patch.object(catalog_prompt, "fetch", return_value=rows),
            patch.object(jobs, "get_job", return_value={"request": request}),
            patch.object(jobs, "_update") as update,
        ):
            jobs._run_job("test")
        status = update.call_args.kwargs
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["project_choices"][1], {
            "key": "SPRING_AR3", "name": "Spring", "community": "Arabian Ranches III",
            "prompt_text": "SPRING_AR3, Ciel",
        })

    def test_generation_requires_catalog_selection_without_history_fallback(self):
        with (
            patch.object(jobs, "get_job", return_value={"request": {"user_id": "buyer"}}),
            patch("app.Viedo_generation.history.searched_projects") as history,
            patch.object(catalog_prompt, "fetch") as fetch,
            self.assertRaisesRegex(catalog_prompt.PromptError, "project or community names"),
        ):
            jobs._pipeline("test")
        history.assert_not_called()
        fetch.assert_not_called()

    def test_explicit_legacy_inputs_resolve_in_the_catalog(self):
        with patch.object(catalog_prompt, "catalog_candidates") as resolve:
            catalog_prompt.resolve_catalog_selection(None, ["CIEL_KEY"], ["Mudon"])
            resolve.assert_called_once_with("CIEL_KEY, Mudon")

    def test_agent_input_uses_catalog_even_without_history(self):
        row = {"key": "PROJECT_KEY", "name": "Example", "area": "Example area"}
        with (
            patch.object(jobs, "get_job", return_value={"request": {
                "user_id": "buyer", "prompt_text": "Example (2 bed)", "with_voice": False,
            }}),
            patch.object(catalog_prompt, "fetch", return_value=[row]) as fetch,
            patch("app.Viedo_generation.history.searched_projects") as history,
            patch("app.Viedo_generation.pack.build_pack", return_value={"ok": False}) as pack,
            self.assertRaises(catalog_prompt.PromptError),
        ):
            jobs._pipeline("test")
        history.assert_not_called()
        self.assertEqual(fetch.call_args.args[1], ("example",) * 3)
        self.assertEqual(pack.call_args.args[0][0]["key"], "PROJECT_KEY")
        self.assertEqual(pack.call_args.kwargs["min_projects"], 1)

    def test_scoped_job_and_range_request(self):
        app = FastAPI()
        app.include_router(router1.router)
        job = {"id": "test-job", "status": "ready", "request": {"scope": "lead-a"}}
        headers = {"X-Video-Key": "internal", "X-Video-Scope": "lead-a"}
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "test-job.mp4"
            path.write_bytes(b"0123456789")
            with (
                patch.object(router1, "INTERNAL_KEY", "internal"),
                patch.object(jobs, "get_job", return_value=job),
                patch.object(router1.storage, "mp4_path", return_value=path),
                TestClient(app) as client,
            ):
                status_url = "/api/video/jobs/test-job"
                file_url = "/api/video/files/test-job.mp4"
                self.assertEqual(client.get("/api/video/health").status_code, 200)
                self.assertEqual(client.get(status_url, headers=headers).status_code, 200)
                self.assertEqual(client.get(file_url).status_code, 401)
                wrong = {**headers, "X-Video-Scope": "lead-b"}
                self.assertEqual(client.get(status_url, headers=wrong).status_code, 404)
                self.assertEqual(client.get(file_url, headers=wrong).status_code, 404)
                response = client.get(file_url, headers={**headers, "Range": "bytes=2-4"})
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.headers["content-range"], "bytes 2-4/10")
                self.assertEqual(response.content, b"234")
