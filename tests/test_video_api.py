import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.Viedo_generation import catalog_prompt, jobs, router1


class VideoApiTests(unittest.TestCase):
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
