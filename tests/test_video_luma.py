import json
import tempfile
import unittest
from pathlib import Path

import httpx

from app.Viedo_generation.luma import LumaError, generate_entrance_clip


class LumaPilotTests(unittest.TestCase):
    def test_submits_frames_polls_and_downloads_without_api_key_on_asset_request(self):
        def respond(request):
            if request.method == "POST":
                self.assertEqual(request.headers["Authorization"], "Bearer test-key")
                payload = json.loads(request.content)
                self.assertEqual(payload["model"], "ray-3.2")
                self.assertEqual(payload["video"]["end_frame"]["url"], "https://photos.example/interior.jpg")
                return httpx.Response(200, json={"id": "pilot", "state": "pending"})
            if request.url.host == "agents.lumalabs.ai":
                self.assertEqual(request.url.path, "/v1/generations/pilot")
                return httpx.Response(200, json={
                    "id": "pilot", "state": "completed",
                    "output": [{"url": "https://assets.example/clip.mp4"}],
                })
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, content=b"mock-video")

        with tempfile.TemporaryDirectory() as work, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            output = Path(work) / "pilot.mp4"
            args = (client, "test-key", "https://photos.example/exterior.jpg",
                    "https://photos.example/interior.jpg", output)
            self.assertEqual(generate_entrance_clip(*args, initial_wait=0), "pilot")
            self.assertEqual(output.read_bytes(), b"mock-video")
            with self.assertRaisesRegex(LumaError, "new .mp4"):
                generate_entrance_clip(*args)
