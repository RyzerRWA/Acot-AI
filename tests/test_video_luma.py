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


class LumaShotsTests(unittest.TestCase):
    def test_parallel_shots_are_cached_capped_and_fall_back_to_stills(self):
        import os
        from unittest.mock import patch

        from app.Viedo_generation import luma

        calls = []

        def fake(client, key, url, interior, output, **kwargs):
            calls.append(url)
            if "bad" in url:
                raise luma.LumaError("boom")
            output.write_bytes(b"clip")
            return "job"

        urls = ["https://p.example/a.jpg", "https://p.example/bad.jpg", "http://p.example/plain.jpg",
                "https://p.example/c.jpg", "https://p.example/a.jpg"]
        env = {"LUMA_AGENTS_API_KEY": "k", "VIDEO_LUMA_MAX_SHOTS": "2"}
        with tempfile.TemporaryDirectory() as work, patch.dict(os.environ, env), patch.object(luma, "generate_entrance_clip", fake):
            shots = luma.animate_photos(urls, Path(work))
            self.assertTrue(shots["https://p.example/a.jpg"].is_file())
            self.assertIsNone(shots["https://p.example/bad.jpg"])
            self.assertIsNone(shots["http://p.example/plain.jpg"])   # not HTTPS
            self.assertIsNone(shots["https://p.example/c.jpg"])      # over the cap of 2
            self.assertEqual(sorted(calls), ["https://p.example/a.jpg", "https://p.example/bad.jpg"])
            calls.clear()
            again = luma.animate_photos(["https://p.example/a.jpg"], Path(work))
            self.assertEqual(calls, [])                               # cached, no second paid call
            self.assertEqual(again["https://p.example/a.jpg"], shots["https://p.example/a.jpg"])
            with patch.dict(os.environ, {"VIDEO_LUMA_ENABLED": "false"}):
                self.assertEqual(luma.animate_photos(["https://p.example/a.jpg"], Path(work)), {"https://p.example/a.jpg": None})

    def test_luma_clip_renders_in_place_of_a_still(self):
        import importlib
        import os
        import subprocess
        from unittest.mock import patch

        from PIL import Image

        renderer = importlib.import_module("app.Viedo_generation.render")
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "WIDTH", 360), patch.object(renderer, "HEIGHT", 640), patch.dict(os.environ, {"VIDEO_PRESENTER_ENABLED": "false"}):
            root = Path(directory)
            ffmpeg = renderer._ffmpeg_bin()
            still = root / "still.png"
            Image.new("RGB", (600, 600), "steelblue").save(still)
            clip = root / "luma.mp4"   # shorter than its slot, odd size and frame rate, like a real 9:16 Luma clip
            renderer._run([ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=size=720x1280:rate=24:duration=2", "-pix_fmt", "yuv420p", str(clip)], "Test clip")
            slides = [{"kind": "project", "title": "Test", "lines": ["Ready"], "photo_urls": ["a", "b"]}]
            out = root / "result.mp4"
            with (
                patch.object(renderer, "_download_image", return_value=still),
                patch.object(renderer, "_total_seconds", return_value=12.0),
                patch("app.Viedo_generation.luma.animate_photos", return_value={"a": clip, "b": None}),
            ):
                renderer.render({"ok": True}, {"ok": True, "slides": slides}, output_path=out)
            probe = subprocess.run([ffmpeg, "-i", str(out), "-f", "null", "-"], capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr[-1000:])
            self.assertIn("360x640", probe.stderr)
            self.assertAlmostEqual(renderer._total_seconds({}, out), 12.0, delta=0.08)
