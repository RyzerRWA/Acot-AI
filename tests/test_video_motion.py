"""Offline timing, composition and encode checks for the property renderer."""
import importlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

from app.Viedo_generation.motion import caption_schedule, draw_overlay, shot_timing
from app.Viedo_generation.presenter import presenter_bounds, resolve_gender

renderer = importlib.import_module("app.Viedo_generation.render")


class VideoMotionTests(unittest.TestCase):
    def test_dissolves_do_not_shorten_timeline(self):
        for frames in (200, 251, 748, 1800):
            for count in range(1, 5):
                lengths, offsets = shot_timing(frames, count)
                self.assertEqual((offsets[-1] if offsets else 0) + lengths[-1], frames)

    def test_slide_timing_uses_narration_and_preserves_frames(self):
        slides = [{"narration": "one two"}, {"narration": "one two three four six seven"}]
        durations = renderer._slide_durations(slides, 10.03)
        self.assertAlmostEqual(sum(durations), 10.04)
        self.assertAlmostEqual(durations[1] / durations[0], 3, delta=0.05)

    def test_all_facts_survive_caption_paging(self):
        facts = ["Developer: Example", "Bedrooms: 2", "Sale AED 1 million", "Rent AED 70,000", "Ready"]
        pages = caption_schedule({"title": "Example", "lines": facts}, 250)
        self.assertEqual(set(pages[0][0]), set(facts))
        self.assertEqual(pages[0][1:], (0, 250))

    def test_agent_is_below_photos_and_gender_matches(self):
        with patch.dict(os.environ, {}, clear=True):
            x, y, width, height = presenter_bounds(1080, 1920)
            self.assertGreaterEqual(y, 1920 * 0.70)
            self.assertLessEqual(y + height, 1920)
            self.assertLessEqual(x + width, 1080)
            self.assertEqual(resolve_gender("en-US-Neural2-F"), "female")
            self.assertEqual(resolve_gender("en-US-Neural2-D"), "male")

    def test_broken_photos_fail_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "_download_image", return_value=None):
            with self.assertRaisesRegex(renderer.RenderError, "No usable property photos"):
                renderer.render({"ok": True}, {"ok": True, "slides": [{"title": "Example", "photo_urls": ["broken"]}]}, output_path=Path(directory) / "result.mp4")

    def test_multiple_slides_and_both_presenters_keep_duration(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "WIDTH", 360), patch.object(renderer, "HEIGHT", 640), patch.dict(os.environ, {"VIDEO_PRESENTER_ENABLED": "true"}):
            root = Path(directory)
            source = root / "fixture.png"
            Image.new("RGB", (600, 600), "steelblue").save(source)
            slides = [{"kind": kind, "title": "Test property", "lines": ["Example"], "photo_urls": ["fixture"]} for kind in ("intro", "project", "outro")]
            for voice in ("male", "female"):
                out = root / f"{voice}.mp4"
                with patch.object(renderer, "_download_image", return_value=source), patch.object(renderer, "_total_seconds", return_value=8.12):
                    renderer.render({"ok": True}, {"ok": True, "slides": slides}, output_path=out, presenter=voice)
                self.assertAlmostEqual(renderer._total_seconds({}, out), 8.12, delta=0.04)

    def test_encode_motion_overlay_audio_and_duration(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "WIDTH", 360), patch.object(renderer, "HEIGHT", 640), patch.dict(os.environ, {"VIDEO_PRESENTER_ENABLED": "false"}):
            root = Path(directory)
            sources = []
            for i, colour in enumerate(("steelblue", "sandybrown", "seagreen", "slateblue")):
                source = root / f"fixture{i}.png"
                photo = Image.new("RGB", (720, 480), colour)
                draw = ImageDraw.Draw(photo)
                for x in range(30, 720, 80):
                    draw.rectangle((x, 60, x + 40, 360), fill="white")
                photo.save(source)
                sources.append(source)
            slide = {"kind": "project", "title": "Test property", "subtitle": "Test community", "lines": ["Developer: Example", "Bedrooms: 2", "Sale AED 1 million", "Rent AED 70,000", "Ready"], "photo_urls": [str(i) for i in range(4)]}
            ffmpeg = renderer._ffmpeg_bin()
            audio = root / "tone.wav"
            renderer._run([ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=14", str(audio)], "Test audio")
            out = root / "result.mp4"
            with patch.object(renderer, "_download_image", side_effect=lambda url, _: sources[int(url)]):
                renderer.render({"ok": True}, {"ok": True, "slides": [slide]}, audio_path=audio, output_path=out)
            probe = subprocess.run([ffmpeg, "-i", str(out), "-f", "null", "-"], capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr[-1000:])
            self.assertIn("360x640", probe.stderr)
            self.assertIn("Audio: aac", probe.stderr)
            self.assertAlmostEqual(renderer._total_seconds({}, out), 14.0, delta=0.08)
            snapshots = []
            for name, timestamp in (("start", 0.2), ("moving", 1.3), ("middle", 7), ("last", 13.8)):
                path = root / f"{name}.png"
                renderer._run([ffmpeg, "-y", "-ss", str(timestamp), "-i", str(out), "-frames:v", "1", str(path)], "Test snapshot")
                with Image.open(path) as frame:
                    snapshots.append(frame.convert("RGB"))
            self.assertIsNotNone(ImageChops.difference(snapshots[0].crop((0, 150, 360, 400)), snapshots[1].crop((0, 150, 360, 400))).getbbox())
            # Panel stays opaque during photo dissolves, and final frames exist.
            for frame in snapshots:
                self.assertLess(max(frame.getpixel((350, 620))), 40)
            overlay = root / "overlay.png"
            draw_overlay(overlay, slide, ["Bedrooms: 2"], 360, 640)
            with Image.open(overlay) as frame:
                self.assertEqual(frame.getpixel((180, 250))[3], 0)


if __name__ == "__main__":
    unittest.main()
