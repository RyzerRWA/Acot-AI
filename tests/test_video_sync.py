"""Regression checks for narration-led captions and project boundaries."""
import importlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.Viedo_generation.motion import caption_schedule
from app.Viedo_generation.narration import narrate_script
from app.Viedo_generation.script import build_script

renderer = importlib.import_module("app.Viedo_generation.render")


def tone(seconds, frequency):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"".join(struct.pack("<h", round(5000 * math.sin(i * frequency * math.tau / 24000))) for i in range(round(seconds * 24000))))
    return buffer.getvalue()


class VideoSyncTests(unittest.TestCase):
    def test_facts_are_paired_with_their_narration(self):
        project = {"name": "Example", "community": "Community", "developer": "Developer", "bedrooms_min": 2, "photos": ["photo"], "recent_sale": {"price": 1200000, "bedrooms": 2, "date": "2026-10-01"}, "rent": {"annual": 80000, "bedrooms": 2, "date": "2026-10-02"}, "handover": "ready"}
        script = build_script({"ok": True, "projects": [project, {**project, "name": "Second"}]})
        cues = script["slides"][1]["cues"]
        self.assertEqual(len(cues), 4)
        self.assertIn("Recent sale:", cues[1]["narration"])
        self.assertTrue(cues[1]["lines"][0].startswith("Sale "))
        self.assertIn("Recent rent:", cues[2]["narration"])
        self.assertTrue(cues[2]["lines"][0].startswith("Rent "))
        self.assertEqual(" ".join(cue["narration"] for cue in cues), script["slides"][1]["narration"])

    def test_invalid_timings_fail_instead_of_drifting(self):
        with self.assertRaises(renderer.RenderError):
            renderer._slide_durations([{"duration_frames": 100}, {}], 8)
        with self.assertRaises(renderer.RenderError):
            renderer._slide_durations([{"duration_frames": 100}], 20)
        with self.assertRaises(renderer.RenderError):
            caption_schedule({"duration_frames": 250, "cues": [{"start_frame": 1, "end_frame": 250}]}, 250)

    def test_measured_audio_controls_encoded_captions_and_project_change(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "WIDTH", 360), patch.object(renderer, "HEIGHT", 640), patch.dict(os.environ, {"VIDEO_PRESENTER_ENABLED": "false"}):
            root = Path(directory)
            # Short text with long audio and long text with short audio deliberately
            # defeat word-count allocation. Encoded colour changes verify timing.
            cues = [{"narration": "Short numerals.", "lines": ["red"]}, {"narration": "This has substantially more words but its test audio is short.", "lines": ["blue"]}]
            script = {"ok": True, "slides": [{"title": "First", "photo_urls": ["fixture"], "cues": cues}, {"title": "Second", "photo_urls": ["fixture"], "cues": [{"narration": "Second project.", "lines": ["green"]}]}]}
            audio = root / "speech.wav"
            with patch("app.Viedo_generation.tts.speak", side_effect=[tone(5.7, 220), tone(1.2, 440), tone(3.37, 880)]):
                timed = narrate_script(script, audio, voice="female")
            self.assertNotIn("duration_frames", script["slides"][0])
            self.assertEqual(timed["slides"][0]["duration_frames"], 250)
            self.assertEqual(timed["slides"][1]["start_frame"], 250)
            self.assertEqual(renderer._slide_durations(timed["slides"], 14), [10, 4])
            self.assertEqual(caption_schedule(timed["slides"][0], 250), [(["red"], 0, 150), (["blue"], 150, 250)])
            self.assertEqual(json.loads(audio.with_suffix(".timing.json").read_text())["timing"]["total_frames"], 350)
            with wave.open(str(audio), "rb") as stream:
                self.assertEqual(stream.getnframes(), 14 * 24000)
                stream.setpos(round(5.8 * 24000))
                self.assertEqual(stream.readframes(100), b"\0\0" * 100)
                stream.setpos(6 * 24000)
                self.assertNotEqual(stream.readframes(100), b"\0\0" * 100)
            fixture = root / "photo.png"
            Image.new("RGB", (400, 400), "gray").save(fixture)
            def overlay(path, slide, lines, width, height):
                frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                ImageDraw.Draw(frame).rectangle((0, int(height * 0.68), width, height), fill=lines[0])
                frame.save(path)
            out = root / "aligned.mp4"
            with patch.object(renderer, "_download_image", return_value=fixture), patch("app.Viedo_generation.motion.draw_overlay", side_effect=overlay):
                renderer.render({"ok": True}, timed, audio_path=audio, output_path=out)
            ffmpeg = renderer._ffmpeg_bin()
            for timestamp, dominant_channel in ((5.92, 0), (6.04, 2), (9.92, 2), (10.04, 1), (13.92, 1)):
                result = subprocess.run([ffmpeg, "-v", "error", "-ss", str(timestamp), "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
                frame = Image.frombytes("RGB", (360, 640), result.stdout)
                rgb = frame.getpixel((180, 570))
                self.assertEqual(rgb.index(max(rgb)), dominant_channel, (timestamp, rgb))
            self.assertAlmostEqual(renderer._total_seconds({}, out), 14, delta=0.04)


if __name__ == "__main__":
    unittest.main()
