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
    def test_short_video_has_two_projects_and_a_45_second_audio_timeline(self):
        project = {"name": "Example", "photos": ["photo"]}
        script = build_script({"ok": True, "projects": [
            project, {**project, "name": "Second"}, {**project, "name": "Excluded"},
        ]})
        self.assertEqual([s["kind"] for s in script["slides"]], ["intro", "project", "project", "outro"])
        self.assertEqual(renderer._slide_durations(script["slides"], 45), [5, 18, 18, 4])
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            with patch("app.Viedo_generation.tts.speak", return_value=tone(2.0, 220)):
                timed = narrate_script(script, audio, voice="female")
            self.assertEqual(renderer._slide_durations(timed["slides"], 45), [5, 18, 18, 4])
            with wave.open(str(audio), "rb") as stream:
                self.assertEqual(stream.getnframes(), 45 * 24000)

    def test_one_picked_property_makes_a_single_property_video(self):
        photos = [f"photo{i}" for i in range(6)]
        script = build_script({"ok": True, "projects": [{"name": "Creek Residences", "community": "Creek", "photos": photos}]})
        self.assertTrue(script["ok"])
        self.assertEqual([s["kind"] for s in script["slides"]], ["intro", "project", "outro"])
        self.assertEqual(renderer._slide_durations(script["slides"], 33), [5, 24, 4])
        self.assertEqual(script["slides"][0]["title"], "Creek Residences")
        self.assertIn("This is Creek Residences, in Creek.", script["slides"][1]["narration"])
        self.assertNotIn("either", script["full_text"])
        # Intro, middle and close each show different real photos.
        self.assertEqual([s["photo_urls"] for s in script["slides"]], [["photo0"], photos[1:5], ["photo5"]])

    def test_cinematic_comparison_starts_on_each_property_and_moves_inside(self):
        project = {"name": "First", "photos": ["building", "pool", "living", "bath"],
                   "interior_photos": [{"url": "living"}, {"url": "bath"}]}
        script = build_script({"ok": True, "projects": [project, {"name": "Second", "photos": ["facade"]}]},
                              cinematic=True)
        self.assertEqual([slide["kind"] for slide in script["slides"]], ["project", "project"])
        self.assertEqual([slide["photo_urls"] for slide in script["slides"]], [["building", "living", "bath"], ["facade"]])
        self.assertEqual(renderer._slide_durations(script["slides"], 45), [22.48, 22.52])

    def test_named_amenities_have_matching_narration_and_precede_the_agent(self):
        project = {"name": "Garden View", "photos": ["building", "pool", "play", "room"],
                   "interior_photos": [{"url": "room", "room": "living room"}],
                   "amenity_urls": ["pool", "play"],
                   "amenity_photos": [{"url": "pool", "name": "Swimming pool"},
                                      {"url": "play", "name": "Children's play area"}],
                   "amenities": ["Shared pool", "Gym"]}
        script = build_script({"ok": True, "projects": [project]}, cinematic=True,
                              broker_card={"name": "Demo Agent"})
        self.assertEqual([slide["kind"] for slide in script["slides"]],
                         ["project", "inside", "amenity", "amenity", "amenity", "agent"])
        self.assertEqual(script["slides"][0]["photo_urls"], ["building"])
        for slide, label, photo in zip(script["slides"][2:5],
                                      ["Swimming pool", "Children's play area", "Gym"],
                                      ["pool", "play", "building"], strict=True):
            self.assertIn(label, slide["narration"])
            self.assertEqual(slide["photo_urls"], [photo])
        self.assertIn("Demo Agent", script["slides"][-1]["narration"])
        self.assertEqual(script["slides"][-1]["target_frames"], 125)

    def test_facts_are_paired_with_their_narration(self):
        project = {"name": "Example", "community": "Community", "developer": "Developer", "bedrooms_min": 2, "photos": ["photo"], "recent_sale": {"price": 1200000, "bedrooms": 2, "date": "2026-10-01"}, "rent": {"annual": 80000, "bedrooms": 2, "date": "2026-10-02"}, "handover": "ready"}
        script = build_script({"ok": True, "projects": [project, {**project, "name": "Second"}]})
        cues = script["slides"][1]["cues"]
        self.assertEqual(len(cues), 4)
        self.assertIn("sold for about 1.2 million dirhams", cues[1]["narration"])
        self.assertTrue(cues[1]["lines"][0].startswith("Sale "))
        self.assertIn("80,000 dirhams a year", cues[2]["narration"])
        self.assertTrue(cues[2]["lines"][0].startswith("Rent "))
        self.assertEqual(" ".join(cue["narration"] for cue in cues), script["slides"][1]["narration"])

    def test_hindi_narration_keeps_the_same_facts(self):
        project = {"name": "Example", "community": "Community", "photos": ["p"] * 5,
                   "recent_sale": {"price": 1200000, "bedrooms": 2, "date": "2026-10-01"},
                   "rent": {"annual": 80000, "bedrooms": 0, "date": "2026-10-02"}, "handover": "Dec 2027"}
        script = build_script({"ok": True, "projects": [project]}, language="hi")
        text = script["full_text"]
        for fact in ("Example", "1.2 मिलियन दिरहम", "80 हज़ार दिरहम", "स्टूडियो", "दिसंबर 2027", "नमस्ते"):
            self.assertIn(fact, text)
        self.assertEqual([s["label"] for s in script["slides"][1]["stats"]], ["RECENT SALE", "RECENT RENT", "HANDOVER"])

    def test_invalid_timings_fail_instead_of_drifting(self):
        with self.assertRaises(renderer.RenderError):
            renderer._slide_durations([{"duration_frames": 100}, {}], 8)
        with self.assertRaises(renderer.RenderError):
            renderer._slide_durations([{"duration_frames": 100}], 20)
        with self.assertRaises(renderer.RenderError):
            caption_schedule({"duration_frames": 250, "cues": [{"start_frame": 1, "end_frame": 250}]}, 250)

    def test_reel_shows_agent_panel_for_exactly_five_seconds_with_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "photo.png"
            frame = Image.new("RGB", (640, 360), (20, 180, 50))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 639, 60), fill="red")
            draw.rectangle((0, 300, 639, 359), fill="blue")
            frame.save(photo)
            audio = root / "speech.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                with wave.open(io.BytesIO(tone(3.0, 220)), "rb") as speech:
                    wav.writeframes(speech.readframes(speech.getnframes()))
                wav.writeframes(b"\0\0" * (4 * 24000))
            slide = {"title": "Example", "photo_urls": ["fixture"], "duration_frames": 25,
                     "cues": [{"start_frame": 0, "end_frame": 25, "lines": []}]}
            agent_slide = {"kind": "agent", "duration_frames": 125,
                           "cues": [{"start_frame": 0, "end_frame": 125, "lines": []}]}
            script = {"ok": True, "slides": [slide, slide, agent_slide]}
            dld_photo = "https://mobile.dubailand.gov.ae/example.jpg"
            with (
                patch.object(renderer, "_download_image", return_value=photo),
                patch.object(renderer, "_broker_photo", return_value=Image.new("RGB", (300, 300), "#973728")) as profile,
                patch("app.Viedo_generation.luma.animate_photos", return_value={}),
                patch.dict(os.environ, {"VIDEO_PRESENTER_ENABLED": "false"}),
            ):
                out = renderer.render(
                    {"ok": True}, script, audio_path=audio, output_path=root / "reel.mp4",
                    broker_card={"name": "Demo Agent", "email": "agent@example.test", "photo_url": dld_photo, "dld_stars": 5},
                )
                profile.assert_called_once_with(dld_photo)
            ffmpeg = renderer._ffmpeg_bin()
            def pixels(x, y):
                result = subprocess.run(
                    [ffmpeg, "-hide_banner", "-i", str(out), "-vf", f"crop=2:2:{x}:{y}",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                    capture_output=True, check=True,
                )
                self.assertIn(b"1080x1920", result.stderr)
                return [result.stdout[i:i + 3] for i in range(0, len(result.stdout), 12)]

            avatar_edge = pixels(540, 510)
            self.assertEqual(len(avatar_edge), 175)
            self.assertTrue(all(pixel[0] < 50 for pixel in avatar_edge[:50]))
            self.assertTrue(all(pixel[0] > 100 for pixel in avatar_edge[50:]))
            self.assertTrue(all(pixel[0] > 60 for pixel in pixels(540, 1160)[50:]))
            self.assertTrue(all(pixel[0] > 200 for pixel in pixels(550, 20)[:50]))
            self.assertTrue(all(pixel[2] > 35 for pixel in pixels(550, 1900)[:50]))
            self.assertTrue(all(pixel[0] < 50 for pixel in pixels(990, 1000)[50:]))
            audio_check = subprocess.run(
                [ffmpeg, "-v", "error", "-ss", "3.2", "-i", str(out), "-map", "0:a:0",
                 "-f", "s16le", "-ar", "24000", "pipe:1"],
                capture_output=True, check=True,
            )
            samples = list(struct.iter_unpack("<h", audio_check.stdout))
            self.assertGreaterEqual(len(samples), 3.76 * 24000)
            self.assertLess(max(abs(value) for value, in samples), 10)

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
            # Slides dissolve over 0.8 s starting at the narration boundary (10 s).
            for timestamp, dominant_channel in ((5.92, 0), (6.04, 2), (9.92, 2), (10.88, 1), (13.92, 1)):
                result = subprocess.run([ffmpeg, "-v", "error", "-ss", str(timestamp), "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
                frame = Image.frombytes("RGB", (360, 640), result.stdout)
                rgb = frame.getpixel((180, 570))
                self.assertEqual(rgb.index(max(rgb)), dominant_channel, (timestamp, rgb))
            # Mid-dissolve: the outgoing blue caption and incoming green caption are both visible (no hard cut).
            result = subprocess.run([ffmpeg, "-v", "error", "-ss", "10.4", "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
            _, green, blue = Image.frombytes("RGB", (360, 640), result.stdout).getpixel((180, 570))
            self.assertGreater(min(green, blue), 50, (green, blue))
            self.assertAlmostEqual(renderer._total_seconds({}, out), 14, delta=0.04)


if __name__ == "__main__":
    unittest.main()
