"""Build audio and caption timings from the same decoded speech segments.

No word-count timing for voiced videos: each TTS segment is decoded to PCM,
measured, and padded to a video-frame boundary before the next one is added.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import wave
from typing import Any

FPS = 25
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
MIN_CUE_FRAMES = 4 * FPS
PAUSE_SAMPLES = round(0.28 * SAMPLE_RATE)


def narrate_script(script: dict[str, Any], path: str | Path, *, voice: str | None = None) -> dict[str, Any]:
    """Write a WAV and return a copy of the script with exact frame boundaries."""
    from app.Viedo_generation.tts import TtsError, resolve_voice, speak
    from app.Viedo_generation.render import _ffmpeg_bin

    if not script.get("ok") or not script.get("slides"):
        raise TtsError("Cannot narrate an empty or incomplete script.")
    out = Path(path)
    if out.suffix.lower() != ".wav":
        raise TtsError("Timed narration output must be a WAV file.")
    timed = deepcopy(script)
    voice_name = resolve_voice(voice)
    ffmpeg = _ffmpeg_bin()
    chunks = []
    total_frames = 0
    with tempfile.TemporaryDirectory(prefix="acot_narration_") as directory:
        source = Path(directory) / "cue.mp3"
        for slide in timed["slides"]:
            cues = slide.get("cues") or [{"narration": slide.get("narration"), "lines": slide.get("lines", [])}]
            sections = []
            for cue in cues:
                text = str(cue.get("narration") or "").strip()
                if not text:
                    raise TtsError("A narration section is empty.")
                source.write_bytes(speak(text, voice=voice_name))
                decoded = subprocess.run(
                    [ffmpeg, "-v", "error", "-i", str(source), "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
                    capture_output=True,
                )
                if decoded.returncode or not decoded.stdout or len(decoded.stdout) % 2:
                    raise TtsError("Could not decode a narration section for synchronization.")
                sections.append(decoded.stdout)
            target = slide.get("target_frames")
            minimum = 3 * FPS if target else MIN_CUE_FRAMES
            lengths = [
                max(minimum, math.ceil((len(pcm) // 2 + PAUSE_SAMPLES) / SAMPLES_PER_FRAME))
                for pcm in sections
            ]
            if target:
                # Distribute time across cues in proportion to measured speech.
                # Cumulative rounding guarantees the requested slide length.
                total = sum(lengths)
                boundary = elapsed = 0
                for index, frames in enumerate(lengths):
                    elapsed += frames
                    end = round(elapsed * target / total)
                    lengths[index] = end - boundary
                    boundary = end
            slide_frames = 0
            for cue, pcm, frames in zip(cues, sections, lengths, strict=True):
                capacity = frames * SAMPLES_PER_FRAME
                if len(pcm) // 2 + PAUSE_SAMPLES > capacity:
                    # Preserve pitch if a long project or price exceeds its slot.
                    room = capacity - PAUSE_SAMPLES
                    rate = len(pcm) / (2 * room) * 1.06 if room > 0 else math.inf
                    if rate > 2:
                        raise TtsError("A narration section is too long for the video.")
                    adjusted = subprocess.run(
                        [ffmpeg, "-v", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE),
                         "-ac", "1", "-i", "pipe:0", "-af", f"atempo={rate:.6f}",
                         "-f", "s16le", "pipe:1"],
                        input=pcm, capture_output=True,
                    )
                    if adjusted.returncode or not adjusted.stdout or len(adjusted.stdout) // 2 > capacity:
                        raise TtsError("Could not fit narration into its video timing.")
                    pcm = adjusted.stdout
                samples = len(pcm) // 2
                padding = capacity - samples
                chunks.append(pcm + b"\0\0" * padding)
                cue.update(start_frame=slide_frames, end_frame=slide_frames + frames, speech_samples=samples)
                slide_frames += frames
            slide["cues"] = cues
            slide["duration_frames"] = slide_frames
            slide["start_frame"] = total_frames
            total_frames += slide_frames
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(b"".join(chunks))
    timed["timing"] = {"fps": FPS, "total_frames": total_frames, "voice": voice_name}
    timed["estimated_seconds"] = total_frames / FPS
    out.with_suffix(".timing.json").write_text(json.dumps(timed, ensure_ascii=False, indent=2), encoding="utf-8")
    return timed
