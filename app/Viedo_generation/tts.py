"""
Google Cloud Text-to-Speech for the video narration.

speak(text) -> MP3 bytes
speak_to_file(text, path) -> Path

Auth (pick one):
  - GOOGLE_APPLICATION_CREDENTIALS = path to a service-account JSON
  - or: gcloud auth application-default login

Env:
  GOOGLE_TTS_VOICE   default en-US-Neural2-D
  GOOGLE_TTS_LANGUAGE default en-US
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Neural2 is clear and reasonably priced for short narrations.
DEFAULT_VOICE = "en-US-Neural2-D"
DEFAULT_LANGUAGE = "en-US"
# Google TTS hard limit is 5000 bytes of input text per request.
MAX_CHARS = 4500


class TtsError(RuntimeError):
    """Raised when speech cannot be synthesized."""


def speak(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    speaking_rate: float = 1.0,
) -> bytes:
    """
    Synthesize `text` to MP3 bytes via Google Cloud Text-to-Speech.

    Raises TtsError if credentials are missing or the API call fails.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise TtsError("Narration text is empty.")
    if len(cleaned) > MAX_CHARS:
        raise TtsError(
            f"Narration is too long ({len(cleaned)} characters). "
            f"Keep it under {MAX_CHARS} for one TTS request."
        )

    voice_name = (voice or os.getenv("GOOGLE_TTS_VOICE") or DEFAULT_VOICE).strip()
    language_code = (language or os.getenv("GOOGLE_TTS_LANGUAGE") or DEFAULT_LANGUAGE).strip()
    # Voice names are like en-US-Neural2-D — language should match the prefix.
    if "-" in voice_name:
        prefix = "-".join(voice_name.split("-")[:2])
        if prefix:
            language_code = prefix

    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise TtsError(
            "google-cloud-texttospeech is not installed. "
            "Run: python -m pip install google-cloud-texttospeech"
        ) from exc

    try:
        client = texttospeech.TextToSpeechClient()
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=cleaned),
            voice=texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=max(0.7, min(1.3, float(speaking_rate))),
            ),
        )
    except Exception as exc:  # credentials / quota / API errors
        message = str(exc)
        if "default credentials" in message.lower() or "could not load" in message.lower():
            raise TtsError(
                "Google TTS credentials are missing. Set "
                "GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path, "
                "or run: gcloud auth application-default login"
            ) from exc
        raise TtsError(f"Google TTS failed: {message}") from exc

    audio = response.audio_content
    if not audio:
        raise TtsError("Google TTS returned empty audio.")
    return bytes(audio)


def speak_to_file(
    text: str,
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Synthesize speech and write an MP3 file. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(speak(text, **kwargs))
    return out


def credentials_ready() -> bool:
    """True if a credentials path is set or ADC is likely available."""
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path and Path(path).is_file():
        return True
    # ADC file used by `gcloud auth application-default login`
    adc = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    if adc.is_file():
        return True
    adc_unix = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return adc_unix.is_file()


# Quick local check:
#   python -c "from app.Viedo_generation.tts import speak_to_file; print(speak_to_file('Hello from ACOT video.', r'C:\\acot_video_output\\tts_test.mp3'))"
