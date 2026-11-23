"""
Google Cloud Text-to-Speech for the video narration.

speak(text) -> MP3 bytes
speak_to_file(text, path) -> Path

Auth (pick one, in order):
  1. GOOGLE_CLOUD_API_KEY  — REST call with ?key=...
  2. GOOGLE_APPLICATION_CREDENTIALS — service-account JSON
  3. gcloud auth application-default login

Env:
  GOOGLE_TTS_GENDER   male | female  (default male)
  GOOGLE_TTS_VOICE    exact Neural2 name (overrides gender)
  GOOGLE_TTS_VOICE_MALE / GOOGLE_TTS_VOICE_FEMALE  optional overrides
  GOOGLE_TTS_LANGUAGE default en-US
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Neural2 is clear and reasonably priced for short narrations.
DEFAULT_VOICE_MALE = "en-US-Neural2-D"
DEFAULT_VOICE_FEMALE = "en-US-Neural2-F"
DEFAULT_VOICE = DEFAULT_VOICE_MALE
DEFAULT_LANGUAGE = "en-US"
DEFAULT_GENDER = "male"
# Google TTS hard limit is 5000 bytes of input text per request.
MAX_CHARS = 4500
TTS_REST_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

VOICE_PRESETS = {
    "male": DEFAULT_VOICE_MALE,
    "female": DEFAULT_VOICE_FEMALE,
    "m": DEFAULT_VOICE_MALE,
    "f": DEFAULT_VOICE_FEMALE,
}


class TtsError(RuntimeError):
    """Raised when speech cannot be synthesized."""


def resolve_voice(
    voice: str | None = None,
    *,
    gender: str | None = None,
) -> str:
    """
    Resolve a Google TTS voice name.

    Priority:
      1. Explicit `voice` (full name like en-US-Neural2-F, or male/female)
      2. `gender` argument (male/female)
      3. GOOGLE_TTS_VOICE env (full name)
      4. GOOGLE_TTS_GENDER env (male/female)
      5. Default male Neural2-D
    """
    raw = (voice or "").strip()
    if raw:
        key = raw.lower()
        if key in VOICE_PRESETS:
            return _gender_voice(key)
        return raw

    g = (gender or "").strip().lower()
    if g in VOICE_PRESETS:
        return _gender_voice(g)

    env_voice = (os.getenv("GOOGLE_TTS_VOICE") or "").strip()
    if env_voice:
        key = env_voice.lower()
        if key in VOICE_PRESETS:
            return _gender_voice(key)
        return env_voice

    env_gender = (os.getenv("GOOGLE_TTS_GENDER") or DEFAULT_GENDER).strip().lower()
    if env_gender in VOICE_PRESETS:
        return _gender_voice(env_gender)
    return DEFAULT_VOICE


def _gender_voice(gender_key: str) -> str:
    if gender_key in ("female", "f"):
        return (os.getenv("GOOGLE_TTS_VOICE_FEMALE") or DEFAULT_VOICE_FEMALE).strip()
    return (os.getenv("GOOGLE_TTS_VOICE_MALE") or DEFAULT_VOICE_MALE).strip()


def list_voices() -> dict[str, str]:
    """Built-in gender presets (after env overrides)."""
    return {
        "male": _gender_voice("male"),
        "female": _gender_voice("female"),
    }


def speak(
    text: str,
    *,
    voice: str | None = None,
    gender: str | None = None,
    language: str | None = None,
    speaking_rate: float = 1.0,
) -> bytes:
    """
    Synthesize `text` to MP3 bytes via Google Cloud Text-to-Speech.

    Pass gender=\"male\"|\"female\" or voice=\"en-US-Neural2-F\" (or male/female).

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

    voice_name = resolve_voice(voice, gender=gender)
    language_code = (language or os.getenv("GOOGLE_TTS_LANGUAGE") or DEFAULT_LANGUAGE).strip()
    # Voice names are like en-US-Neural2-D — language should match the prefix.
    if "-" in voice_name:
        prefix = "-".join(voice_name.split("-")[:2])
        if prefix:
            language_code = prefix

    rate = max(0.7, min(1.3, float(speaking_rate)))
    api_key = (
        os.getenv("GOOGLE_CLOUD_API_KEY", "").strip()
        or os.getenv("GOOGLE_TTS_API_KEY", "").strip()
    )
    if api_key:
        return _speak_rest(cleaned, voice_name, language_code, rate, api_key)
    return _speak_client(cleaned, voice_name, language_code, rate)


def _speak_rest(
    text: str,
    voice_name: str,
    language_code: str,
    speaking_rate: float,
    api_key: str,
) -> bytes:
    import httpx

    payload = {
        "input": {"text": text},
        "voice": {"languageCode": language_code, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": speaking_rate,
        },
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(TTS_REST_URL, params={"key": api_key}, json=payload)
    except Exception as exc:
        raise TtsError(f"Google TTS request failed: {exc}") from exc

    if resp.status_code != 200:
        detail = (resp.text or "")[:500]
        raise TtsError(
            f"Google TTS HTTP {resp.status_code}: {detail}. "
            "If this key is Maps-only, enable Cloud Text-to-Speech API "
            "and allow this key to call it (or use a service-account JSON)."
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise TtsError("Google TTS returned non-JSON.") from exc

    encoded = data.get("audioContent")
    if not encoded:
        raise TtsError("Google TTS returned empty audio.")
    try:
        return base64.b64decode(encoded)
    except Exception as exc:
        raise TtsError("Google TTS audioContent was not valid base64.") from exc


def _speak_client(
    text: str,
    voice_name: str,
    language_code: str,
    speaking_rate: float,
) -> bytes:
    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise TtsError(
            "google-cloud-texttospeech is not installed. "
            "Run: python -m pip install google-cloud-texttospeech "
            "(or set GOOGLE_CLOUD_API_KEY to use the REST API instead)."
        ) from exc

    try:
        client = texttospeech.TextToSpeechClient()
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
            ),
        )
    except Exception as exc:
        message = str(exc)
        if "default credentials" in message.lower() or "could not load" in message.lower():
            raise TtsError(
                "Google TTS credentials are missing. Set GOOGLE_CLOUD_API_KEY, "
                "or GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path, "
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
    """True if an API key, credentials path, or ADC file is available."""
    if os.getenv("GOOGLE_CLOUD_API_KEY", "").strip() or os.getenv("GOOGLE_TTS_API_KEY", "").strip():
        return True
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path and Path(path).is_file():
        return True
    adc = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    if adc.is_file():
        return True
    adc_unix = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return adc_unix.is_file()


# Quick local check:
#   python -c "from app.Viedo_generation.tts import speak_to_file; print(speak_to_file('Hello from ACOT video.', r'C:\\acot_video_output\\tts_test.mp3'))"
