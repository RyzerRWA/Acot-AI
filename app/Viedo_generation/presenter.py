"""Build a small transparent ACOT presenter animation for video overlays."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "presenters"
POSE_NAMES = ("idle", "talking", "pointing", "presenting", "positive", "waving")


class PresenterError(RuntimeError):
    """Raised when presenter assets or frames cannot be prepared."""


@dataclass(frozen=True)
class PresenterSequence:
    pattern: str
    frame_rate: int
    x: int
    y: int


def presenter_enabled() -> bool:
    raw = (os.getenv("VIDEO_PRESENTER_ENABLED") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def resolve_gender(voice: str | None) -> str:
    """Map male/female or the configured Neural2 voice to a presenter asset set."""
    raw = (
        voice
        or os.getenv("GOOGLE_TTS_GENDER")
        or os.getenv("VIDEO_PRESENTER_GENDER")
        or "male"
    ).strip().lower()
    if raw == "female" or raw.endswith("-f"):
        return "female"
    if raw == "male" or raw.endswith("-d"):
        return "male"
    fallback = (os.getenv("VIDEO_PRESENTER_GENDER") or "male").strip().lower()
    return "female" if fallback == "female" else "male"


def build_presenter_sequence(
    slides: list[dict[str, Any]],
    durations: list[float],
    work: Path,
    *,
    voice: str | None,
    video_width: int,
    video_height: int,
) -> PresenterSequence:
    """Render small transparent PNG frames for FFmpeg to overlay on the slideshow."""
    from PIL import Image

    if not slides or len(slides) != len(durations):
        raise PresenterError("Presenter timeline does not match the video slides.")

    gender = resolve_gender(voice)
    frame_rate = _bounded_int("VIDEO_PRESENTER_FPS", default=10, low=6, high=15)
    x, y, overlay_width, overlay_height = presenter_bounds(video_width, video_height)
    poses = _load_poses(gender, overlay_width, overlay_height)

    frame_dir = work / "presenter_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    total_seconds = sum(max(0.0, float(value)) for value in durations)
    frame_count = max(1, math.ceil(total_seconds * frame_rate))

    slide_index = 0
    slide_start = 0.0
    for frame_index in range(frame_count):
        timestamp = min(total_seconds, (frame_index + 0.5) / frame_rate)
        while (
            slide_index < len(durations) - 1
            and timestamp >= slide_start + durations[slide_index]
        ):
            slide_start += durations[slide_index]
            slide_index += 1

        slide = slides[slide_index]
        duration = max(0.1, durations[slide_index])
        local_time = max(0.0, timestamp - slide_start)
        pose_name = _pose_for_slide(slide, local_time, duration)
        sprite = poses[pose_name]

        if pose_name == "waving":
            angle = math.sin(local_time * math.tau * 1.4) * 1.8
            sprite = sprite.rotate(angle, resample=Image.Resampling.BICUBIC)

        bob = round(math.sin(local_time * math.tau / 1.8) * 4)
        shift_x = _entrance_exit_shift(slide, local_time, duration, overlay_width)
        frame = Image.new("RGBA", (overlay_width, overlay_height), (0, 0, 0, 0))
        frame.alpha_composite(sprite, (shift_x, bob))
        frame.save(
            frame_dir / f"presenter_{frame_index:05d}.png",
            format="PNG",
            compress_level=2,
        )

    return PresenterSequence(
        pattern=str(frame_dir / "presenter_%05d.png"),
        frame_rate=frame_rate,
        x=x,
        y=y,
    )


def presenter_bounds(video_width: int, video_height: int) -> tuple[int, int, int, int]:
    """Reserve the lower-right information panel for the agent and its gestures."""
    width_ratio = _bounded_float("VIDEO_PRESENTER_WIDTH_RATIO", default=0.27, low=0.18, high=0.34)
    y_ratio = _bounded_float("VIDEO_PRESENTER_Y_RATIO", default=0.75, low=0.70, high=0.80)
    height = min(round(video_width * width_ratio * 1.5), round(video_height * 0.25))
    width = round(height / 1.5)
    margin = max(8, round(video_width * 0.025))
    return video_width - width - margin, min(video_height - height - margin, round(video_height * y_ratio)), width, height


def _load_poses(gender: str, width: int, height: int) -> dict[str, Any]:
    from PIL import Image

    asset_dir = ASSET_ROOT / gender
    poses: dict[str, Any] = {}
    for name in POSE_NAMES:
        path = asset_dir / f"{name}.png"
        if not path.is_file():
            raise PresenterError(f"Presenter asset is missing: {path}")
        try:
            image = Image.open(path).convert("RGBA")
            alpha = image.getchannel("A").point(lambda value: 0 if value < 12 else value)
            image.putalpha(alpha)
            poses[name] = image.resize(
                (width, height),
                resample=Image.Resampling.LANCZOS,
            )
        except Exception as exc:
            raise PresenterError(f"Could not load presenter asset {path}: {exc}") from exc
    return poses


def _pose_for_slide(slide: dict[str, Any], local_time: float, duration: float) -> str:
    kind = str(slide.get("kind") or "").lower()
    progress = min(1.0, local_time / duration)

    if kind == "intro":
        if progress < 0.32:
            return "waving"
        return _talking_pose(local_time)
    if kind == "outro":
        return "positive" if progress < 0.55 else "waving"
    if kind == "project":
        if progress < 0.18:
            return "presenting"
        if progress < 0.55:
            return "pointing"
        return _talking_pose(local_time)
    return "idle"


def _talking_pose(local_time: float) -> str:
    return "talking" if int(local_time * 3.0) % 2 == 0 else "idle"


def _entrance_exit_shift(
    slide: dict[str, Any],
    local_time: float,
    duration: float,
    overlay_width: int,
) -> int:
    kind = str(slide.get("kind") or "").lower()
    if kind == "intro" and local_time < 0.8:
        progress = max(0.0, min(1.0, local_time / 0.8))
        eased = 1.0 - (1.0 - progress) ** 3
        return round((1.0 - eased) * overlay_width)
    if kind == "outro" and duration - local_time < 0.7:
        progress = max(0.0, min(1.0, (0.7 - (duration - local_time)) / 0.7))
        return round((progress**2) * overlay_width)
    return 0


def _bounded_int(name: str, *, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name) or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


def _bounded_float(name: str, *, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name) or default)
    except ValueError:
        value = default
    return max(low, min(high, value))
