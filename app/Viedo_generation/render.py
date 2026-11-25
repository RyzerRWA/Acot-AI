"""
Render a property explainer MP4 from a pack + script.

- Prepares real photos and fixed caption overlays with Pillow
- Animates photos and dissolves with FFmpeg (imageio-ffmpeg or system ffmpeg)
- Overlays an animated ACOT presenter selected from the narration voice
- Measured WAV narration and caption cues from narration.py

Local output default: C:\\acot_video_output\\ (outside OneDrive)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

# Default: vertical 9:16 (phone / Reels / Stories). Override with VIDEO_WIDTH / VIDEO_HEIGHT.
# Landscape 16:9 example: VIDEO_WIDTH=1280 VIDEO_HEIGHT=720
WIDTH = int(os.getenv("VIDEO_WIDTH") or 1080)
HEIGHT = int(os.getenv("VIDEO_HEIGHT") or 1920)
DEFAULT_OUT_DIR = Path(os.getenv("VIDEO_OUTPUT_DIR") or r"C:\acot_video_output")
DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "acot-logo.png"
LOGO_PATH = Path(os.getenv("VIDEO_LOGO_PATH") or DEFAULT_LOGO_PATH)


class RenderError(RuntimeError):
    """Raised when the video cannot be rendered."""


def render(
    pack: dict[str, Any],
    script: dict[str, Any],
    *,
    audio_path: str | Path | None = None,
    output_path: str | Path | None = None,
    job_id: str = "preview",
    presenter: str | None = None,
) -> Path:
    """
    Build an MP4 from pack + script.

    Args:
      pack / script: outputs of build_pack / build_script (both ok=True)
      audio_path: timed narration WAV (legacy audio inputs also supported); if missing, silent video
      output_path: where to write the mp4; default VIDEO_OUTPUT_DIR / {job_id}.mp4
      job_id: used in the default filename
      presenter: male/female or a Neural2 voice name used to select the character

    Returns:
      Path to the finished MP4
    """
    if not pack or not pack.get("ok"):
        raise RenderError((pack or {}).get("reason") or "Pack is not ready.")
    if not script or not script.get("ok"):
        raise RenderError((script or {}).get("reason") or "Script is not ready.")

    slides = script.get("slides") or []
    if not slides:
        raise RenderError("Script has no slides.")

    out = Path(output_path) if output_path else DEFAULT_OUT_DIR / f"{job_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    audio = Path(audio_path) if audio_path else None
    if audio and not audio.is_file():
        raise RenderError(f"Audio file not found: {audio}")

    ffmpeg = _ffmpeg_bin()
    total_seconds = _total_seconds(script, audio)
    slide_durations = _slide_durations(slides, total_seconds)

    work = Path(tempfile.mkdtemp(prefix="acot_video_"))
    try:
        from app.Viedo_generation.motion import render_slides

        clips = render_slides(slides, pack, work, slide_durations, ffmpeg, presenter=presenter)

        # Every clip already has an exact frame count and identical encoding.
        list_file = work / "frames.txt"
        with list_file.open("w", encoding="utf-8") as fh:
            for path in clips:
                # FFmpeg concat needs forward slashes on Windows too.
                safe = path.resolve().as_posix().replace("'", r"'\''")
                fh.write(f"file '{safe}'\n")

        # Stream-copy the 25fps clips; avoid another lossy encode at this stage.
        silent = work / "silent.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(silent),
        ]
        _run(cmd, "FFmpeg slide stitch failed")

        if audio:
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(silent),
                "-i",
                str(audio),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out),
            ]
            _run(cmd, "FFmpeg audio merge failed")
        else:
            shutil.copyfile(silent, out)

        if not out.is_file() or out.stat().st_size < 1000:
            raise RenderError("Render finished but the MP4 is missing or too small.")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _total_seconds(script: dict[str, Any], audio: Path | None) -> float:
    if audio:
        if audio.suffix.lower() == ".wav":
            import wave
            with wave.open(str(audio), "rb") as source:
                return source.getnframes() / source.getframerate()
        # Prefer audio length when present.
        try:
            import imageio_ffmpeg

            probe = subprocess.run(
                [
                    _ffmpeg_bin(),
                    "-i",
                    str(audio),
                ],
                capture_output=True,
                text=True,
            )
            # ffmpeg prints duration on stderr: Duration: 00:01:05.12
            for line in (probe.stderr or "").splitlines():
                if "Duration:" in line:
                    part = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = part.split(":")
                    return max(8.0, float(h) * 3600 + float(m) * 60 + float(s))
        except Exception:
            pass
    estimated = float(script.get("estimated_seconds") or 60)
    return max(20.0, estimated)


def _draw_all_slides(
    slides: list[dict[str, Any]],
    pack: dict[str, Any],
    work: Path,
    durations: list[float],
) -> list[tuple[Path, float]]:
    frames: list[tuple[Path, float]] = []
    projects = pack.get("projects") or []

    for i, (slide, duration) in enumerate(zip(slides, durations)):
        # For project slides, cycle photos across the slide time.
        urls = list(slide.get("photo_urls") or [])
        if not urls and slide.get("kind") == "project":
            idx = slide.get("project_index")
            if isinstance(idx, int) and 0 <= idx < len(projects):
                urls = list(projects[idx].get("photos") or [])

        if not urls:
            # Solid background slide.
            path = work / f"slide_{i:02d}_0.png"
            _draw_slide(path, None, slide)
            frames.append((path, max(2.0, duration)))
            continue

        # Split this slide's time across up to 3 photos.
        photos = urls[:3]
        each = max(2.0, duration / len(photos))
        for j, url in enumerate(photos):
            path = work / f"slide_{i:02d}_{j}.png"
            image = _download_image(url, work / f"src_{i:02d}_{j}")
            _draw_slide(path, image, slide)
            frames.append((path, each))

    return frames


def _slide_durations(
    slides: list[dict[str, Any]], total_seconds: float
) -> list[float]:
    if any("duration_frames" in slide for slide in slides):
        frames = [slide.get("duration_frames") for slide in slides]
        if not all(isinstance(value, int) and value > 0 for value in frames):
            raise RenderError("Narration timings are incomplete. Generate narration again.")
        durations = [value / 25 for value in frames]
        if abs(sum(durations) - total_seconds) > 0.04:
            raise RenderError("Narration audio does not match the video timings. Generate narration again.")
        return durations
    # Weight by spoken words so project boundaries follow the narration more closely.
    # This is an estimate; Google TTS does not supply per-slide timestamps here.
    weights: list[float] = []
    for slide in slides:
        words = len(str(slide.get("narration") or "").split())
        if words:
            weights.append(float(words))
            continue
        kind = slide.get("kind")
        if kind == "intro":
            weights.append(1.0)
        elif kind == "outro":
            weights.append(1.2)
        else:
            weights.append(3.0)
    total_w = sum(weights) or 1.0
    # Quantize cumulative boundaries, avoiding rounding drift across many clips.
    frame_count = max(len(slides), round(total_seconds * 25))
    boundaries = [0]
    cumulative = 0.0
    for index, weight in enumerate(weights[:-1]):
        cumulative += weight
        boundary = round(frame_count * cumulative / total_w)
        boundaries.append(max(boundaries[-1] + 1, min(boundary, frame_count - len(slides) + index + 1)))
    boundaries.append(frame_count)
    return [(end - start) / 25 for start, end in zip(boundaries, boundaries[1:])]


def _draw_slide(out_path: Path, photo_path: Path | None, slide: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (42, 33, 28))
    if photo_path and photo_path.is_file():
        try:
            photo = Image.open(photo_path).convert("RGB")
            canvas.paste(_cover(photo, WIDTH, HEIGHT), (0, 0))
        except Exception:
            pass

    # Darker lower band for captions (taller on 9:16).
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fade_start = int(HEIGHT * 0.48)
    for y in range(fade_start, HEIGHT):
        t = (y - fade_start) / max(1, HEIGHT - fade_start)
        alpha = int(30 + 200 * t)
        draw.line([(0, y), (WIDTH, y)], fill=(20, 14, 12, min(230, alpha)))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    canvas = _add_logo(canvas)
    draw = ImageDraw.Draw(canvas)

    # Scale type to frame height so 9:16 and 16:9 both read well.
    scale = HEIGHT / 1920
    title_font = _font(max(44, int(68 * scale)), bold=True)
    line_font = _font(max(34, int(44 * scale)), bold=False)
    small_font = _font(max(32, int(38 * scale)), bold=False)
    pill_font = _font(max(34, int(40 * scale)), bold=True)
    title_step = max(54, int(78 * scale))
    line_step = max(44, int(54 * scale))
    small_step = max(42, int(50 * scale))
    margin = max(40, int(64 * scale))
    bottom_pad = max(56, int(88 * scale))

    title = str(slide.get("title") or "")
    subtitle = str(slide.get("subtitle") or "")
    lines = [str(x) for x in (slide.get("lines") or []) if x]
    text_width = WIDTH - (margin * 2)
    title_lines = _wrap_to_width(draw, title, title_font, text_width)
    subtitle_lines = _wrap_to_width(draw, subtitle, line_font, text_width)
    body_lines: list[str] = []
    for line in lines:
        if line == title:
            continue
        body_lines.extend(_wrap_to_width(draw, line, small_font, text_width))
    show_subtitle = bool(subtitle) and subtitle not in lines and subtitle != title

    block_h = title_step * len(title_lines)
    if show_subtitle:
        block_h += line_step * len(subtitle_lines)
    block_h += small_step * len(body_lines)
    y = max(int(HEIGHT * 0.55), HEIGHT - bottom_pad - block_h)

    for line in title_lines:
        _text(draw, line, margin, y, title_font, fill=(255, 255, 255))
        y += title_step
    if show_subtitle:
        for line in subtitle_lines:
            _text(draw, line, margin, y, line_font, fill=(240, 220, 210))
            y += line_step
    for line in body_lines:
        _text(draw, line, margin, y, small_font, fill=(235, 230, 225))
        y += small_step

    # Kind pill at top-right, leaving the top-left clear for the ACOT logo.
    kind = str(slide.get("kind") or "")
    if kind:
        label = {"intro": "Overview", "project": "Project", "outro": "Summary"}.get(kind, kind)
        bbox = draw.textbbox((0, 0), label, font=pill_font)
        pw, ph = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x = max(18, int(24 * scale))
        pad_y = max(10, int(14 * scale))
        top = max(36, int(48 * scale))
        left = WIDTH - margin - pw - (pad_x * 2)
        draw.rounded_rectangle(
            (left + 4, top + 6, left + pw + (pad_x * 2) + 4, top + ph + (pad_y * 2) + 6),
            radius=max(16, int(20 * scale)),
            fill=(0, 0, 0),
        )
        draw.rounded_rectangle(
            (left, top, left + pw + (pad_x * 2), top + ph + (pad_y * 2)),
            radius=max(16, int(20 * scale)),
            fill=(133, 92, 72),
        )
        draw.text(
            (left + pad_x, top + pad_y - bbox[1]),
            label,
            font=pill_font,
            fill=(255, 255, 255),
        )

    canvas.save(out_path, format="PNG", compress_level=4)


def _add_logo(canvas: Any) -> Any:
    """Place the ACOT logo in a readable badge at the top-left of every frame."""
    from PIL import Image, ImageDraw

    if not LOGO_PATH.is_file():
        return canvas

    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        target_width = max(120, int(WIDTH * 0.18))
        target_height = max(1, round(target_width * logo.height / logo.width))
        logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)

        pad_x = max(14, round(target_width * 0.10))
        pad_y = max(10, round(target_height * 0.18))
        margin = max(28, round(min(WIDTH, HEIGHT) * 0.04))
        badge_width = target_width + (pad_x * 2)
        badge_height = target_height + (pad_y * 2)
        radius = max(14, round(badge_height * 0.18))

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            (
                margin + 5,
                margin + 7,
                margin + badge_width + 5,
                margin + badge_height + 7,
            ),
            radius=radius,
            fill=(0, 0, 0, 70),
        )
        draw.rounded_rectangle(
            (margin, margin, margin + badge_width, margin + badge_height),
            radius=radius,
            fill=(255, 255, 255, 225),
        )
        overlay.alpha_composite(logo, (margin + pad_x, margin + pad_y))
        return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert(canvas.mode)
    except Exception:
        return canvas


def _wrap_to_width(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    rows: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            rows.append(current)
            current = word
    rows.append(current)
    return rows


def _cover(image: Any, width: int, height: int) -> Any:
    """Scale/crop image to fill width x height."""
    from PIL import Image, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image)
    iw, ih = image.size
    scale = max(width / iw, height / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = image.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    if scale < 1:
        resized = resized.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
    left = (nw - width) // 2
    top = (nh - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    if bold:
        candidates += [
            fr"{windir}\Fonts\segoeuib.ttf",
            fr"{windir}\Fonts\arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates += [
            fr"{windir}\Fonts\segoeui.ttf",
            fr"{windir}\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text(draw: Any, text: str, x: int, y: int, font: Any, fill: tuple[int, int, int]) -> None:
    # Soft shadow for readability on bright photos.
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def _download_image(url: str, dest_base: Path) -> Path | None:
    try:
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200 or not resp.content:
                return None
            ctype = (resp.headers.get("content-type") or "").lower()
            ext = ".jpg"
            if "png" in ctype or url.lower().endswith(".png"):
                ext = ".png"
            elif "webp" in ctype or url.lower().endswith(".webp"):
                ext = ".webp"
            path = Path(str(dest_base) + ext)
            path.write_bytes(resp.content)
            return path
    except Exception:
        return None


def _ffmpeg_bin() -> str:
    env = os.getenv("FFMPEG_PATH", "").strip()
    if env and Path(env).is_file():
        return env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    which = shutil.which("ffmpeg")
    if which:
        return which
    raise RenderError(
        "FFmpeg not found. Install imageio-ffmpeg or set FFMPEG_PATH."
    )


def _run(cmd: list[str], error_prefix: str) -> None:
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()[-1200:]
        raise RenderError(f"{error_prefix}: {detail}")


# Quick local check (silent, no TTS needed):
#   python -c "from app.Viedo_generation.history import searched_projects; from app.Viedo_generation.pack import build_pack; from app.Viedo_generation.script import build_script; from app.Viedo_generation.render import render; p=build_pack(searched_projects('USER')); s=build_script(p); print(render(p,s, job_id='silent_test'))"
