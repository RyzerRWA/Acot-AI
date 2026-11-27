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
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

# Default: landscape 16:9, full-frame shots. Override with VIDEO_WIDTH / VIDEO_HEIGHT
# (a portrait size such as 1080x1920 switches back to the photo-over-panel layout).
WIDTH = int(os.getenv("VIDEO_WIDTH") or 1920)
HEIGHT = int(os.getenv("VIDEO_HEIGHT") or 1080)
from app.Viedo_generation.storage import DEFAULT_OUT_DIR
DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "acot-logo.png"
LOGO_PATH = Path(os.getenv("VIDEO_LOGO_PATH") or DEFAULT_LOGO_PATH)
LOGO_OPACITY = 0.7
REEL_SIZE = (1080, 1920)
AGENT_PANEL_SECONDS = 5


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
    broker_card: dict[str, Any] | None = None,
) -> Path:
    """
    Build an MP4 from pack + script.

    Args:
      pack / script: outputs of build_pack / build_script (both ok=True)
      audio_path: timed narration WAV (legacy audio inputs also supported); if missing, silent video
      output_path: where to write the mp4; default VIDEO_OUTPUT_DIR / {job_id}.mp4
      job_id: used in the default filename
      presenter: male/female or a Neural2 voice name used to select the character
      broker_card: agent details shown for the last five seconds of a portrait Reel

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

        frame_size = REEL_SIZE if broker_card else (WIDTH, HEIGHT)
        clips = render_slides(
            slides, pack, work, slide_durations, ffmpeg, frame_size=frame_size, presenter=presenter,
            broker_card=broker_card,
        )

        # Each clip but the last carries a dissolve handle. Offsets land on the narration boundaries.
        from app.Viedo_generation.motion import FPS as MOTION_FPS, SLIDE_FADE_FRAMES

        silent = work / "silent.mp4"
        cmd = [ffmpeg, "-y", "-filter_complex_threads", "2"]
        for path in clips:
            cmd.extend(["-i", str(path)])
        filters, previous, elapsed = [], "0:v", 0
        if broker_card and len(clips) > 1:
            inputs = "".join(f"[{i}:v]" for i in range(len(clips)))
            filters.append(f"{inputs}concat=n={len(clips)}:v=1:a=0[film]")
            previous = "film"
        elif not broker_card:
            for i, seconds in enumerate(slide_durations[:-1], 1):
                elapsed += max(1, round(seconds * MOTION_FPS))
                filters.append(
                    f"[{previous}][{i}:v]xfade=transition=fade:duration={SLIDE_FADE_FRAMES / MOTION_FPS:.4f}"
                    f":offset={elapsed / MOTION_FPS:.4f}[x{i}]"
                )
                previous = f"x{i}"
        if filters:
            cmd.extend(["-filter_complex", ";".join(filters), "-map", f"[{previous}]"])
        cmd.extend([
            "-an", "-r", str(MOTION_FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-maxrate", "8000k", "-bufsize", "16000k", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(silent),
        ])
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
                *(["-af", "apad"] if broker_card else []),
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
    if all(isinstance(slide.get("target_frames"), int) and slide["target_frames"] > 0 for slide in slides):
        durations = [slide["target_frames"] / 25 for slide in slides]
        if abs(sum(durations) - total_seconds) <= 0.04:
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
    """Place the transparent ACOT logo at the top-left of every frame."""
    from PIL import Image, ImageFilter

    if not LOGO_PATH.is_file():
        return canvas

    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        # Small and quiet: never upscale the 142px source (that made it blurry), and keep it translucent.
        target_width = min(logo.width, max(64, round(min(canvas.size) * 0.105)))
        target_height = max(1, round(target_width * logo.height / logo.width))
        logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)
        logo.putalpha(logo.getchannel("A").point(lambda a: round(a * LOGO_OPACITY)))

        margin = max(24, round(min(canvas.size) * 0.045))
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow = Image.new("RGBA", logo.size)
        shadow.putalpha(logo.getchannel("A").point(lambda a: a // 3).filter(ImageFilter.GaussianBlur(3)))
        overlay.alpha_composite(shadow, (margin, margin + 1))
        overlay.alpha_composite(logo, (margin, margin))
        return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert(canvas.mode)
    except Exception:
        return canvas


def _broker_photo(url: str) -> Any:
    from PIL import Image, ImageOps

    try:
        source = urlsplit(url)
        if source.scheme != "https" or source.netloc.lower() != "mobile.dubailand.gov.ae":
            return None
        content = bytearray()
        with httpx.stream("GET", url, timeout=4.0, follow_redirects=False) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > 4_000_000:
                    return None
        with Image.open(BytesIO(content)) as photo:
            if photo.width * photo.height <= 12_000_000:
                return ImageOps.exif_transpose(photo).convert("RGB")
    except (httpx.HTTPError, OSError, ValueError, Image.DecompressionBombError):
        pass
    return None


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


def _cover(image: Any, width: int, height: int, *, focus_x: float = 0.5) -> Any:
    """Scale/crop image to fill width x height."""
    from PIL import Image, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image)
    iw, ih = image.size
    scale = max(width / iw, height / ih)
    nw, nh = max(width, int(iw * scale)), max(height, int(ih * scale))
    resized = image.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    if scale < 1:
        resized = resized.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
    left = min(max(0, int(nw * focus_x - width / 2)), nw - width)
    top = (nh - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    windir = os.environ.get("WINDIR", r"C:\Windows")
    # (path, face index). VIDEO_FONT / VIDEO_FONT_BOLD point at a brand font on the VM.
    if bold:
        candidates = [(os.getenv("VIDEO_FONT_BOLD"), 0), (fr"{windir}\Fonts\segoeuisb.ttf", 0),
                      (fr"{windir}\Fonts\segoeuib.ttf", 0), ("/System/Library/Fonts/Avenir Next.ttc", 2),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)]
    else:
        candidates = [(os.getenv("VIDEO_FONT"), 0), (fr"{windir}\Fonts\segoeui.ttf", 0),
                      ("/System/Library/Fonts/Avenir Next.ttc", 5),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)]
    for path, index in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size=size, index=index)
    # Never fall back to the tiny bitmap font silently.
    return ImageFont.load_default(size=size)


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
