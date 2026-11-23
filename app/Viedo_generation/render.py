"""
Render a property explainer MP4 from a pack + script.

- Draws each slide with Pillow (photo + captions)
- Stitches slides with FFmpeg (imageio-ffmpeg or system ffmpeg)
- Optional MP3 narration from tts.py

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


class RenderError(RuntimeError):
    """Raised when the video cannot be rendered."""


def render(
    pack: dict[str, Any],
    script: dict[str, Any],
    *,
    audio_path: str | Path | None = None,
    output_path: str | Path | None = None,
    job_id: str = "preview",
) -> Path:
    """
    Build an MP4 from pack + script.

    Args:
      pack / script: outputs of build_pack / build_script (both ok=True)
      audio_path: optional MP3 from speak_to_file; if missing, silent video
      output_path: where to write the mp4; default VIDEO_OUTPUT_DIR / {job_id}.mp4
      job_id: used in the default filename

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

    work = Path(tempfile.mkdtemp(prefix="acot_video_"))
    try:
        frame_paths = _draw_all_slides(slides, pack, work, total_seconds)
        if not frame_paths:
            raise RenderError("No slide frames were drawn.")

        # concat demuxer list: file '...' + duration
        list_file = work / "frames.txt"
        with list_file.open("w", encoding="utf-8") as fh:
            for path, duration in frame_paths:
                # FFmpeg concat needs forward slashes on Windows too.
                safe = path.resolve().as_posix().replace("'", r"'\''")
                fh.write(f"file '{safe}'\n")
                fh.write(f"duration {duration:.3f}\n")
            # Repeat last file so the final duration is honoured.
            last = frame_paths[-1][0].resolve().as_posix().replace("'", r"'\''")
            fh.write(f"file '{last}'\n")

        # CFR at 25fps — VFR (~0.17fps) looks "empty"/black in many Windows players.
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
            "-vf",
            "fps=25,format=yuv420p",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
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
    return max(20.0, min(90.0, estimated))


def _draw_all_slides(
    slides: list[dict[str, Any]],
    pack: dict[str, Any],
    work: Path,
    total_seconds: float,
) -> list[tuple[Path, float]]:
    # Weight: intro/outro shorter, project slides share the rest.
    weights: list[float] = []
    for slide in slides:
        kind = slide.get("kind")
        if kind == "intro":
            weights.append(1.0)
        elif kind == "outro":
            weights.append(1.2)
        else:
            weights.append(3.0)
    total_w = sum(weights) or 1.0
    durations = [total_seconds * (w / total_w) for w in weights]

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
            path = work / f"slide_{i:02d}_0.jpg"
            _draw_slide(path, None, slide)
            frames.append((path, max(2.0, duration)))
            continue

        # Split this slide's time across up to 3 photos.
        photos = urls[:3]
        each = max(2.0, duration / len(photos))
        for j, url in enumerate(photos):
            path = work / f"slide_{i:02d}_{j}.jpg"
            image = _download_image(url, work / f"src_{i:02d}_{j}")
            _draw_slide(path, image, slide)
            frames.append((path, each))

    return frames


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
    draw = ImageDraw.Draw(canvas)

    # Scale type to frame height so 9:16 and 16:9 both read well.
    scale = HEIGHT / 1920
    title_font = _font(max(36, int(52 * scale)), bold=True)
    line_font = _font(max(26, int(34 * scale)), bold=False)
    small_font = _font(max(22, int(28 * scale)), bold=False)
    title_step = max(40, int(58 * scale))
    line_step = max(30, int(40 * scale))
    small_step = max(28, int(36 * scale))
    margin = max(36, int(56 * scale))
    bottom_pad = max(48, int(72 * scale))

    title = str(slide.get("title") or "")
    subtitle = str(slide.get("subtitle") or "")
    lines = [str(x) for x in (slide.get("lines") or []) if x]
    # Soft wrap long caption lines to the frame width.
    max_chars = max(22, WIDTH // max(14, int(18 * scale)))
    body_lines: list[str] = []
    for line in lines:
        if line == title:
            continue
        body_lines.extend(_wrap_text(line, max_chars))
    show_subtitle = bool(subtitle) and subtitle not in lines and subtitle != title

    block_h = 0
    if title:
        block_h += title_step
    if show_subtitle:
        block_h += line_step
    block_h += small_step * len(body_lines)
    y = max(int(HEIGHT * 0.55), HEIGHT - bottom_pad - block_h)

    if title:
        _text(draw, title, margin, y, title_font, fill=(255, 255, 255))
        y += title_step
    if show_subtitle:
        _text(draw, subtitle, margin, y, line_font, fill=(240, 220, 210))
        y += line_step
    for line in body_lines:
        _text(draw, line, margin, y, small_font, fill=(235, 230, 225))
        y += small_step

    # Kind pill
    kind = str(slide.get("kind") or "")
    if kind:
        label = {"intro": "Overview", "project": "Project", "outro": "Summary"}.get(kind, kind)
        pill = f"  {label}  "
        bbox = draw.textbbox((0, 0), pill, font=small_font)
        pw, ph = bbox[2] - bbox[0], bbox[3] - bbox[1]
        top = max(36, int(48 * scale))
        draw.rounded_rectangle(
            (margin, top, margin + pw + 8, top + ph + 10),
            radius=14,
            fill=(133, 92, 72),
        )
        draw.text((margin + 4, top + 4), pill, font=small_font, fill=(255, 255, 255))

    canvas.save(out_path, format="JPEG", quality=90)


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    rows: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if len(trial) <= max_chars:
            current = trial
        else:
            rows.append(current)
            current = word
    rows.append(current)
    return rows


def _cover(image: Any, width: int, height: int) -> Any:
    """Scale/crop image to fill width x height."""
    iw, ih = image.size
    scale = max(width / iw, height / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = image.resize((nw, nh))
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
