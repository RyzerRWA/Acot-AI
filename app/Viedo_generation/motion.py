"""Photo motion and dissolves using Pillow and FFmpeg, with no generated imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

FPS = 25
TRANSITION_FRAMES = 15
MIN_PHOTO_FRAMES = 6 * FPS


def shot_timing(total_frames: int, count: int) -> tuple[list[int], list[int]]:
    """Include dissolve handles without shortening the narrated slide."""
    overlap = min(TRANSITION_FRAMES, max(0, total_frames // (count * 3)))
    frames = total_frames + overlap * (count - 1)
    lengths = [frames // count + (i < frames % count) for i in range(count)]
    offsets = []
    elapsed = lengths[0]
    for length in lengths[1:]:
        offsets.append(elapsed - overlap)
        elapsed += length - overlap
    return lengths, offsets


def motion_filter(index: int, frames: int, width: int, height: int) -> str:
    """Push through each shot while drifting toward an off-center focal point."""
    progress = f"min(on/{max(1, frames - 1)},1)"
    eased = f"({progress}*{progress}*(3-2*{progress}))"
    direction = 1 if index % 2 == 0 else -1
    travel = (0.18, 0.14, 0.20, 0.16)[index % 4]
    zoom = f"1.01+{travel}*{eased}"
    x = f"(iw-iw/zoom)*(0.5+({direction * 0.24})*{eased})"
    y = f"(ih-ih/zoom)*(0.5+({0.09 if index % 3 else -0.09})*{eased})"
    return (
        f"zoompan=z='{zoom}':x='{x}':y='{y}':"
        f"d={frames}:s={width}x{height}:fps={FPS},"
        f"setsar=1,format=yuv420p,settb=1/{FPS},setpts=PTS-STARTPTS"
    )


def _prepare_photo(source: Path, target: Path, width: int, height: int) -> None:
    from PIL import Image, ImageFilter, ImageOps
    from app.Viedo_generation.render import _cover

    with Image.open(source) as opened:
        photo = ImageOps.exif_transpose(opened).convert("RGB")
    # Keep the complete room/building visible. A soft extension fills the space
    # around landscape photos on a portrait canvas, rather than a severe crop.
    size = (width * 2, height * 2)
    background = _cover(photo, width, height).filter(ImageFilter.GaussianBlur(24))
    background = background.resize(size, Image.Resampling.LANCZOS)
    foreground = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    background.paste(foreground, ((size[0] - foreground.width) // 2, (size[1] - foreground.height) // 2))
    background.save(target, compress_level=2)


def caption_schedule(slide: dict[str, Any], total_frames: int) -> list[tuple[list[str], int, int]]:
    """Captions follow measured speech cues, never photo-change boundaries."""
    from app.Viedo_generation.render import RenderError

    title, subtitle = slide.get("title"), slide.get("subtitle")
    def clean(lines: list[Any]) -> list[str]:
        return [str(line) for line in lines if line and line not in (title, subtitle)]

    if "duration_frames" not in slide:
        # Silent/legacy input has no measured speech. Keep facts visible rather
        # than implying that estimated caption changes are synchronized.
        return [(clean(slide.get("lines", [])), 0, total_frames)]
    result = []
    cursor = 0
    for cue in slide.get("cues", []):
        start, end = cue.get("start_frame"), cue.get("end_frame")
        if start != cursor or not isinstance(end, int) or end <= cursor:
            raise RenderError("Caption timings contain a gap or overlap.")
        result.append((clean(cue.get("lines", [])), start, end))
        cursor = end
    if cursor != total_frames:
        raise RenderError("Caption timings do not cover the narrated slide.")
    return result


def draw_overlay(path: Path, slide: dict[str, Any], lines: list[str], width: int, height: int) -> None:
    from PIL import Image, ImageDraw
    from app.Viedo_generation.render import _add_logo, _font, _wrap_to_width
    from app.Viedo_generation.presenter import presenter_enabled, presenter_bounds

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    top = round(height * 0.68)
    # Dedicated information panel: the photos, captions and agent do not overlap.
    draw.rectangle((0, top, width, height), fill=(29, 23, 20, 255))
    scale = min(width / 1080, height / 1920)
    margin = max(12, round(48 * scale))
    draw.rectangle((margin, top + round(26 * scale), margin + round(90 * scale), top + round(31 * scale)), fill=(183, 143, 116, 255))
    text_right = width - margin
    if presenter_enabled():
        agent_x, _, _, _ = presenter_bounds(width, height)
        text_right = agent_x - margin // 2
    available = max(80, text_right - margin)
    y_start = top + round(48 * scale)
    bottom = height - margin
    # Fit complete captions, including long community/project names.
    rows = []
    for factor in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        rows = []
        for text, size, bold, colour in [
            (str(slide.get("title") or ""), 64, True, (255, 255, 255, 255)),
            (str(slide.get("subtitle") or ""), 40, False, (205, 173, 151, 255)),
            *[(line, 38, False, (243, 238, 232, 255)) for line in lines],
        ]:
            font = _font(max(12, round(size * scale * factor)), bold=bold)
            step = round(size * scale * factor * 1.3)
            rows.extend((row, font, step, colour) for row in _wrap_to_width(draw, text, font, available))
        if y_start + sum(row[2] for row in rows) <= bottom:
            break
    y = y_start
    for text, font, step, colour in rows:
        draw.text((margin, y), text, font=font, fill=colour)
        y += step
    label = {"intro": "Overview", "outro": "Summary"}.get(slide.get("kind"), "Project")
    font = _font(max(12, round(40 * scale)), bold=True)
    box = draw.textbbox((0, 0), label, font=font)
    pad = max(8, round(22 * scale))
    pill_w, pill_h = box[2] - box[0] + 2 * pad, box[3] - box[1] + pad
    x = width - margin - pill_w
    draw.rounded_rectangle((x, margin, x + pill_w, margin + pill_h), radius=max(8, round(18 * scale)), fill=(133, 92, 72, 245))
    draw.text((x + pad, margin + pad // 2 - box[1]), label, font=font, fill="white")
    canvas = _add_logo(canvas)
    canvas.save(path, compress_level=2)


def render_slides(slides: list[dict[str, Any]], pack: dict[str, Any], work: Path,
                  durations: list[float], ffmpeg: str, *, presenter: str | None = None) -> list[Path]:
    from app.Viedo_generation.render import WIDTH, HEIGHT, RenderError, _download_image, _run
    from app.Viedo_generation.presenter import build_presenter_sequence, presenter_enabled

    photo_height = round(HEIGHT * 0.68) // 2 * 2
    cache: dict[str, Path | None] = {}
    clips = []
    for slide_index, (slide, duration) in enumerate(zip(slides, durations)):
        urls = list(dict.fromkeys(slide.get("photo_urls") or []))
        project_index = slide.get("project_index")
        if not urls and isinstance(project_index, int) and 0 <= project_index < len(pack.get("projects", [])):
            urls = list(dict.fromkeys(pack["projects"][project_index].get("photos") or []))
        sources = []
        for url in urls[:4]:
            if url not in cache:
                source = _download_image(url, work / f"download_{len(cache)}")
                target = work / f"photo_{len(cache)}.png"
                try:
                    if source:
                        _prepare_photo(source, target, WIDTH, photo_height)
                    cache[url] = target if source else None
                except (OSError, ValueError):
                    cache[url] = None
            if cache[url]:
                sources.append(cache[url])
        if not sources:
            raise RenderError(f"No usable property photos for {slide.get('title') or 'slide'}. Please check the image URLs.")
        total_frames = max(1, round(duration * FPS))
        # Allow at least six seconds per photo. Caption timing is independent.
        sources = sources[:max(1, min(len(sources), total_frames // MIN_PHOTO_FRAMES))]
        lengths, offsets = shot_timing(total_frames, len(sources))
        overlap = (sum(lengths) - total_frames) // max(1, len(sources) - 1)
        cmd = [ffmpeg, "-y", "-filter_complex_threads", "2"]
        for source in sources:
            cmd.extend(["-i", str(source)])
        pages = caption_schedule(slide, total_frames)
        for i, (page, _, _) in enumerate(pages):
            overlay = work / f"overlay_{slide_index}_{i}.png"
            draw_overlay(overlay, slide, page, WIDTH, HEIGHT)
            cmd.extend(["-i", str(overlay)])
        filters = [f"[{i}:v]{motion_filter(i + slide_index, length, WIDTH, photo_height)}[p{i}]" for i, length in enumerate(lengths)]
        previous = "p0"
        for i, offset in enumerate(offsets, 1):
            filters.append(f"[{previous}][p{i}]xfade=transition=fade:duration={overlap / FPS:.6f}:offset={offset / FPS:.6f}[mix{i}]")
            previous = f"mix{i}"
        filters.append(f"[{previous}]pad={WIDTH}:{HEIGHT}:0:0:color=0x1d1714[base]")
        previous = "base"
        for i, (_, start, end) in enumerate(pages):
            filters.append(f"[{previous}][{len(sources) + i}:v]overlay=0:0:enable='gte(n,{start})*lt(n,{end})':format=yuv420[card{i}]")
            previous = f"card{i}"
        if presenter_enabled():
            agent = build_presenter_sequence(
                [slide], [duration], work / f"agent_{slide_index}",
                voice=presenter, video_width=WIDTH, video_height=HEIGHT,
            )
            cmd.extend(["-framerate", str(agent.frame_rate), "-start_number", "0", "-i", agent.pattern])
            # Framesync holds the most recent sprite until the next 10fps frame.
            # The photo stream determines the clip length, including the last frame.
            filters.append(
                f"[{previous}][{len(sources) + len(pages)}:v]"
                f"overlay=x={agent.x}:y={agent.y}:format=yuv420[presented]"
            )
            previous = "presented"
        target = work / f"clip_{slide_index:02d}.mp4"
        cmd.extend(["-filter_complex", ";".join(filters), "-map", f"[{previous}]", "-frames:v", str(total_frames),
                    "-an", "-r", str(FPS), "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                    "-crf", "23", "-maxrate", "1600k", "-bufsize", "3200k",
                    "-pix_fmt", "yuv420p", "-video_track_timescale", "12800", str(target)])
        _run(cmd, f"Photo motion render failed for {slide.get('title')}")
        clips.append(target)
    return clips
