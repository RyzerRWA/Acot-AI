"""Photo motion and dissolves using Pillow and FFmpeg, with no generated imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

FPS = 25
TRANSITION_FRAMES = 25
MIN_PHOTO_FRAMES = 6 * FPS
# Each slide (except the last) runs this much longer so the next one can dissolve over it.
SLIDE_FADE_FRAMES = 20
LUMA_FRAMES = 5 * FPS


def shot_timing(total_frames: int, count: int, *, moving: set[int] | None = None) -> tuple[list[int], list[int]]:
    """Include dissolve handles without shortening the narrated slide."""
    fade_frames = 0 if moving is not None else TRANSITION_FRAMES
    overlap = min(fade_frames, max(0, total_frames // (count * 3)))
    frames = total_frames + overlap * (count - 1)
    lengths = [frames // count + (i < frames % count) for i in range(count)]
    if moving and len(moving) < count:
        # Keep five-second camera clips close to native speed. Give any remaining
        # time to photo shots so every narration cue keeps its original position.
        stills = [i for i in range(count) if i not in moving]
        excess = 0
        for i in moving:
            capped = min(lengths[i], 6 * FPS)
            excess += lengths[i] - capped
            lengths[i] = capped
        for remainder_rank, i in enumerate(stills):
            lengths[i] += excess // len(stills) + (remainder_rank < excess % len(stills))
    offsets = []
    elapsed = lengths[0]
    for length in lengths[1:]:
        offsets.append(elapsed - overlap)
        elapsed += length - overlap
    return lengths, offsets


def motion_filter(index: int, frames: int, width: int, height: int, *, cinematic: bool = False) -> str:
    """Push through each shot while drifting toward an off-center focal point."""
    progress = f"min(on/{max(1, frames - 1)},1)"
    eased = f"({progress}*{progress}*(3-2*{progress}))"
    direction = 1 if index % 2 == 0 else -1
    travel = ((0.19, 0.14, 0.20, 0.16) if cinematic else (0.09, 0.07, 0.10, 0.08))[index % 4]
    zoom = f"1.01+{travel}*{eased}"
    drift = 0 if cinematic else 0.12
    x = f"(iw-iw/zoom)*(0.5+({direction * drift})*{eased})"
    y = "(ih-ih/zoom)*0.5" if cinematic else f"(ih-ih/zoom)*(0.5+({0.05 if index % 3 else -0.05})*{eased})"
    return (
        f"zoompan=z='{zoom}':x='{x}':y='{y}':"
        f"d={frames}:s={width}x{height}:fps={FPS},"
        f"setsar=1,format=yuv420p,settb=1/{FPS},setpts=PTS-STARTPTS"
    )


def shot_filter(frames: int, width: int, height: int, *, full_bleed: bool = False) -> str:
    """Fit a 5 s Luma clip to its slot by slowing it down smoothly, never freezing on the last frame."""
    stretch = max(1.0, frames / LUMA_FRAMES)
    if height > width and not full_bleed:
        framing = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x1d1714,"
        )
    else:
        framing = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},"
    resample = f"fps={FPS}," if full_bleed else f"framerate=fps={FPS},fps={FPS},"
    return (
        f"setpts={stretch:.4f}*(PTS-STARTPTS),{resample}"
        f"{framing}"
        "unsharp=5:5:0.4:5:5:0.0,"
        f"tpad=stop_mode=clone:stop=-1,trim=end_frame={frames},"
        f"setsar=1,format=yuv420p,settb=1/{FPS},setpts=PTS-STARTPTS"
    )


def _prepare_photo(
    source: Path, target: Path, width: int, height: int, *, full_bleed: bool = False, focus_x: float = 0.5,
) -> None:
    from PIL import Image, ImageFilter, ImageOps
    from app.Viedo_generation.render import _cover

    with Image.open(source) as opened:
        photo = ImageOps.exif_transpose(opened).convert("RGB")
    if full_bleed or (width >= height and photo.width >= photo.height * 1.2):
        _cover(photo, width * 2, height * 2, focus_x=focus_x).save(target, compress_level=2)
        return
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


def draw_overlay(path: Path, slide: dict[str, Any], lines: list[str], width: int, height: int,
                 *, cinematic: bool = False, broker_card: dict[str, Any] | None = None) -> None:
    from PIL import Image, ImageDraw
    from app.Viedo_generation.render import _add_logo, _font, _wrap_to_width
    from app.Viedo_generation.presenter import presenter_enabled, presenter_bounds

    if width >= height or cinematic or broker_card:
        return draw_wide_overlay(path, slide, lines, width, height, broker_card=broker_card)

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


def _draw_broker_avatar(canvas: Any, position: tuple[int, int], size: int, name: str, url: str | None) -> None:
    from PIL import Image, ImageDraw, ImageOps
    from app.Viedo_generation.render import _broker_photo, _font

    photo = _broker_photo(url) if url else None
    avatar = Image.new("RGB", (size, size), "#55382C") if photo is None else ImageOps.fit(
        photo, (size, size), Image.Resampling.LANCZOS,
    )
    if photo is None:
        initials = "".join(word[0] for word in name.split()[:2]).upper() or "A"
        ImageDraw.Draw(avatar).text(
            (size // 2, size // 2), initials, font=_font(round(size * 0.36), bold=True),
            fill="white", anchor="mm",
        )
    mask = Image.new("L", (size, size))
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    canvas.paste(avatar, position, mask)


def draw_wide_overlay(path: Path, slide: dict[str, Any], lines: list[str], width: int, height: int,
                      *, broker_card: dict[str, Any] | None = None) -> None:
    """Film overlays for both aspect ratios, with facts laid over the footage."""
    from PIL import Image, ImageDraw
    from app.Viedo_generation.render import _add_logo, _font

    portrait = height > width
    s = min(width, height) / 1080
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # Gradient keeps the shot clean up top and the text readable at the bottom.
    band = round(height * 0.5)
    gradient = Image.new("L", (1, band))
    gradient.putdata([round(215 * (i / band) ** 1.6) for i in range(band)])
    shade = Image.new("RGBA", (width, band), (10, 8, 7, 255))
    shade.putalpha(gradient.resize((width, band)))
    canvas.alpha_composite(shade, (0, height - band))
    draw = ImageDraw.Draw(canvas)

    margin = round(80 * s)
    white, soft, muted = (255, 255, 255, 255), (226, 218, 210, 235), (196, 186, 176, 220)
    stats = [x for x in (slide.get("stats") or []) if x.get("value")][:3]
    title = str(slide.get("title") or "")
    subtitle = str(slide.get("subtitle") or "")
    extra = "" if stats else next((str(line) for line in lines if line), "")

    def fit(text: str, size: int, bold: bool, max_width: int):
        font = _font(round(size * s), bold=bold)
        while draw.textlength(text, font=font) > max_width and font.size > 14:
            font = _font(font.size - 2, bold=bold)
        return font

    def spaced(x: int, y: int, text: str, font, fill, gap: float) -> None:
        for ch in text:
            draw.text((x, y), ch, font=font, fill=fill)
            x += round(draw.textlength(ch, font=font) + gap)

    bottom = height - round((200 if portrait else 64) * s)
    if broker_card:
        name = broker_card.get("name") or "ACOT Agent"
        scale = min(width / 1080, height / 1920)
        center = width // 2
        canvas.paste((28, 21, 17, 255), (0, 0, width, height))
        top = round(height * 0.19)
        avatar_size = round(420 * scale)
        avatar_x = center - avatar_size // 2
        avatar_y = top
        _draw_broker_avatar(canvas, (avatar_x, avatar_y), avatar_size, name, broker_card.get("photo_url"))
        rows = [
            ("YOUR AGENT", 32, True, (243, 195, 159, 255)),
            (name, 80, True, white),
            (broker_card.get("agency"), 46, False, soft),
        ]
        y = avatar_y + avatar_size + round(55 * scale)
        text_width = width - round(160 * scale)
        for value, size, bold, colour in rows:
            if value:
                font = fit(value, max(1, round(size * scale / s)), bold, text_width)
                draw.text((center, y), value, font=font, fill=colour, anchor="mt")
                y += round((size + 20) * scale)
        y += round(35 * scale)
        for label, key in (("PHONE", "phone"), ("EMAIL", "email")):
            if contact := broker_card.get(key):
                draw.text((center, y), label, font=_font(round(27 * scale), bold=True), fill=muted, anchor="mt")
                y += round(38 * scale)
                font = fit(contact, max(1, round(49 * scale / s)), True, text_width)
                draw.text((center, y), contact, font=font, fill=white, anchor="mt")
                y += round(85 * scale)
        if stars := broker_card.get("dld_stars"):
            label = f"DLD · {stars} {'Star' if stars == 1 else 'Stars'}"
            badge_font = _font(round(30 * scale), bold=True)
            badge_y = y + round(23 * scale)
            half_w = round(draw.textlength(label, font=badge_font) / 2 + 24 * scale)
            half_h = round(24 * scale)
            draw.rounded_rectangle((center - half_w, badge_y - half_h, center + half_w, badge_y + half_h),
                                   radius=half_h, fill="#55382C")
            draw.text((center, badge_y), label, font=badge_font, fill="#F3C9A4", anchor="mm")
        _add_logo(canvas).save(path, compress_level=2)
        return
    if stats:
        col_w = (width - 2 * margin) // 3
        label_f, value_f, sub_f = _font(round(19 * s), bold=True), _font(round(38 * s), bold=True), _font(round(20 * s))
        row_top = bottom - round(112 * s)
        for i, stat in enumerate(stats):
            x = margin + i * col_w
            spaced(x, row_top, stat["label"], label_f, muted, 2.2 * s)
            value = stat["value"]
            vf = value_f
            while draw.textlength(value, font=vf) > col_w - round(24 * s) and vf.size > 16:
                vf = _font(vf.size - 2, bold=True)
            draw.text((x, row_top + round(30 * s)), value, font=vf, fill=white)
            if stat.get("sub"):
                sub_fitted = fit(stat["sub"], 20, False, col_w - round(20 * s)) if portrait else sub_f
                draw.text((x, row_top + round(80 * s)), stat["sub"], font=sub_fitted, fill=muted)
        rule_y = row_top - round(26 * s)
        draw.line((margin, rule_y, width - margin, rule_y), fill=(255, 255, 255, 70), width=max(1, round(s)))
        text_bottom = rule_y - round(22 * s)
    else:
        text_bottom = bottom
        if extra:
            ef = fit(extra, 26, False, width - 2 * margin)
            draw.text((margin, text_bottom - ef.size), extra, font=ef, fill=muted)
            text_bottom -= ef.size + round(18 * s)

    sub_f = fit(subtitle, 30, False, width - 2 * margin) if subtitle else None
    title_f = fit(title, 66 if not stats else 60, True, width - 2 * margin)
    if sub_f:
        draw.text((margin, text_bottom - sub_f.size), subtitle, font=sub_f, fill=soft)
        text_bottom -= sub_f.size + round(12 * s)
    draw.text((margin, text_bottom - round(title_f.size * 1.15)), title, font=title_f, fill=white)
    # Thin accent above the title, in ACOT brown.
    accent_y = text_bottom - round(title_f.size * 1.15) - round(22 * s)
    draw.line((margin, accent_y, margin + round(56 * s), accent_y), fill=(193, 146, 116, 255), width=max(2, round(3 * s)))
    canvas = _add_logo(canvas)
    canvas.save(path, compress_level=2)


def render_slides(slides: list[dict[str, Any]], pack: dict[str, Any], work: Path,
                  durations: list[float], ffmpeg: str, *, frame_size: tuple[int, int],
                  presenter: str | None = None, broker_card: dict[str, Any] | None = None) -> list[Path]:
    from app.Viedo_generation.render import AGENT_PANEL_SECONDS, RenderError, _download_image, _run
    from app.Viedo_generation.presenter import build_presenter_sequence, presenter_enabled

    from app.Viedo_generation import storage
    from app.Viedo_generation.luma import animate_photos

    cinematic = bool(broker_card)
    width, height = frame_size
    photo_height = height if cinematic or width >= height else round(height * 0.68) // 2 * 2
    focal_points = {
        url: slide["focus_x"] for slide in slides if slide.get("focus_x") is not None
        for url in slide.get("photo_urls", [])
    }
    cache: dict[str, Path | None] = {}
    plans = []
    for slide, duration in zip(slides, durations):
        total_frames = max(1, round(duration * FPS))
        if cinematic and slide.get("kind") == "agent":
            plans.append((slide, duration, total_frames, []))
            continue
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
                        _prepare_photo(
                            source, target, width, photo_height,
                            full_bleed=cinematic, focus_x=focal_points.get(url, 0.5),
                        )
                    cache[url] = target if source else None
                except (OSError, ValueError):
                    cache[url] = None
            if cache[url]:
                sources.append((url, cache[url]))
        if not sources:
            raise RenderError(f"No usable property photos for {slide.get('title') or 'slide'}. Please check the image URLs.")
        shot_minimum = 5 * FPS if cinematic else MIN_PHOTO_FRAMES
        plans.append((slide, duration, total_frames, sources[:max(1, total_frames // shot_minimum)]))

    motion_plans = [plan for plan in plans if plan[0].get("kind") not in ("amenity", "agent")]
    ordered = [sources[0][0] for *_, sources in motion_plans] + [url for *_, sources in motion_plans for url, _ in sources[1:]]
    amenities = [
        sources[0][0] for slide, _, _, sources in plans
        if slide.get("kind") == "amenity" and slide.get("amenity_photo")
    ]
    interiors = {
        room["url"] for project in pack.get("projects", [])
        for room in project.get("interior_photos") or [] if room.get("url")
    }
    if cinematic:
        exterior = next((url for url in ordered if url not in interiors), None)
        interior = next((url for url in ordered if url in interiors), None)
        ordered = amenities + [url for url in (exterior, interior) if url] + ordered
    shots = animate_photos(
        ordered, storage.DEFAULT_OUT_DIR / "luma", frame_size=frame_size,
        flythrough=cinematic, interior_urls=interiors, amenity_urls=set(amenities),
        prepared_photos=cache if cinematic else None,
    )
    clips = []
    for slide_index, (slide, duration, total_frames, sources) in enumerate(plans):
        last_slide = slide_index == len(plans) - 1
        agent_slide = slide.get("kind") == "agent"
        target = work / f"clip_{slide_index:02d}.mp4"
        if cinematic and agent_slide:
            card = work / "agent.png"
            draw_overlay(card, slide, [], width, height, broker_card=broker_card)
            _run([ffmpeg, "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(card),
                  "-frames:v", str(total_frames), "-an", "-vf", "setsar=1", "-c:v", "libx264", "-preset", "faster",
                  "-crf", "15", "-pix_fmt", "yuv420p", "-video_track_timescale", "12800", str(target)],
                 "Agent screen render failed")
            clips.append(target)
            continue
        show_broker = last_slide and bool(broker_card) and not agent_slide
        tail = 0
        if show_broker:
            tail = AGENT_PANEL_SECONDS * FPS
        elif not last_slide and not cinematic:
            tail = SLIDE_FADE_FRAMES
        frames = total_frames + tail
        moving = {i for i, (url, _) in enumerate(sources) if shots.get(url)} if cinematic else None
        lengths, offsets = shot_timing(frames, len(sources), moving=moving)
        overlap = (sum(lengths) - frames) // max(1, len(sources) - 1)
        cmd = [ffmpeg, "-y", "-filter_complex_threads", "2"]
        for url, still in sources:
            cmd.extend(["-i", str(shots.get(url) or still)])
        pages = caption_schedule(slide, total_frames)
        if not show_broker:
            # The last caption stays up through the dissolve into the next slide.
            pages[-1] = (pages[-1][0], pages[-1][1], frames)
        caption_intervals = [(start, end) for _, start, end in pages]
        for i, (page, _, _) in enumerate(pages):
            overlay = work / f"overlay_{slide_index}_{i}.png"
            draw_overlay(overlay, slide, page, width, height, **({"cinematic": True} if cinematic else {}))
            cmd.extend(["-i", str(overlay)])
        if show_broker:
            overlay = work / "broker-overlay.png"
            draw_overlay(overlay, slide, [], width, height, broker_card=broker_card)
            cmd.extend(["-i", str(overlay)])
            caption_intervals.append((total_frames, frames))
        filters = [
            f"[{i}:v]{shot_filter(length, width, photo_height, full_bleed=cinematic) if shots.get(url) else motion_filter(i + slide_index, length, width, photo_height, cinematic=cinematic)}[p{i}]"
            for i, ((url, _), length) in enumerate(zip(sources, lengths))
        ]
        previous = "p0"
        if cinematic and len(sources) > 1:
            inputs = "".join(f"[p{i}]" for i in range(len(sources)))
            filters.append(f"{inputs}concat=n={len(sources)}:v=1:a=0,settb=1/{FPS},setpts=N[mix]")
            previous = "mix"
        else:
            for i, offset in enumerate(offsets, 1):
                filters.append(f"[{previous}][p{i}]xfade=transition=fade:duration={overlap / FPS:.6f}:offset={offset / FPS:.6f}[mix{i}]")
                previous = f"mix{i}"
        filters.append(f"[{previous}]pad={width}:{height}:0:0:color=0x1d1714[base]")
        previous = "base"
        for i, (start, end) in enumerate(caption_intervals):
            filters.append(
                f"[{previous}][{len(sources) + i}:v]overlay=0:0"
                f":enable='gte(t,{start / FPS:.2f})*lt(t,{end / FPS:.2f})':format=yuv420[card{i}]"
            )
            previous = f"card{i}"
        if not cinematic and presenter_enabled():
            agent = build_presenter_sequence(
                [slide], [duration], work / f"agent_{slide_index}",
                voice=presenter, video_width=width, video_height=height,
            )
            cmd.extend(["-framerate", str(agent.frame_rate), "-start_number", "0", "-i", agent.pattern])
            # Framesync holds the most recent sprite until the next 10fps frame.
            # The photo stream determines the clip length, including the last frame.
            filters.append(
                f"[{previous}][{len(sources) + len(caption_intervals)}:v]"
                f"overlay=x={agent.x}:y={agent.y}:format=yuv420[presented]"
            )
            previous = "presented"
        # Near-lossless intermediate; the final stitch sets the delivery quality.
        cmd.extend(["-filter_complex", ";".join(filters), "-map", f"[{previous}]", "-frames:v", str(frames),
                    "-an", "-r", str(FPS), "-c:v", "libx264", "-threads", "2", "-preset", "faster",
                    "-crf", "15",
                    "-pix_fmt", "yuv420p", "-video_track_timescale", "12800", str(target)])
        _run(cmd, f"Photo motion render failed for {slide.get('title')}")
        clips.append(target)
    return clips
