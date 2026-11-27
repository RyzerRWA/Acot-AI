"""Create a short, factual fly-through-style reel from real property photos."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WIDTH = 1080
HEIGHT = 1920
FPS = 25
SHOT_SECONDS = 5.0
TRANSITION_SECONDS = 0.8


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _prepare_photo(source: Path, target: Path) -> None:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    canvas_size = (WIDTH * 2, HEIGHT * 2)
    with Image.open(source) as opened:
        photo = ImageOps.exif_transpose(opened).convert("RGB")

    background = ImageOps.fit(photo, canvas_size, Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(42))
    background = ImageEnhance.Brightness(background).enhance(0.68)

    foreground = ImageOps.contain(photo, canvas_size, Image.Resampling.LANCZOS)
    top = (canvas_size[1] - foreground.height) // 2
    background.paste(foreground, ((canvas_size[0] - foreground.width) // 2, top))
    background.save(target, format="PNG", compress_level=2)


def _brand_overlay(target: Path, logo_path: Path, title: str, subtitle: str) -> None:
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    margin = 42

    if logo_path.is_file():
        with Image.open(logo_path) as opened:
            logo = opened.convert("RGBA")
        logo_width = 180
        logo = logo.resize((logo_width, round(logo.height * logo_width / logo.width)), Image.Resampling.LANCZOS)
        badge = (margin, margin, margin + logo.width + 38, margin + logo.height + 30)
        draw.rounded_rectangle(badge, radius=18, fill=(255, 255, 255, 230))
        overlay.alpha_composite(logo, (margin + 19, margin + 15))

    title_font = _font(52, bold=True)
    subtitle_font = _font(34)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    panel_width = max(title_box[2], subtitle_box[2]) + 54
    panel_top = HEIGHT - 220
    draw.rounded_rectangle(
        (margin, panel_top, margin + panel_width, HEIGHT - margin),
        radius=22,
        fill=(24, 18, 15, 205),
    )
    draw.text((margin + 27, panel_top + 25), title, font=title_font, fill="white")
    draw.text((margin + 27, panel_top + 98), subtitle, font=subtitle_font, fill=(220, 190, 170, 255))
    overlay.save(target, format="PNG", compress_level=2)


def create_sample(images: list[Path], output: Path, logo_path: Path, title: str, subtitle: str) -> Path:
    from app.Viedo_generation.render import _ffmpeg_bin, _run

    if len(images) < 2:
        raise ValueError("At least two property photos are required.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_per_shot = round(SHOT_SECONDS * FPS)
    transition_frames = round(TRANSITION_SECONDS * FPS)
    total_frames = frames_per_shot * len(images) - transition_frames * (len(images) - 1)

    work = Path(tempfile.mkdtemp(prefix="acot_flythrough_"))
    try:
        prepared = []
        for index, source in enumerate(images):
            target = work / f"photo_{index}.png"
            _prepare_photo(source, target)
            prepared.append(target)
        brand = work / "brand.png"
        _brand_overlay(brand, logo_path, title, subtitle)

        ffmpeg = _ffmpeg_bin()
        command = [ffmpeg, "-y"]
        for source in prepared:
            command += ["-loop", "1", "-framerate", str(FPS), "-i", str(source)]
        command += ["-loop", "1", "-framerate", str(FPS), "-i", str(brand)]

        filters = []
        for index in range(len(prepared)):
            progress = f"min(on/{frames_per_shot - 1},1)"
            eased = f"({progress}*{progress}*(3-2*{progress}))"
            if index % 2:
                zoom = f"1.07-0.06*{eased}"
                x = f"(iw-iw/zoom)*(1-{eased})"
            else:
                zoom = f"1.01+0.06*{eased}"
                x = f"(iw-iw/zoom)*{eased}"
            filters.append(
                f"[{index}:v]zoompan=z='{zoom}':x='{x}':y='(ih-ih/zoom)/2':"
                f"d={frames_per_shot}:s={WIDTH}x{HEIGHT}:fps={FPS},"
                f"setsar=1,settb=1/{FPS},setpts=PTS-STARTPTS[v{index}]"
            )

        previous = "v0"
        elapsed = SHOT_SECONDS
        for index in range(1, len(prepared)):
            offset = elapsed - TRANSITION_SECONDS
            filters.append(
                f"[{previous}][v{index}]xfade=transition=hblur:"
                f"duration={TRANSITION_SECONDS}:offset={offset}[mix{index}]"
            )
            previous = f"mix{index}"
            elapsed += SHOT_SECONDS - TRANSITION_SECONDS
        filters.append(
            f"[{previous}][{len(prepared)}:v]overlay=0:0:shortest=1:format=auto[final]"
        )

        command += [
            "-filter_complex", ";".join(filters),
            "-map", "[final]",
            "-frames:v", str(total_frames),
            "-an",
            "-r", str(FPS),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output),
        ]
        _run(command, "Fly-through sample render failed")
        return output
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Ciel")
    parser.add_argument("--subtitle", default="Marsa Dubai")
    args = parser.parse_args()
    images = sorted(path for path in args.input_dir.glob("photo_*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    logo = Path(__file__).resolve().parents[1] / "app" / "Viedo_generation" / "assets" / "acot-logo.png"
    print(create_sample(images, args.output, logo, args.title, args.subtitle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
