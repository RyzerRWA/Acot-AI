"""
One-shot local runner: catalog → pack → script → (optional TTS) → MP4.

Usage (from Acot-AI- folder):
  python -m app.Viedo_generation.run USER_ID --projects "Ciel, Mudon Al Ranim 3"
  python -m app.Viedo_generation.run USER_ID --projects "Dubai Marina" --with-voice --voice female

Output default: C:\\acot_video_output\\{job_id}.mp4
Without --with-voice the video is silent. Requested narration must succeed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an ACOT property explainer MP4.")
    parser.add_argument(
        "user_id",
        help="Buyer or lead identifier for the generation request.",
    )
    parser.add_argument("--projects", required=True, help="Comma-separated catalog project or community names.")
    parser.add_argument(
        "--bedrooms",
        type=int,
        default=None,
        help="Optional bedroom filter (1-4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Legacy option. Videos compare two projects.",
    )
    parser.add_argument(
        "--job-id",
        default="preview",
        help="Output filename stem (default: preview → preview.mp4).",
    )
    parser.add_argument(
        "--with-voice",
        action="store_true",
        help="Try Google TTS narration (GOOGLE_CLOUD_API_KEY or service-account JSON).",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Narration voice: male | female | or a Neural2 name (e.g. en-US-Neural2-F).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Full path for the MP4 (overrides --job-id / VIDEO_OUTPUT_DIR).",
    )
    args = parser.parse_args(argv)

    # Allow `python app/Viedo_generation/run.py ...` from any cwd.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.Viedo_generation.catalog_prompt import PromptError, resolve_catalog_selection
    from app.Viedo_generation.pack import build_pack
    from app.Viedo_generation.render import render
    from app.Viedo_generation.script import build_script

    t0 = time.time()
    print(f"user={args.user_id}")
    print("1/4 catalog selection...")
    try:
        candidates = resolve_catalog_selection(args.projects)
    except PromptError as exc:
        print(str(exc))
        return 1

    print("2/4 pack...")
    pack_kwargs: dict = {"limit": args.limit}
    if args.bedrooms is not None:
        pack_kwargs["bedrooms"] = args.bedrooms
    pack = build_pack(candidates, **pack_kwargs)
    if not pack.get("ok"):
        print(f"pack failed: {pack.get('reason')}")
        return 1
    print(f"   projects={len(pack.get('projects') or [])}")

    print("3/4 script...")
    script = build_script(pack)
    if not script.get("ok"):
        print(f"script failed: {script.get('reason')}")
        return 1
    print(f"   ~{script.get('estimated_seconds')}s  words={script.get('word_count')}")

    audio_path: Path | None = None
    if args.with_voice:
        print("4/4 TTS + render...")
        try:
            from app.Viedo_generation.tts import resolve_voice
            from app.Viedo_generation.narration import narrate_script

            voice_name = resolve_voice(args.voice)
            print(f"   tts voice: {voice_name}")
            out_dir = Path(args.output).parent if args.output else Path(r"C:\acot_video_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_path = out_dir / f"{args.job_id}.wav"
            script = narrate_script(script, audio_path, voice=voice_name)
            print(f"   synchronized duration: {script['estimated_seconds']:.2f}s")
            print(f"   audio: {audio_path}")
        except Exception as exc:
            name = type(exc).__name__
            print(f"   Narration failed ({name}: {exc})")
            return 1
    else:
        print("4/4 render (silent)...")

    t1 = time.time()
    path = render(
        pack,
        script,
        audio_path=audio_path,
        output_path=args.output,
        job_id=args.job_id,
        presenter=args.voice,
    )
    t2 = time.time()

    size_kb = path.stat().st_size / 1024
    print(f"wrote: {path}")
    print(f"size:  {size_kb:.0f} KB")
    print(f"render: {t2 - t1:.1f}s | total: {t2 - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
