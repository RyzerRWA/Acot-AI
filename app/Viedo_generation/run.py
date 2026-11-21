"""
One-shot local runner: history → pack → script → (optional TTS) → MP4.

Usage (from Acot-AI- folder):
  python -m app.Viedo_generation.run USER_ID
  python -m app.Viedo_generation.run USER_ID --bedrooms 2 --job-id my_video
  python -m app.Viedo_generation.run USER_ID --with-voice

Output default: C:\\acot_video_output\\{job_id}.mp4
Without --with-voice (or if TTS credentials are missing) the video is silent.
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
        help="Buyer / lead user UUID (searched projects come from their chat history).",
    )
    parser.add_argument(
        "--bedrooms",
        type=int,
        default=None,
        help="Optional bedroom filter (1-4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Max projects to enrich into the pack (default 4; script uses up to 3).",
    )
    parser.add_argument(
        "--job-id",
        default="preview",
        help="Output filename stem (default: preview → preview.mp4).",
    )
    parser.add_argument(
        "--with-voice",
        action="store_true",
        help="Try Google TTS narration (needs GOOGLE_APPLICATION_CREDENTIALS).",
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

    from app.Viedo_generation.history import searched_projects
    from app.Viedo_generation.pack import build_pack
    from app.Viedo_generation.render import render
    from app.Viedo_generation.script import build_script

    t0 = time.time()
    print(f"user={args.user_id}")
    print("1/4 history...")
    history = searched_projects(args.user_id)

    print("2/4 pack...")
    pack_kwargs: dict = {"limit": args.limit}
    if args.bedrooms is not None:
        pack_kwargs["bedrooms"] = args.bedrooms
    pack = build_pack(history, **pack_kwargs)
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
            from app.Viedo_generation.tts import TtsError, speak_to_file

            out_dir = Path(args.output).parent if args.output else Path(r"C:\acot_video_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_path = speak_to_file(
                script.get("full_text") or "",
                out_dir / f"{args.job_id}.mp3",
            )
            print(f"   voice: {audio_path}")
        except Exception as exc:
            # TtsError or missing creds — still produce a silent video.
            name = type(exc).__name__
            print(f"   TTS skipped ({name}: {exc})")
            audio_path = None
            print("   rendering silent mp4...")
    else:
        print("4/4 render (silent)...")

    t1 = time.time()
    path = render(
        pack,
        script,
        audio_path=audio_path,
        output_path=args.output,
        job_id=args.job_id,
    )
    t2 = time.time()

    size_kb = path.stat().st_size / 1024
    print(f"wrote: {path}")
    print(f"size:  {size_kb:.0f} KB")
    print(f"render: {t2 - t1:.1f}s | total: {t2 - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
