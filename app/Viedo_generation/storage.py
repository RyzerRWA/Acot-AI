"""
Local video file storage (GCS upload can plug in later).

Default dir: VIDEO_OUTPUT_DIR or C:\\acot_video_output
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_OUT_DIR = Path(
    os.getenv("VIDEO_OUTPUT_DIR")
    or (r"C:\acot_video_output" if os.name == "nt" else Path.home() / "acot_video_output")
)


def output_dir() -> Path:
    path = DEFAULT_OUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def mp4_path(job_id: str) -> Path:
    return output_dir() / f"{job_id}.mp4"


def mp3_path(job_id: str) -> Path:
    return output_dir() / f"{job_id}.mp3"


def file_url(job_id: str, *, base_url: str = "") -> str:
    """Public-ish path Nest can fetch. base_url e.g. http://127.0.0.1:8090"""
    rel = f"/api/video/files/{job_id}.mp4"
    if base_url:
        return f"{base_url.rstrip('/')}{rel}"
    return rel
