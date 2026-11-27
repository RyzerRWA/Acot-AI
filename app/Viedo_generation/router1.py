"""
FastAPI router for the video generation service (Nest-only).

Mounted by video_api.py on port 8090.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.Viedo_generation import jobs, storage

router = APIRouter(prefix="/api/video", tags=["video"])

INTERNAL_KEY = (os.getenv("VIDEO_INTERNAL_KEY") or "").strip()


class CreateJobBody(BaseModel):
    user_id: str = Field(..., min_length=1)
    voice: Literal["male", "female"] | None = "female"
    language: Literal["en", "hi"] = "en"
    with_voice: bool = True
    bedrooms: int | None = None
    limit: int = 2
    project_keys: list[str] | None = None
    communities: list[str] | None = None
    prompt_text: str | None = Field(default=None, max_length=1000)


def _check_key(x_video_key: str | None) -> None:
    if not INTERNAL_KEY:
        return  # local/dev: open if unset
    if (x_video_key or "").strip() != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="Invalid video API key.")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "acot-video"}


@router.get("/prompt/{user_id}")
def get_prompt(user_id: str, x_video_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Auto-fill line: searched project names + bedrooms (blank if none)."""
    _check_key(x_video_key)
    return jobs.prompt_for_user(user_id)


@router.post("/jobs")
def create_job(
    body: CreateJobBody,
    x_video_key: str | None = Header(default=None),
    x_video_scope: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_key(x_video_key)
    job = jobs.create_job({**body.model_dump(), "scope": x_video_scope})
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
    }


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    x_video_key: str | None = Header(default=None),
    x_video_scope: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_key(x_video_key)
    job = jobs.get_job(job_id)
    if not job or job["request"].get("scope") != x_video_scope:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "id": job["id"],
        "status": job["status"],
        "video_url": job.get("video_url"),
        "error": job.get("error"),
        "pack": job.get("pack"),
        "script_preview": job.get("script_preview"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
    }


@router.get("/files/{filename}")
def get_file(
    filename: str,
    x_video_key: str | None = Header(default=None),
    x_video_scope: str | None = Header(default=None),
) -> FileResponse:
    """Serve a finished MP4 (Nest proxies this; do not expose publicly in prod)."""
    _check_key(x_video_key)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.mp4", filename):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    job_id = filename[:-4]
    if x_video_scope is not None:
        # Agent preview: the job must belong to this agent's lead.
        if get_job(job_id, x_video_key, x_video_scope)["status"] != "ready":
            raise HTTPException(status_code=404, detail="File not found.")
    # No scope: Nest's signed buyer link, already checked by Nest. The internal key still applies.
    elif (job := jobs.get_job(job_id)) is not None and job["status"] != "ready":
        raise HTTPException(status_code=404, detail="File not found.")
    path = storage.mp4_path(job_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
    )
