"""
In-memory video jobs (queued → rendering → ready | failed).

One render at a time so FFmpeg does not pile up. Nest can later swap this
for acot.agent_videos without changing the router contract.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from app.Viedo_generation import storage
from app.Viedo_generation.catalog_prompt import PromptError

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_render_lock = threading.Lock()


def create_job(request: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    job = {
        "id": job_id,
        "status": "queued",
        "request": dict(request),
        "pack": None,
        "script_preview": None,
        "video_path": None,
        "video_url": None,
        "error": None,
        "created_at": now,
        "finished_at": None,
    }
    with _lock:
        _jobs[job_id] = job
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return dict(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.update(fields)


def _run_job(job_id: str) -> None:
    with _render_lock:
        _update(job_id, status="rendering")
        try:
            result = _pipeline(job_id)
            _update(
                job_id,
                status="ready",
                pack=result.get("pack"),
                script_preview=result.get("script_preview"),
                video_path=result.get("video_path"),
                video_url=result.get("video_url"),
                finished_at=time.time(),
                error=None,
            )
        except Exception as exc:
            if not isinstance(exc, PromptError):
                logging.getLogger(__name__).exception("Video generation failed for job %s", job_id)
            _update(
                job_id,
                status="failed",
                error=str(exc) if isinstance(exc, PromptError) else "Video generation failed. Please try again.",
                finished_at=time.time(),
            )


def _pipeline(job_id: str) -> dict[str, Any]:
    from app.Viedo_generation.catalog_prompt import resolve_catalog_selection
    from app.Viedo_generation.pack import build_pack, label_film_interiors
    from app.Viedo_generation.render import render
    from app.Viedo_generation.script import build_script
    from app.Viedo_generation.tts import friendly_voice
    from app.Viedo_generation.narration import narrate_script

    job = get_job(job_id) or {}
    req = job.get("request") or {}
    user_id = (req.get("user_id") or "").strip()
    if not user_id:
        raise RuntimeError("user_id is required.")

    with_voice = bool(req.get("with_voice", True))
    voice = req.get("voice")
    bedrooms = req.get("bedrooms")
    limit = int(req.get("limit") or 2)
    candidates = resolve_catalog_selection(req.get("prompt_text"), req.get("project_keys"), req.get("communities"))
    # One picked property (the one-click flow) makes a single-property video.
    single = len(req.get("project_keys") or []) == 1 and not req.get("prompt_text")
    pack_kwargs: dict[str, Any] = {"limit": limit, "min_projects": 1 if single else 2, "web": single}
    if bedrooms is not None:
        pack_kwargs["bedrooms"] = int(bedrooms)
    pack = build_pack(candidates, **pack_kwargs)
    if not pack.get("ok"):
        raise PromptError(pack.get("reason") or "No complete project data.")

    language = "hi" if req.get("language") == "hi" else "en"
    cinematic = bool(req.get("broker_card"))
    if cinematic and not single:
        label_film_interiors(pack)
    script = build_script(pack, language=language, cinematic=cinematic, broker_card=req.get("broker_card"))
    if not script.get("ok"):
        raise RuntimeError(script.get("reason") or "Script failed.")

    audio_path = None
    if with_voice:
        voice = friendly_voice(voice, language)
        audio_path = storage.mp3_path(job_id).with_suffix(".wav")
        script = narrate_script(script, audio_path, voice=voice)

    out = render(
        pack,
        script,
        audio_path=audio_path,
        job_id=job_id,
        presenter=voice,
        broker_card=req.get("broker_card"),
    )
    preview = {
        "word_count": script.get("word_count"),
        "estimated_seconds": script.get("estimated_seconds"),
        "project_names": [p.get("name") for p in (pack.get("projects") or [])],
    }
    return {
        "pack": {
            "ok": True,
            "projects": preview["project_names"],
            "bedrooms": pack.get("bedrooms"),
        },
        "script_preview": preview,
        "video_path": str(out),
        "video_url": storage.file_url(job_id),
    }


def prompt_for_user(user_id: str) -> dict[str, Any]:
    """Auto-fill payload for the agent editable line."""
    from app.Viedo_generation.history import searched_projects

    history = searched_projects(user_id)
    projects = history.get("projects") or []
    bed_bits: list[str] = []
    for p in projects[:12]:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        lo, hi = p.get("bedrooms_min"), p.get("bedrooms_max")
        if lo is not None and hi is not None and lo != hi:
            bed_bits.append(f"{name} ({int(lo)}-{int(hi)} bed)")
        elif lo is not None:
            bed_bits.append(f"{name} ({int(lo)} bed)" if float(lo) > 0 else f"{name} (studio)")
        else:
            bed_bits.append(name)

    communities = history.get("communities") or []
    community_names = [c.get("name") for c in communities if isinstance(c, dict) and c.get("name")]
    line = ", ".join(bed_bits or community_names[:12])
    return {
        "user_id": user_id,
        "prompt_text": line,
        "projects": [
            {
                "key": p.get("key"),
                "name": p.get("name"),
                "community": p.get("community"),
                "bedrooms_min": p.get("bedrooms_min"),
                "bedrooms_max": p.get("bedrooms_max"),
            }
            for p in projects[:20]
        ],
        "communities": community_names[:20],
        "empty": not bool(line),
    }
