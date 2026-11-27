"""Run one paid Ray 3.2 entrance-clip pilot from a verified pair of photo URLs."""

import argparse
import base64
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote, urlsplit

import httpx
from dotenv import load_dotenv

BASE = "https://agents.lumalabs.ai/v1"
PROMPT = (
    "One slow, steady forward camera move from the exterior through this exact "
    "entrance into the lobby. Match both supplied frames. Preserve the building, "
    "doorway, layout, materials and furniture. No people, text or graphic overlays."
)


class LumaError(RuntimeError):
    pass


def _https(value: str) -> bool:
    url = urlsplit(value)
    return url.scheme == "https" and bool(url.hostname) and url.username is None


def _json(response: httpx.Response) -> dict:
    if not response.is_success:
        raise LumaError(f"Luma API returned HTTP {response.status_code}.")
    try:
        data = response.json()
    except ValueError:
        raise LumaError("Luma API returned invalid JSON.") from None
    if not isinstance(data, dict):
        raise LumaError("Luma API returned an unexpected response.")
    return data


def generate_entrance_clip(
    client: httpx.Client, api_key: str, exterior: str | bytes, interior: str | None, output: Path,
    *, initial_wait: int = 30, poll_interval: int = 2,
    resolution: str = "720p", prompt: str = PROMPT, aspect_ratio: str = "9:16",
) -> str:
    if not api_key.strip():
        raise LumaError("Set LUMA_AGENTS_API_KEY in the Python environment.")
    if (isinstance(exterior, str) and not _https(exterior)) or (interior is not None and not _https(interior)):
        raise LumaError("Photo URLs must use HTTPS.")
    start_frame = (
        {"data": base64.b64encode(exterior).decode("ascii"), "media_type": "image/jpeg"}
        if isinstance(exterior, bytes) else {"url": exterior}
    )
    if output.suffix.lower() != ".mp4" or output.exists():
        raise LumaError("Choose a new .mp4 output path.")
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    job = _json(client.post(f"{BASE}/generations", headers=headers, json={
        "model": "ray-3.2",
        "type": "video",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "video": {
            "resolution": resolution,
            "duration": "5s",
            "start_frame": start_frame,
            **({"end_frame": {"url": interior}} if interior else {}),
        },
    }))
    job_id = job.get("id")
    if not isinstance(job_id, str) or not job_id:
        raise LumaError("Luma did not return a generation ID.")
    deadline = time.monotonic() + 600
    wait = initial_wait
    while job.get("state") not in {"completed", "failed", "cancelled", "canceled"}:
        if time.monotonic() + wait >= deadline:
            raise LumaError(f"Luma job {job_id} timed out. Check that job before submitting again.")
        time.sleep(wait)
        job = _json(client.get(f"{BASE}/generations/{quote(job_id, safe='')}", headers=headers))
        wait = poll_interval
    if job["state"] != "completed":
        raise LumaError(f"Luma job {job_id} ended with state {job['state']}.")
    outputs = job.get("output") or []
    url = outputs[0].get("url") if outputs and isinstance(outputs[0], dict) else None
    if not isinstance(url, str) or not _https(url):
        raise LumaError(f"Luma job {job_id} returned no HTTPS download URL.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with client.stream("GET", url) as response:
            if not response.is_success:
                raise LumaError(f"Luma download returned HTTP {response.status_code}.")
            with NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as file:
                temporary = Path(file.name)
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise LumaError("The clip exceeds the 100 MB pilot download limit.")
                    file.write(chunk)
                if not size:
                    raise LumaError("Luma returned an empty clip.")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return job_id


SHOT_PROMPT = (
    "One slow, steady cinematic camera move through this exact scene. Preserve the "
    "building, layout, materials, furniture and view exactly as shown. Add nothing new. "
    "No people, text or graphic overlays."
)


def luma_enabled() -> bool:
    off = (os.getenv("VIDEO_LUMA_ENABLED") or "true").strip().lower() in {"0", "false", "no", "off"}
    return not off and bool((os.getenv("LUMA_AGENTS_API_KEY") or "").strip())


def animate_photos(urls: list[str], cache_dir: Path) -> dict[str, Path | None]:
    """Turn real photos into 5s Luma shots, in parallel, cached per photo URL.

    A photo that fails (or is over the cap, or not HTTPS) maps to None so the caller
    falls back to the still-photo motion. Never raises.
    """
    import hashlib
    import logging
    from concurrent.futures import ThreadPoolExecutor

    unique = list(dict.fromkeys(u for u in urls if isinstance(u, str)))
    result: dict[str, Path | None] = dict.fromkeys(unique)
    if not luma_enabled():
        return result
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Cost balance: a few hero shots move with Luma; the rest use free photo motion. Cached shots are free.
    cap = max(0, int(os.getenv("VIDEO_LUMA_MAX_SHOTS") or 3))
    key = os.getenv("LUMA_AGENTS_API_KEY", "")
    from app.Viedo_generation.render import HEIGHT, WIDTH
    aspect = "16:9" if WIDTH >= HEIGHT else "9:16"
    todo = []
    for url in unique:
        path = cache_dir / f"{hashlib.sha256(f'{url}|{SHOT_PROMPT}|{aspect}'.encode()).hexdigest()[:24]}.mp4"
        if path.is_file():
            result[url] = path
        elif _https(url) and len(todo) < cap:
            todo.append((url, path))

    def one(item: tuple[str, Path]) -> tuple[str, Path | None]:
        url, path = item
        try:
            with httpx.Client(timeout=30) as client:
                generate_entrance_clip(client, key, url, None, path, prompt=SHOT_PROMPT, aspect_ratio=aspect)
            return url, path
        except (LumaError, httpx.HTTPError, OSError) as exc:
            logging.getLogger(__name__).warning("Luma shot failed, using still photo: %s", exc)
            return url, None

    # ponytail: threads block on Luma polling; fine for one render at a time.
    with ThreadPoolExecutor(max_workers=4) as pool:
        result.update(pool.map(one, todo))
    return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exterior", required=True, help="HTTPS URL for the exterior entrance photo")
    parser.add_argument("--interior", help="Optional HTTPS URL for a matching end frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", choices=("360p", "540p", "720p"), default="720p")
    parser.add_argument("--prompt", default=PROMPT, help="Override the camera direction for this shot")
    options = parser.parse_args()
    try:
        with httpx.Client(timeout=30) as client:
            job_id = generate_entrance_clip(
                client, os.getenv("LUMA_AGENTS_API_KEY", ""),
                options.exterior, options.interior, options.output,
                resolution=options.resolution, prompt=options.prompt,
            )
    except LumaError as exc:
        parser.exit(1, f"{exc}\n")
    except httpx.RequestError:
        parser.exit(1, "Could not reach Luma. Check the account before resubmitting.\n")
    print(f"Saved Luma job {job_id} to {options.output}")


if __name__ == "__main__":
    main()
