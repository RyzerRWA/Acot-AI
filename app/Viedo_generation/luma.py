"""Run one paid Ray 3.2 entrance-clip pilot from a verified pair of photo URLs."""

import argparse
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
    client: httpx.Client, api_key: str, exterior: str, interior: str, output: Path,
    *, initial_wait: int = 30, poll_interval: int = 2,
) -> str:
    if not api_key.strip():
        raise LumaError("Set LUMA_AGENTS_API_KEY in the Python environment.")
    if not all(_https(url) for url in (exterior, interior)):
        raise LumaError("Both photo URLs must use HTTPS.")
    if output.suffix.lower() != ".mp4" or output.exists():
        raise LumaError("Choose a new .mp4 output path.")
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    job = _json(client.post(f"{BASE}/generations", headers=headers, json={
        "model": "ray-3.2",
        "type": "video",
        "prompt": PROMPT,
        "aspect_ratio": "9:16",
        "video": {
            "resolution": "720p",
            "duration": "5s",
            "start_frame": {"url": exterior},
            "end_frame": {"url": interior},
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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exterior", required=True, help="HTTPS URL for the exterior entrance photo")
    parser.add_argument("--interior", required=True, help="HTTPS URL for the matching interior entrance photo")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    try:
        with httpx.Client(timeout=30) as client:
            job_id = generate_entrance_clip(
                client, os.getenv("LUMA_AGENTS_API_KEY", ""),
                options.exterior, options.interior, options.output,
            )
    except LumaError as exc:
        parser.exit(1, f"{exc}\n")
    except httpx.RequestError:
        parser.exit(1, "Could not reach Luma. Check the account before resubmitting.\n")
    print(f"Saved Luma job {job_id} to {options.output}")


if __name__ == "__main__":
    main()
