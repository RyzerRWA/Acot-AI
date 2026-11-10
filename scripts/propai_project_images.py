"""
Copy Prop-AI project images into our Google Cloud photos bucket and record them in public.propai_project_images.

    python scripts/propai_project_images.py [--pages N]     # --pages N = only the first N pages of 100 projects

Prop-AI image links are presigned and expire after 1 hour, so each page of projects is downloaded and uploaded right away.
Projects that already have images recorded are skipped, so re-running resumes. Costs one Prop-AI request per 100 projects
(about 62 in all). Needs `gcloud auth login` with write access to the bucket; stops at the first auth error.
Env: PROJECT_IMAGES_BUCKET (default kinetic-axle-442714-m6-dubai-photos), PROPAI_API_KEY, PROPAI_BASE_URL.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlparse

import httpx

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.dirname(here))
import propai_sync as ps  # noqa: E402  (loads .env, gives us the project key function)
from app.database.postgres import _get_pool  # noqa: E402

BUCKET = os.getenv("PROJECT_IMAGES_BUCKET", "kinetic-axle-442714-m6-dubai-photos")
WORKERS = 8
_token = {"value": None, "at": 0.0}


class AuthError(RuntimeError):
    pass


def token():
    if time.time() - _token["at"] > 40 * 60:
        out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
        if out.returncode or not out.stdout.strip():
            raise AuthError("gcloud is not logged in. Run: gcloud auth login")
        _token.update(value=out.stdout.strip(), at=time.time())
    return _token["value"]


def folder(project_key):
    return "projects/p_" + hashlib.sha1(project_key.encode()).hexdigest()[:10]


def ordered(urls):  # cover image first
    return sorted(urls, key=lambda u: "_cover" not in urlparse(u).path)


MIME = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
BIG = 500_000      # bytes: smaller images are copied untouched
MAX_SIDE = 1600    # px


def shrink(data: bytes) -> bytes:
    """Resize a large image to at most MAX_SIDE px and re-encode as JPEG, using macOS `sips`. Falls back to the original."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in"), os.path.join(tmp, "out.jpg")
        with open(src, "wb") as f:
            f.write(data)
        done = subprocess.run(["sips", "-Z", str(MAX_SIDE), "-s", "format", "jpeg", "-s", "formatOptions", "75", src, "--out", dst],
                              capture_output=True)
        if done.returncode == 0 and os.path.exists(dst):
            with open(dst, "rb") as f:
                small = f.read()
            if 0 < len(small) < len(data):
                return small
    return data


def copy_one(args):
    project_key, position, url = args
    ext = os.path.splitext(urlparse(url).path)[1].lower().lstrip(".") or "jpg"
    if ext == "jpeg":
        ext = "jpg"
    try:
        image = httpx.get(url, timeout=60)
        if image.status_code != 200 or not image.content:
            return None
        data = image.content
        if len(data) > BIG:
            shrunk = shrink(data)
            if shrunk is not data:
                data, ext = shrunk, "jpg"
        name = f"{folder(project_key)}/{position:02d}.{ext}"
        ctype = MIME.get(ext, "image/jpeg")
        up = httpx.post(f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name={quote(name, safe='')}",
                        content=data, headers={"Authorization": f"Bearer {token()}", "Content-Type": ctype}, timeout=120)
    except httpx.HTTPError:
        return None
    if up.status_code == 401:  # the access token expired: renew it once and retry this image
        _token["at"] = 0.0
        up = httpx.post(f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name={quote(name, safe='')}",
                        content=data, headers={"Authorization": f"Bearer {token()}", "Content-Type": ctype}, timeout=120)
    if up.status_code in (401, 403):
        raise AuthError(f"upload refused ({up.status_code}): {up.text[:200]}")
    if up.status_code != 200:
        return None
    return (project_key, position, name, len(data), ctype)


if __name__ == "__main__":
    max_pages = int(sys.argv[sys.argv.index("--pages") + 1]) if "--pages" in sys.argv else None
    key_of = ps.JOBS["projects"]["key"]
    pool = _get_pool()
    with pool.connection() as c:
        done = {r[0] for r in c.execute("select distinct project_key from propai_project_images")}
    token()  # fail fast if gcloud is logged out
    copied = requests = 0
    try:
        with httpx.Client(base_url=os.environ["PROPAI_BASE_URL"], timeout=120, headers={"x-api-key": os.environ["PROPAI_API_KEY"]}) as api, \
                ThreadPoolExecutor(WORKERS) as threads:
            page = 1
            while max_pages is None or page <= max_pages:
                resp = api.get("/data/projects/basic-details", params={"region": "dubai", "page_size": 100, "page": page})
                requests += 1
                if resp.status_code == 429:
                    print("STOPPED: Prop-AI request limit reached (429). Re-run when the limit is raised or reset.")
                    sys.exit(1)
                resp.raise_for_status()
                body = resp.json()
                rows = body["data"]
                jobs = [(key_of(r), i, u) for r in rows if key_of(r) not in done
                        for i, u in enumerate(ordered(r.get("image_urls") or []))]
                results = [x for x in threads.map(copy_one, jobs) if x]
                if results:
                    with pool.connection() as c:
                        c.cursor().executemany(
                            "insert into propai_project_images(project_key,position,object_name,bytes,content_type) values (%s,%s,%s,%s,%s) "
                            "on conflict (project_key,position) do update set object_name=excluded.object_name, bytes=excluded.bytes, copied_at=now()",
                            results)
                copied += len(results)
                print(f"page {page}: {len(results)}/{len(jobs)} images copied, {copied} total, {requests} Prop-AI requests")
                if len(rows) < 100:
                    break
                page += 1
    except AuthError as error:
        print("STOPPED:", error)
        sys.exit(1)
