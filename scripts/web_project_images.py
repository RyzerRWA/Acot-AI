"""
Photos from each developer's official site for every project (exterior, interiors, garden and amenities), copied into our
Google Cloud photos bucket and recorded in public.propai_project_images (source='web'), so agents never upload them.
Uses the video collector (AI web search, official sources only, listing portals blocked, images checked >= 900x500),
then a vision check drops people, maps, logos and floor plans.
Projects with Prop-AI images keep them; web photos are added after them. Busiest projects first, projects without
any photo before those that have some; resumable (projects that already have web photos are skipped).

    python scripts/web_project_images.py [--limit N] [--workers 4]
Needs OPENAI_API_KEY and `gcloud auth login` with write access to the bucket.
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.dirname(here))
import propai_project_images as pi  # noqa: E402  (loads .env; GCS upload + resize)
from app.database.postgres import _get_pool  # noqa: E402
from app.Viedo_generation.collector import _ask, collect, label_photos  # noqa: E402

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
pool = _get_pool()


def one(row):
    key, name, community, developer = row
    try:
        found = collect(key, name, community or "", developer or "")
    except Exception as exc:  # one bad site must not stop the run
        print(f"skip {name}: {exc}", flush=True)
        return 0
    # Official pages also carry event photos, maps and logos: keep only what the vision check calls the project itself,
    # outside first. No labels (key or network down) = store nothing, so a rerun retries the project.
    labels = label_photos(found["photos"])
    keep = [x["url"] for kind in ("exterior", "interior", "amenity") for x in labels if x["kind"] == kind][:12]
    with pool.connection() as c:
        start = c.execute("select coalesce(max(position) + 1, 0) from propai_project_images where project_key=%s", (key,)).fetchone()[0]
    copied = [x for x in map(pi.copy_one, [(key, start + i, u) for i, u in enumerate(keep)]) if x]
    if copied:
        with pool.connection() as c:
            c.cursor().executemany(
                "insert into propai_project_images(project_key,position,object_name,bytes,content_type,source,source_url) "
                "values (%s,%s,%s,%s,%s,'web',%s) on conflict (project_key,position) do nothing",
                [(*x, found["official_url"]) for x in copied])
    return len(copied)


if __name__ == "__main__":
    with pool.connection() as c:
        todo = c.execute("""
            select p.key, p.name, p.master_project, p.developer from propai_projects p
            left join lateral (select count(*) n from propai_sales s where s.location_key_project = p.location_key
                               and s.instance_date >= now() - interval '12 months') s on true
            where not exists (select 1 from propai_project_images i where i.project_key = p.key and i.source = 'web')
            order by exists (select 1 from propai_project_images i where i.project_key = p.key), s.n desc, p.name""").fetchall()
    todo = todo[:limit] if limit else todo
    print(f"{len(todo)} projects to look up", flush=True)
    projects = images = empty_in_row = 0
    try:
        pi.token()  # fail fast if gcloud is logged out
        with ThreadPoolExecutor(workers) as threads:
            for n, got in enumerate(threads.map(one, todo), 1):
                projects += bool(got); images += got
                empty_in_row = 0 if got else empty_in_row + 1
                if empty_in_row >= 25:  # many developers have no official photo page; stop only if OpenAI stopped answering
                    try:
                        _ask("Reply with OK")
                    except Exception as exc:
                        sys.exit(f"stopped after 25 projects in a row with no photos ({n} checked): OpenAI failed: {exc}")
                    empty_in_row = 0
                if n % 25 == 0:
                    print(f"{n} checked, {projects} projects got {images} photos", flush=True)
    except pi.AuthError as error:
        sys.exit(f"STOPPED: {error}")
    print(f"done: {len(todo)} checked, {projects} projects got {images} photos")
