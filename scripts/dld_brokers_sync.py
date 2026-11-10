"""
Load every broker from Dubai Land Department's public broker search into public.propai_brokers.

    python scripts/dld_brokers_sync.py [--pages N]      # --pages N = only the first N pages (smoke test)

Idempotent (upsert by BRN), so re-running refreshes. Needs DLD_CONSUMER_ID in .env (the public key the DLD site's own
search uses). Polite by design: one request at a time with a pause. Fields the DLD feed lacks (status, transaction
history, awards detail) stay NULL; sign-in eligibility uses card_expiry_date instead of status.
"""
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database.postgres import _get_pool  # noqa: E402

URL = "https://gateway.dubailand.gov.ae/brokers/"
PAGE = int(os.getenv("DLD_PAGE_SIZE", "100"))   # ponytail: untested ceiling; lower it if the gateway returns fewer rows
PAUSE = float(os.getenv("DLD_PAUSE_SECONDS", "0.5"))

UPSERT = """
insert into propai_brokers (brn,name_en,name_ar,agency_name,agency_name_ar,email,mobile,landline,orn,card_rank,card_rank_id,
  office_rank,office_rank_id,card_issue_date,card_expiry_date,office_issue_date,office_expiry_date,photo_url,office_logo,
  awards_count,license_number,source,synced_at)
values (%(brn)s,%(name_en)s,%(name_ar)s,%(agency)s,%(agency_ar)s,%(email)s,%(mobile)s,%(landline)s,%(orn)s,%(card_rank)s,
  %(card_rank_id)s,%(office_rank)s,%(office_rank_id)s,%(ci)s,%(ce)s,%(oi)s,%(oe)s,%(photo)s,%(logo)s,%(awards)s,%(lic)s,'dld',now())
on conflict (brn) do update set name_en=excluded.name_en,name_ar=excluded.name_ar,agency_name=excluded.agency_name,
  agency_name_ar=excluded.agency_name_ar,email=excluded.email,mobile=excluded.mobile,landline=excluded.landline,
  orn=excluded.orn,card_rank=excluded.card_rank,card_rank_id=excluded.card_rank_id,office_rank=excluded.office_rank,
  office_rank_id=excluded.office_rank_id,card_issue_date=excluded.card_issue_date,card_expiry_date=excluded.card_expiry_date,
  office_issue_date=excluded.office_issue_date,office_expiry_date=excluded.office_expiry_date,photo_url=excluded.photo_url,
  office_logo=excluded.office_logo,awards_count=excluded.awards_count,license_number=excluded.license_number,
  source='dld',synced_at=now()
"""


def phone(v):  # DLD sends "971|50-6444437" -> "+971506444437"
    if not v or "|" not in v:
        return None
    cc, num = v.split("|", 1)
    digits = "".join(ch for ch in num if ch.isdigit())
    return f"+{cc}{digits}" if digits else None


def date(v):
    return v[:10] if v and not v.startswith("0001") else None


def row(r):
    clean = lambda v: (v or "").strip() or None
    return dict(brn=str(r["CardNumber"]), name_en=clean(r["CardHolderNameEn"]), name_ar=clean(r["CardHolderNameAr"]),
                agency=clean(r["OfficeNameEn"]), agency_ar=clean(r["OfficeNameAr"]),
                email=(clean(r["CardHolderEmail"]) or "").lower() or None,
                mobile=phone(r["CardHolderMobile"]), landline=phone(r["CardHolderPhone"]), orn=clean(r["RealEstateNumber"]),
                card_rank=clean(r["CardRank"]), card_rank_id=r["CardRankId"], office_rank=clean(r["OfficeRank"]),
                office_rank_id=r["OfficeRankId"], ci=date(r["CardIssueDate"]), ce=date(r["CardExpiryDate"]),
                oi=date(r["OfficeIssueDate"]), oe=date(r["OfficeExpiryDate"]), photo=clean(r["CardHolderPhoto"]),
                logo=clean(r["OfficeLogo"]), awards=r["AwardsCount"], lic=clean(r["LicenseNumber"]) if r["LicenseNumber"] != "-" else None)


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--pages") + 1]) if "--pages" in sys.argv else None
    pool, total, index, done = _get_pool(), None, 0, 0
    with httpx.Client(timeout=60, headers={"accept": "application/json", "user-agent": "acot-ai-sync/1.0"}) as client:
        while limit is None or index < limit:
            resp = client.get(URL, params={"sortCriteria": 2, "pageIndex": index, "pageSize": PAGE,
                                           "consumer-id": os.environ["DLD_CONSUMER_ID"]})
            resp.raise_for_status()
            body = resp.json()
            rows = body if isinstance(body, list) else next(v for v in body.values() if isinstance(v, list))
            if not rows:
                break
            total = rows[0]["TotalRowsCount"]
            with pool.connection() as conn:
                conn.cursor().executemany(UPSERT, [row(r) for r in rows])
            done += len(rows)
            index += 1
            print(f"page {index}: {done}/{total}")
            if done >= total:
                break
            time.sleep(PAUSE)
