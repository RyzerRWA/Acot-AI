"""Live Prop-AI API figures our synced tables do not hold: matched rental yield and multi-year price growth.

Two calls per area (metrics + yearly trends), cached in-process for 12 hours, so a busy day costs a
handful of requests. Any failure returns None and callers fall back to the database figures.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "propai_live_cache.json"
_TTL_SECONDS = 12 * 3600


def _load_cache() -> dict[str, list]:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


# Persisted so a restart or a slow Prop-AI call (10-25s under load) never flips an area's yield back to our
# own DB figure: the same question keeps the same number.
_CACHE: dict[str, list] = _load_cache()


def _save_cache() -> None:
    try:
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(_CACHE)))
        tmp.replace(_CACHE_FILE)  # atomic: a crash mid-write never leaves a broken cache
    except (OSError, RuntimeError) as exc:
        print(f"Prop-AI cache not saved: {exc}")


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=os.getenv("PROPAI_BASE_URL", "https://api.prop-ai.com"),
        timeout=30,
        headers={"x-api-key": os.environ["PROPAI_API_KEY"], "Accept": "application/json"},
    )


def area_returns(official_area: str) -> dict | None:
    """Apartment rental yield, yearly price change and 3-year price growth for one DLD area.

    rental_yield_pct: Prop-AI's 12-month rental yield (rents and prices matched within the area).
    growth_3y_pct: compound yearly growth of the 12-month median price per sq ft over the last 3 years,
    computed here from the yearly series (Prop-AI's own CAGR field did not match the series it returns).
    """
    key = str(official_area or "").strip().upper()
    if not key or not os.getenv("PROPAI_API_KEY"):
        return None
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL_SECONDS:
        return hit[1]
    try:
        with _client() as client:
            with ThreadPoolExecutor(2) as pool:
                metrics = pool.submit(client.get, f"/data/metrics/areas/{key}", params={"property_type": "Apartment"})
                trends = pool.submit(client.get, f"/data/trends/areas/{key}", params={
                    "property_type": "Apartment", "time_interval": "Yearly", "timeperiod": 4,
                    "metric_type": "Sales", "metric_calculation": 12,
                })
                metrics, trends = metrics.result().json(), trends.result().json()
    except Exception as exc:
        print(f"Prop-AI returns lookup failed for {key}: {exc}")
        return hit[1] if hit else None  # last good figure, however old; the next question retries

    series = sorted(
        (int(year), float(period.get("sales_pricepersqft_median_p12m") or 0))
        for year, period in (trends.get("periods") or {}).items()
        if str(year).isdigit()
    )
    series = [(year, psf) for year, psf in series if psf > 0]
    growth = None
    if len(series) >= 2:
        (first_year, first), (last_year, last) = series[0], series[-1]
        growth = round(100 * ((last / first) ** (1 / (last_year - first_year)) - 1), 1)
    yld = metrics.get("rental_yield_p12m")
    result = {
        "rental_yield_pct": round(100 * yld, 1) if yld else None,
        "price_change_yoy_pct": round(metrics["price_change_yoy"], 1) if metrics.get("price_change_yoy") is not None else None,
        "growth_3y_pct": growth,
        "growth_years": (series[0][0], series[-1][0]) if len(series) >= 2 else None,
        "median_rent": metrics.get("annual_rent_median_p12m"),
        "median_price": metrics.get("sales_price_median_p12m"),
    }
    if result["rental_yield_pct"] is None and growth is None:
        result = None
    _CACHE[key] = [time.time(), result]
    _save_cache()
    return result
