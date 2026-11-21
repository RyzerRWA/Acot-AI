"""Same question, same number: every sales statistic shares one filter and one default window."""
import pathlib
import re

from app.database.postgres import CLEAN_SALE
from app.listings.tools import _sales_where

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_sales_search_defaults_to_last_12_months():
    where, _, used = _sales_where({})
    assert CLEAN_SALE in where
    assert "current_date - 365" in where
    assert used["last_days"] == 365


def test_explicit_period_overrides_default():
    where, _, used = _sales_where({"date_from": "2020-01-01"})
    assert "current_date - 365" not in where and used["date_from"] == "2020-01-01"


def test_no_ad_hoc_sale_filters_left():
    # A private price floor or a missing outlier filter makes one answer path disagree with the rest.
    for path in ("app", "run.py"):
        for file in (ROOT / path).rglob("*.py") if path == "app" else [ROOT / path]:
            text = file.read_text()
            assert not re.search(r"sale_price between \d+ and", text), file
