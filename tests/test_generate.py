"""Assert the synthetic generator actually injects the effects we claim.

If these fail, every downstream accuracy number is meaningless -- the recovery test
would be checking that the model finds an effect that was never put there.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.egypt_calendar import CALENDAR
from app.core.generate import generate
from app.core.items import BY_SKU


@pytest.fixture(scope="module")
def data():
    df = generate(start="2023-07-01", end="2025-06-30", seed=7)
    cal = CALENDAR.feature_frame(df["date"].min(), df["date"].max())
    cal["date"] = pd.to_datetime(cal["date"])
    return df.merge(cal, on="date")


def _ratio(df, sku, flag):
    s = df[df["sku"] == sku]
    on = s[s[flag] == 1]["true_demand"].mean()
    off = s[s[flag] == 0]["true_demand"].mean()
    return on / off


# -- structure ----------------------------------------------------------------------


def test_covers_two_years_of_all_items(data):
    assert data["sku"].nunique() == len(BY_SKU)
    span_days = (data["date"].max() - data["date"].min()).days
    assert 720 <= span_days <= 732


def test_stock_identity_mostly_holds(data):
    """production - sales = leftover, except for the miscounts we deliberately inject."""
    ok = data["production_qty"] - data["sales_qty"] == data["leftover_qty"]
    assert ok.mean() > 0.98, "too many rows violate the stock identity"
    assert ok.mean() < 1.0, "no miscounts injected -- cleaning would be untested"


def test_sales_never_exceed_production(data):
    assert (data["sales_qty"] <= data["production_qty"]).all()


def test_sales_are_censored_by_production(data):
    """The core data problem: on stockout days, sales understate true demand."""
    so = data[data["is_stockout"] == 1]
    assert len(so) > 0
    assert (so["true_demand"] > so["sales_qty"]).all()
    # Leftover is zero on stockout days apart from the injected inventory miscounts.
    assert (so["leftover_qty"] == 0).mean() > 0.98


def test_closed_days_are_missing_rows_not_zeros(data):
    """Eid day 1 is dropped entirely; a zero would wrongly teach 'no demand'."""
    closed = data[data["eid_fitr_day_index"] == 1]
    assert len(closed) == 0


def test_realistic_waste_and_stockout_rates(data):
    """The manager should both waste and stock out -- that's the problem we're selling against."""
    waste_rate = data["leftover_qty"].sum() / data["production_qty"].sum()
    assert 0.10 < waste_rate < 0.35, f"implausible waste rate {waste_rate:.1%}"
    assert 0.05 < data["is_stockout"].mean() < 0.40


# -- injected effects ---------------------------------------------------------------


def test_weekend_effects_match_catalogue(data):
    """Fri/Sat uplift should be recoverable within noise for each item."""
    for sku, item in BY_SKU.items():
        if item.seasonal_only:
            continue
        observed = _ratio(data, sku, "is_weekend")
        assert observed == pytest.approx(item.weekend_mult, rel=0.25), (
            f"{sku}: weekend effect {observed:.2f} vs injected {item.weekend_mult}"
        )


def test_ramadan_suppresses_breakfast_and_lifts_desserts(data):
    """The signature Egyptian effect: daylight fasting inverts the product mix."""
    assert _ratio(data, "PASTRY_CROISSANT", "is_ramadan") < 0.7
    assert _ratio(data, "SAVOURY_SANDWICH", "is_ramadan") < 0.6
    assert _ratio(data, "SWEET_KONAFA", "is_ramadan") > 3.0
    assert _ratio(data, "SWEET_BASBOUSA", "is_ramadan") > 1.6


def test_kahk_is_seasonal_and_not_baked_year_round(data):
    kahk = data[data["sku"] == "SEASONAL_KAHK"]
    produced_days = (kahk["production_qty"] > 0).sum()
    assert 10 < produced_days < 120, "kahk should run only around Eid"
    assert _ratio(data, "SEASONAL_KAHK", "is_kahk_window") > 20


def test_sham_el_nessim_spikes_feteer(data):
    assert _ratio(data, "SAVOURY_FETEER", "is_sham_el_nessim") > 1.5


def test_school_term_lifts_sandwiches(data):
    assert _ratio(data, "SAVOURY_SANDWICH", "is_school_term") > 1.2


def test_generation_is_reproducible():
    a = generate(start="2024-01-01", end="2024-03-31", seed=99)
    b = generate(start="2024-01-01", end="2024-03-31", seed=99)
    pd.testing.assert_frame_equal(a, b)


# -- economics ----------------------------------------------------------------------


def test_newsvendor_quantile_is_item_specific():
    """q* must vary by item -- a single global service level is wrong for both ends.

    Low-markup same-day staples should sit below 0.5 (waste hurts more than a missed
    sale); long-shelf-life items should sit well above it (leftovers keep, stockouts
    don't). If these ever collapse to one value, the quantile framing is doing nothing.
    """
    quantiles = {sku: i.newsvendor_quantile for sku, i in BY_SKU.items()}
    assert all(0.0 < q < 1.0 for q in quantiles.values())
    assert max(quantiles.values()) - min(quantiles.values()) > 0.2

    # Long shelf life pushes q* toward 1: leftovers keep, so err on making more.
    assert quantiles["DRY_PETITFOUR"] > 0.85, "14-day shelf life should push q* high"
    assert quantiles["SEASONAL_KAHK"] > 0.85, "30-day shelf life should push q* high"
    # A same-day pastry with low goodwill cost is the most conservative case: waste is a
    # total write-off and running out barely dents loyalty.
    assert quantiles["PASTRY_DONUT"] < quantiles["DRY_PETITFOUR"]
    assert quantiles["PASTRY_DONUT"] < quantiles["SEASONAL_KAHK"]


def test_spoilage_severity_tracks_shelf_life():
    assert BY_SKU["BREAD_BALADI"].spoilage_severity == 1.0
    assert BY_SKU["DRY_PETITFOUR"].spoilage_severity < 0.5
    assert (
        BY_SKU["SEASONAL_KAHK"].spoilage_severity
        <= BY_SKU["PASTRY_CROISSANT"].spoilage_severity
    )
