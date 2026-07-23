"""The cold-start -> trained-model transition, and event-aware routing.

This is the behaviour the user specifically asked for: rule-based until an item has
enough history, then the trained model -- but never the model for a calendar event it
has not yet lived through.
"""

import datetime as dt

import pandas as pd
import pytest

from app.core.generate import generate
from app.models.service import ForecastService

RAMADAN_DAY = dt.date(2025, 3, 15)
NORMAL_DAY = dt.date(2024, 10, 15)


@pytest.fixture(scope="module")
def full():
    df = generate()
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_cold_start_is_entirely_rule_based():
    svc = ForecastService(train_threshold=90).start_cold()
    st = svc.status()
    assert st["items_rule_based"] == len(st["items"])
    assert st["items_trained"] == 0
    # It still returns a usable forecast with no data at all.
    r = svc.forecast("PASTRY_CROISSANT", NORMAL_DAY)
    assert r.source == "rule_based"
    assert r.quantity > 0


def test_rule_based_knows_ramadan_without_any_data():
    svc = ForecastService(train_threshold=90).start_cold()
    normal = svc.forecast("PASTRY_CROISSANT", NORMAL_DAY).quantity
    ramadan = svc.forecast("PASTRY_CROISSANT", RAMADAN_DAY).quantity
    assert ramadan < normal  # croissants fall in Ramadan, from rules alone


def test_item_switches_to_model_after_threshold(full):
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=120)])
    # A normal day (an event-type the window contained) uses the model.
    assert svc.forecast("PASTRY_CROISSANT", NORMAL_DAY).source == "batch"


def test_model_defers_to_rules_for_unseen_event(full):
    """The key safety net: 120 days without a Ramadan must NOT forecast Ramadan by model."""
    d0 = full["date"].min()  # 2023-07-01; +120d has no Ramadan
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=120)])
    assert "is_ramadan" not in svc.observed_events
    # Item is trained, but the Ramadan day still routes to the rules.
    assert svc.forecast("PASTRY_CROISSANT", RAMADAN_DAY).source == "rule_based"
    # ...while an ordinary day uses the model.
    assert svc.forecast("PASTRY_CROISSANT", NORMAL_DAY).source == "batch"


def test_model_takes_over_event_once_seen(full):
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=400)])  # includes Ramadan 2024
    assert "is_ramadan" in svc.observed_events
    assert svc.forecast("PASTRY_CROISSANT", RAMADAN_DAY).source == "batch"


def test_ingest_reports_transitions(full):
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    out = svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=120)])
    assert out["model_retrained"] is True
    assert len(out["newly_switched_to_ml"]) == full["sku"].nunique()
