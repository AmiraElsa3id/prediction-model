"""The cold-start -> trained-model transition after the rule layer was removed.

This is the behaviour the user specifically asked for: below the training threshold an
item has NO forecast (no rule-based fallback exists anymore, see HANDOFF.md §8). Once it
has enough history its own trained model serves it. While the model is present but a
major calendar event was never in the training window, the model still forecasts -- it
just marks the day `low` confidence and widens the interval, because there is no rule
path to route to anymore.
"""

import datetime as dt

import pandas as pd
import pytest

from app.core.generate import generate
from app.models.service import ForecastService, ModelNotReadyError

RAMADAN_DAY = dt.date(2025, 3, 15)
NORMAL_DAY = dt.date(2024, 10, 15)


@pytest.fixture(scope="module")
def full():
    df = generate()
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_cold_start_has_no_forecast_at_all():
    svc = ForecastService(train_threshold=90).start_cold()
    st = svc.status()
    assert st["items_untrained"] == len(st["items"])
    assert st["items_trained"] == 0
    # With the rules gone there is NO forecast, not a guessed number.
    with pytest.raises(ModelNotReadyError):
        svc.forecast("PASTRY_CROISSANT", NORMAL_DAY)


def test_item_switches_to_model_after_threshold(full):
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=120) + pd.Timedelta(hours=1)])
    # A normal day uses the model.
    assert svc.forecast("PASTRY_CROISSANT", NORMAL_DAY).source == "batch"


def test_unseen_event_stays_on_model_with_low_confidence(full):
    """120 days without a Ramadan must not invent a rule-based Ramadan forecast.

    The model still forecasting the date, but marks it low-confidence and widens the
    interval -- there is no rule layer left to hand the task to.
    """
    d0 = full["date"].min()  # 2023-07-01; +120d has no Ramadan
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=120) + pd.Timedelta(hours=1)])
    assert "is_ramadan" not in svc.observed_events
    # Both dates now run through the model...
    ramadan = svc.forecast("PASTRY_CROISSANT", RAMADAN_DAY)
    normal = svc.forecast("PASTRY_CROISSANT", NORMAL_DAY)
    assert ramadan.source == "batch"
    assert normal.source == "batch"
    # ...but the unseen-event day is flagged with low confidence.
    assert ramadan.confidence == "low"
    assert normal.confidence == "medium"


def test_model_takes_over_event_once_seen(full):
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=400)])
    assert "is_ramadan" in svc.observed_events
    r = svc.forecast("PASTRY_CROISSANT", RAMADAN_DAY)
    assert r.source == "batch"
    assert r.confidence == "high"   # the model has now trained through a Ramadan


def test_ingest_reports_transitions(full):
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    out = svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=120) + pd.Timedelta(hours=1)])
    assert out["model_retrained"] is True
    assert len(out["newly_switched_to_ml"]) == full["sku"].nunique()


def test_forecast_all_raises_model_not_ready_while_training():
    svc = ForecastService(train_threshold=90).start_cold()
    with pytest.raises(ModelNotReadyError):
        svc.forecast_all(NORMAL_DAY, ["PASTRY_CROISSANT"])


def test_start_cold_wipes_previously_seen_events(full):
    """start_cold() must fully reset, including observed_events.

    Before this fix the reset cleared models/observed_days but left observed_events
    from an earlier training behind, so the unseen-event safety net wrongly trusted
    the model on events this run never trained on.
    """
    d0 = full["date"].min()
    svc = ForecastService(train_threshold=90).start_cold()
    svc.ingest(full[full["date"] < d0 + pd.Timedelta(days=400)])
    assert "is_ramadan" in svc.observed_events
    svc.start_cold()
    assert svc.observed_events == set()
    assert svc.observed_days == {}


def test_data_source_is_honest_about_real_ingest(full):
    """/health provenance: never call generated data SIMULATED once real actuals
    have been appended, and never claim SIMULATED for a pure cold-start+ingest."""
    d0 = full["date"].min()

    cold = ForecastService(train_threshold=90).start_cold()
    assert cold.data_source() == "NONE"
    cold.ingest(full[full["date"] < d0 + pd.Timedelta(days=10)])
    assert cold.data_source() == "REAL"

    mixed = ForecastService(train_threshold=90).train()
    assert mixed.data_source() == "SIMULATED"
    mixed.ingest(full[full["date"] < d0 + pd.Timedelta(days=10)])
    assert mixed.data_source() == "SIMULATED + REAL"


def test_status_reports_progress_towards_trained():
    """status() keeps driving a per-item progress bar, now toward 'trained'."""
    svc = ForecastService(train_threshold=90).start_cold()
    item = svc.status()["items"][0]
    assert item["mode"] == "untrained"
    assert item["progress"] == 0.0
    assert item["days_until_switch"] == 90