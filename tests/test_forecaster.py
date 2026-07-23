"""Lock in the decomposition result: the model must ANTICIPATE calendar events.

The single-model approach passed every accuracy metric while being wrong where it
mattered -- it forecast Ramadan by lagging it. These tests guard specifically against
that regression, on the transition days a WAPE average would wash out.
"""

import datetime as dt

import pytest

from app.core.features import build_features
from app.core.generate import generate
from app.models.forecaster import CalendarDecomposed, SeasonalNaive
from app.core.evaluation import backtest


@pytest.fixture(scope="module")
def feats():
    return build_features(generate(), horizon=1)


@pytest.fixture(scope="module")
def model(feats):
    return CalendarDecomposed().fit(feats)


def test_calendar_multipliers_have_correct_sign(model, feats):
    """Ramadan must halve croissants and multiply konafa, per the injected truth."""
    ramadan_day = feats[(feats["sku"] == "PASTRY_CROISSANT") & (feats["date"] == "2025-03-15")]
    croissant_mult = model.calendar_multiplier(ramadan_day)[0]
    assert croissant_mult < 0.75, f"croissant Ramadan multiplier {croissant_mult:.2f} not suppressed"

    konafa_day = feats[(feats["sku"] == "SWEET_KONAFA") & (feats["date"] == "2025-03-15")]
    konafa_mult = model.calendar_multiplier(konafa_day)[0]
    assert konafa_mult > 2.0, f"konafa Ramadan multiplier {konafa_mult:.2f} not amplified"


def test_forecast_drops_on_first_day_of_ramadan(model, feats):
    """The regression that started all this: day 1 of Ramadan must forecast DOWN.

    The single-model version forecast croissants UP on this day, tracking sales a few
    days late. The decomposed model applies the calendar effect immediately.
    """
    sub = feats[feats["sku"] == "PASTRY_CROISSANT"]
    before = model.predict(sub[sub["date"] == "2025-02-25"])[0]   # week before Ramadan
    day1 = model.predict(sub[sub["date"] == "2025-03-01"])[0]     # first day of Ramadan
    assert day1 < before * 0.8, (
        f"forecast did not drop on day 1 of Ramadan: {before:.0f} -> {day1:.0f}"
    )


def test_explanation_is_exact_and_signed(model, feats):
    ramadan_day = feats[(feats["sku"] == "PASTRY_CROISSANT") & (feats["date"] == "2025-03-15")]
    factors = model.explain(ramadan_day)
    ramadan = [f for f in factors if "Ramadan" in f["factor"]]
    assert ramadan, "Ramadan not attributed"
    assert ramadan[0]["direction"] == "decrease"


def test_decomposed_beats_seasonal_naive(feats):
    """The headline claim, guarded: the model must clear the naive floor on WAPE."""
    _, naive = backtest(feats, SeasonalNaive().fit_predict,
                        target="sales_qty", eval_target="true_demand",
                        n_folds=4, test_days=28, horizon=1)
    _, decomp = backtest(feats, CalendarDecomposed().fit_predict,
                        target="sales_qty", eval_target="true_demand",
                        n_folds=4, test_days=28, horizon=1)
    assert decomp["wape"] < naive["wape"], (
        f"decomposed WAPE {decomp['wape']:.3f} did not beat naive {naive['wape']:.3f}"
    )
    # Should be a decisive margin, not a coin-flip.
    assert (naive["wape"] - decomp["wape"]) / naive["wape"] > 0.15
