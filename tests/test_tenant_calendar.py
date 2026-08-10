"""Per-restaurant calendar: a learned level must land on the DATE it was asked for.

The bug these pin down: `/integration/restomind/production-plan` answered every date of
the year with the same quantity for any product outside the trained catalogue, because
the registry's learned level is a quiet-day mean and nothing re-applied seasonality to
it after the rule-based layer was removed. Meanwhile `/forecast/daily-batch` -- the
trained path, same underlying sales -- moved with the calendar. Two endpoints, one
dataset, two different answers.
"""

import datetime as dt
import os

from fastapi.testclient import TestClient
import pytest

from app.api.main import app
from app.integration.registry import MIN_DAYS_FOR_LEARNED, RestaurantRegistry
from app.integration.restomind import ProductInput

os.environ.setdefault("REGISTRY_STORE", "")   # in-memory registry for tests

# Egypt's weekend. `date.weekday()` is Monday-zero, so Friday is 4 and Saturday 5.
_WEEKEND = {4, 5}

# Enough calendar days that the quiet (non-weekend, non-event) subset clears the
# learning threshold with room to spare.
_SEED_DAYS = int(MIN_DAYS_FOR_LEARNED * 2.4)

# A Tuesday and the Friday of the same week, well clear of Ramadan and any holiday.
ORDINARY_DAY = "2025-05-13"
WEEKEND_DAY = "2025-05-16"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _shaped_sales(product_id, n, weekday_level, weekend_level, start="2024-06-03"):
    """Sales with a real weekday shape: weekends genuinely busier than midweek.

    Deliberately not noise-free -- a fit that only survives on perfectly clean data
    would not survive a real POS export either.
    """
    import random
    rng = random.Random(11)
    d0 = dt.date.fromisoformat(start)
    rows = []
    for i in range(n):
        day = d0 + dt.timedelta(days=i)
        level = weekend_level if day.weekday() in _WEEKEND else weekday_level
        rows.append({"date": day.isoformat(), "productId": product_id,
                     "salesQty": max(0, int(rng.gauss(level, level * 0.05)))})
    return rows


def _seed(client, restaurant_id, product, weekday_level=100, weekend_level=180):
    client.post("/integration/restomind/ingest", json={
        "restaurantId": restaurant_id,
        "records": _shaped_sales(product["productId"], _SEED_DAYS, weekday_level, weekend_level),
        "products": [product],
    })


def _plan(client, restaurant_id, product, date):
    return client.post("/integration/restomind/production-plan", json={
        "restaurantId": restaurant_id, "date": date, "products": [product],
    }).json()["items"][0]


PRODUCT = {"productId": "TC_P1", "title": "فطير مشلتت", "category": "معجنات",
           "price": 30.0, "unitCost": 12.0, "freshnessWindow": 2, "avgDailySales": 100}


# -- the reported bug ------------------------------------------------------------------


def test_plan_differs_between_a_weekday_and_a_weekend_once_learned(client):
    """The headline fix: the same product, two dates, two quantities.

    Before this the plan repeated one number for every date, so a bakery that really
    sells 80% more on a Friday was told to make the Tuesday quantity.
    """
    rid = "TC_R1"
    _seed(client, rid, PRODUCT)

    weekday = _plan(client, rid, PRODUCT, ORDINARY_DAY)
    weekend = _plan(client, rid, PRODUCT, WEEKEND_DAY)

    assert weekday["levelSource"] == "learned_from_sales"
    assert weekend["recommendedQty"] > weekday["recommendedQty"], (
        "a learned product's plan must move with the calendar, not repeat one level"
    )
    # The seeded shape is +80% on the weekend. Ridge shrinks a little, so the recovered
    # ratio lands just under; a band this tight is what catches the effect being HALVED
    # rather than merely present -- which is what a stray filter does to it silently.
    ratio = weekend["recommendedQty"] / weekday["recommendedQty"]
    assert 1.6 < ratio < 2.0, f"weekend/weekday ratio {ratio:.2f} does not reflect the data"


def test_the_multiplier_is_neutral_on_an_ordinary_day(client):
    """`level x multiplier` must reproduce the level on the days the level came from.

    The multiplier is fitted against a rolling clean baseline, not against the quiet-day
    mean the registry stores, so it is normalised before use. Without that step every
    quantity would shift by a constant the moment this feature shipped.
    """
    rid = "TC_R2"
    _seed(client, rid, PRODUCT)

    item = _plan(client, rid, PRODUCT, ORDINARY_DAY)
    assert item["calendarMultiplier"] == pytest.approx(1.0, abs=0.15)
    assert item["recommendedQty"] == pytest.approx(item["baseDailyLevel"], rel=0.2)


def test_the_plan_explains_the_date(client):
    """A quantity that moves without saying why is a quantity a manager overrides."""
    rid = "TC_R3"
    _seed(client, rid, PRODUCT)

    weekend = _plan(client, rid, PRODUCT, WEEKEND_DAY)
    assert weekend["factors"], "a calendar-moved quantity must carry its drivers"
    assert any(f["factor"] == "Day of week" for f in weekend["factors"])
    assert weekend["baseDailyLevel"] != weekend["recommendedQty"], (
        "baseDailyLevel is the ordinary-day level, reported separately from the plan"
    )


def test_owner_estimate_stays_flat_across_dates(client):
    """No history, no calendar. An estimate must not be given a shape it never earned.

    This is the line between learning and guessing -- the hand-written category
    multipliers were removed (HANDOFF.md §8) precisely so this case stays honest.
    """
    product = {"productId": "TC_FLAT", "title": "توست", "price": 10.0,
               "unitCost": 4.0, "freshnessWindow": 2, "avgDailySales": 60}
    weekday = _plan(client, "TC_R4", product, ORDINARY_DAY)
    weekend = _plan(client, "TC_R4", product, WEEKEND_DAY)

    assert weekday["levelSource"] == "owner_estimate"
    assert weekday["calendarMultiplier"] == 1.0
    assert weekday["recommendedQty"] == weekend["recommendedQty"]
    assert weekday["factors"] == []


def test_weekly_prediction_has_a_shape_not_seven_identical_days(client):
    """`/predict` feeds RestoMind's predictions collection; its breakdown must vary too."""
    rid = "TC_R5"
    _seed(client, rid, PRODUCT)

    out = client.post("/integration/restomind/predict", json={
        "restaurantId": rid, "productId": PRODUCT["productId"], "title": PRODUCT["title"],
        "category": PRODUCT["category"], "avgDailySales": PRODUCT["avgDailySales"],
        "targetWeek": ORDINARY_DAY,
    }).json()

    quantities = [d["predictedQuantity"] for d in out["dailyBreakdown"]]
    assert len(set(quantities)) > 1, "a learned week must not be one number seven times"
    assert out["predictedOrders"] == sum(quantities)
    assert out["featuresUsed"]["calendarSource"] == "learned_from_sales"
    assert out["factors"]


def test_status_says_whether_a_product_is_calendar_aware(client):
    """Otherwise the only way to find out is to call the plan twice and compare."""
    rid = "TC_R6"
    _seed(client, rid, PRODUCT)

    status = client.get(f"/integration/restomind/status/{rid}").json()
    item = next(i for i in status["items"] if i["productId"] == PRODUCT["productId"])
    assert item["calendarAware"] is True


# -- the filter that used to erase the effect ------------------------------------------


def test_a_consistent_weekend_is_not_flagged_as_an_outlier():
    """A recurring weekly pattern is the most predictable thing in this data.

    `flag_outliers` spares calendar-explained spikes but did not count the weekend as
    one. The rolling median over a 5-weekday / 2-weekend mix sits at the weekday level,
    so a real Friday uplift reads as a ~10-MAD deviation: every weekend day was dropped
    before `CalendarEffects` ever saw it, and the fitted weekday profile came out flat.
    The tighter the business's weekly rhythm, the more completely it was erased.
    """
    import pandas as pd

    from app.core.features import build_features

    rows = _shaped_sales("P1", _SEED_DAYS, weekday_level=100, weekend_level=180)
    frame = pd.DataFrame({
        "date": pd.to_datetime([r["date"] for r in rows]),
        "sku": "P1",
        "sales_qty": [float(r["salesQty"]) for r in rows],
        "production_qty": float("nan"),
        "leftover_qty": float("nan"),
    })

    features = build_features(frame, horizon=1)
    weekend = features["date"].dt.weekday.isin(_WEEKEND)
    assert weekend.sum() > 0
    assert features.loc[weekend, "is_outlier"].sum() == 0, (
        "the weekend is a calendar event, not an anomaly"
    )


# -- economics -------------------------------------------------------------------------


def test_zero_unit_cost_is_treated_as_unknown_not_as_free(client):
    """A zero cost makes q* exactly 1.0, planning every such product at its upper bound.

    RestoMind's registry is full of `unitCost: 0` against real prices -- an unset column,
    not a free product -- so that was a silent, permanent +10% on quantity.
    """
    free = {"productId": "TC_ZERO", "title": "عيش", "price": 5.0,
            "unitCost": 0, "freshnessWindow": 2, "avgDailySales": 100}
    item = _plan(client, "TC_R7", free, ORDINARY_DAY)
    assert item["recommendedQty"] == 100, "zero cost must not push the plan to the upper bound"
    assert item["upperBound"] == 110


# -- persistence -----------------------------------------------------------------------


def test_a_learned_calendar_survives_a_restart(tmp_path):
    """The fit is stored as plain floats beside the levels, never as a pickle.

    A calendar that had to be refitted on every boot would either make startup slow or
    quietly return the flat behaviour until the next ingest.
    """
    import pandas as pd

    store = tmp_path / "registry.json"
    rows = pd.DataFrame(_shaped_sales("P1", _SEED_DAYS, 100, 180))
    product = ProductInput(product_id="P1", title="فطير", price=30.0, unit_cost=12.0)

    first = RestaurantRegistry(persist_path=store)
    first.ingest("R1", rows, [product])
    before = first.get("R1").calendar
    assert before is not None and "P1" in before

    day = dt.date.fromisoformat(WEEKEND_DAY)
    reopened = RestaurantRegistry(persist_path=store).get("R1").calendar
    assert reopened is not None
    assert reopened.multiplier("P1", day) == pytest.approx(before.multiplier("P1", day))
    assert reopened.explain("P1", day) == before.explain("P1", day)

    # Plain JSON -- no estimator objects on a path that is read back at startup.
    import json
    stored = json.loads(store.read_text(encoding="utf-8"))["R1"]["calendar"]
    assert set(stored) == {"columns", "coefficients", "intercepts", "references"}
    assert all(isinstance(c, float) for c in stored["coefficients"]["P1"])


def test_an_unlearned_product_gets_a_neutral_multiplier(tmp_path):
    """Products the fit never saw must fall through to the old flat behaviour exactly."""
    import pandas as pd

    rows = pd.DataFrame(_shaped_sales("P1", _SEED_DAYS, 100, 180))
    reg = RestaurantRegistry()
    reg.ingest("R1", rows, [ProductInput(product_id="P1", title="فطير")])

    calendar = reg.get("R1").calendar
    assert "P_UNSEEN" not in calendar
    assert calendar.multiplier("P_UNSEEN", dt.date.fromisoformat(WEEKEND_DAY)) == 1.0
    assert calendar.explain("P_UNSEEN", dt.date.fromisoformat(WEEKEND_DAY)) == []


def test_too_little_history_learns_no_calendar():
    """Below the level threshold there is no calendar either -- nothing to fit it from."""
    import pandas as pd

    rows = pd.DataFrame(_shaped_sales("P1", 30, 100, 180))
    reg = RestaurantRegistry()
    reg.ingest("R1", rows, [ProductInput(product_id="P1", title="فطير")])

    assert reg.get("R1").calendar is None
