"""RestoMind integration bridge: accepts their product shapes, returns basis-level output.

The rule-based calendar layer is gone (see HANDOFF.md §8). The bridge forecasts from a
basis daily level -- a learned level from this restaurant's real sales, else the owner's
estimate -- and answers a product with no basis at all with a "still training" message
rather than a guessed number.
"""

from fastapi.testclient import TestClient
import pytest

from app.api.main import app
from app.integration.restomind import to_business_time

import datetime as dt
import os

os.environ.setdefault("REGISTRY_STORE", "")   # in-memory registry for tests


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


PRODUCTS = [
    {"productId": "p1", "title": "كرواسون", "category": "معجنات", "price": 18, "freshnessWindow": 2, "avgDailySales": 180},
    {"productId": "p2", "title": "كنافة", "category": "حلويات شرقية", "price": 45, "freshnessWindow": 2, "avgDailySales": 40},
    {"productId": "p3", "title": "عيش فينو", "category": "مخبوزات", "price": 2.5, "freshnessWindow": 1, "avgDailySales": 420},
]


def test_production_plan_returns_one_row_per_product(client):
    r = client.post("/integration/restomind/production-plan",
                    json={"restaurantId": "R1", "date": "2025-02-11", "products": PRODUCTS})
    assert r.status_code == 200
    b = r.json()
    assert len(b["items"]) == len(PRODUCTS)
    assert b["totalRecommendedQty"] == sum(i["recommendedQty"] for i in b["items"])
    # With no ingested sales the plan is the owner's estimate.
    assert all(i["levelSource"] == "owner_estimate" for i in b["items"])
    assert all(i["trainingMessage"] is None for i in b["items"])


def test_production_plan_is_basis_level_not_a_calendar_guess(client):
    """Post-rule, the plan is the basis level: no category multiplier, date-independent."""
    normal = client.post("/integration/restomind/production-plan",
                         json={"restaurantId": "R1", "date": "2025-02-11", "products": PRODUCTS}).json()
    ramadan = client.post("/integration/restomind/production-plan",
                         json={"restaurantId": "R1", "date": "2025-03-15", "products": PRODUCTS}).json()
    n = {i["productId"]: i["recommendedQty"] for i in normal["items"]}
    rm = {i["productId"]: i["recommendedQty"] for i in ramadan["items"]}
    assert n == rm, "the removed rule layer must not move the plan by calendar anymore"


def test_production_plan_with_no_basis_returns_training_message(client):
    """No owner estimate, no learned level -> 'still training', not a guess."""
    r = client.post("/integration/restomind/production-plan",
                    json={"restaurantId": "R1", "date": "2025-02-11",
                          "products": [{"productId": "p_new", "title": "صنف جديد",
                                        "price": 30, "freshnessWindow": 2}]})
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item["recommendedQty"] == 0
    assert item["trainingMessage"]
    assert item["source"] == "training"


def test_production_plan_uses_trained_model_for_sku_product(client):
    """A product with a trained catalogue SKU gets the real model, not the flat level.

    The trained CalendarDecomposed model is calendar-aware: the plan for the same
    product must differ between a normal day and Ramadan, and must agree exactly with
    /forecast/daily (they are the same call).
    """
    product = {"productId": "p_croissant", "title": "كرواسون", "category": "معجنات",
               "price": 18, "freshnessWindow": 2, "avgDailySales": 180,
               "sku": "PASTRY_CROISSANT"}
    plan = client.post("/integration/restomind/production-plan",
                       json={"restaurantId": "R1", "date": "2025-02-11",
                             "products": [product]}).json()["items"][0]
    assert plan["levelSource"] == "trained_model"
    assert plan["source"] == "batch"
    assert plan["trainingMessage"] is None

    daily = client.post("/forecast/daily", json={"sku": "PASTRY_CROISSANT",
                                                 "date": "2025-02-11"}).json()
    assert plan["recommendedQty"] == daily["recommended_quantity"]
    assert plan["lowerBound"] == daily["lower_bound"]
    assert plan["upperBound"] == daily["upper_bound"]
    assert plan["confidence"] == daily["confidence"]

    ramadan = client.post("/integration/restomind/production-plan",
                          json={"restaurantId": "R1", "date": "2025-03-15",
                                "products": [product]}).json()["items"][0]
    assert ramadan["recommendedQty"] != plan["recommendedQty"], (
        "a trained product's plan must move with the calendar (Ramadan)"
    )


def test_production_plan_unknown_sku_falls_back_to_owner_estimate(client):
    """An unusable SKU must not break the plan; it degrades to the basis level."""
    product = {"productId": "p_flat", "title": "توست", "category": "مخبوزات",
               "price": 10, "freshnessWindow": 1, "avgDailySales": 100,
               "sku": "NOT_A_THING"}
    r = client.post("/integration/restomind/production-plan",
                    json={"restaurantId": "R1", "date": "2025-02-11",
                          "products": [product]})
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item["levelSource"] == "owner_estimate"
    assert item["recommendedQty"] == 100


def test_surplus_offers_flags_risk_and_writes_arabic_copy(client):
    stock = [{**p, "currentStock": cs} for p, cs in zip(PRODUCTS, [40, 25, 150])]
    r = client.post("/integration/restomind/surplus-offers",
                    json={"restaurantId": "R1", "stock": stock,
                          "timestamp": "2025-02-11T19:30:00", "closeHour": 22})
    assert r.status_code == 200
    items = r.json()["itemsAtRisk"]
    assert len(items) > 0
    for it in items:
        assert 5 <= it["suggestedDiscountPct"] <= 70
        assert it["offerCopyAr"] and it["title"] in it["offerCopyAr"]


def test_surplus_quiet_in_the_morning(client):
    stock = [{**PRODUCTS[0], "currentStock": 40}]
    r = client.post("/integration/restomind/surplus-offers",
                    json={"restaurantId": "R1", "stock": stock,
                          "timestamp": "2025-02-11T08:00:00", "closeHour": 22})
    assert r.json()["itemsAtRisk"] == []


def test_surplus_offers_respects_a_zero_avg_daily_sales(client):
    """avgDailySales=0 means "sells nothing" on the surplus path too.

    A dead-slow product honestly reporting 0/day with 15 units on hand must be flagged
    as pure surplus -- never credited with expected sell-through it will not have.
    """
    dead_slow = {
        "productId": "p_dead", "title": "بسبوسة قديمة", "category": "حلويات شرقية",
        "price": 30, "freshnessWindow": 2, "avgDailySales": 0.0, "currentStock": 15,
    }
    r = client.post("/integration/restomind/surplus-offers",
                    json={"restaurantId": "R1", "stock": [dead_slow],
                          "timestamp": "2025-02-11T14:00:00", "closeHour": 22})
    assert r.status_code == 200
    items = r.json()["itemsAtRisk"]
    assert len(items) == 1, "a 0/day product with 15 units on hand must be flagged"
    assert items[0]["productId"] == "p_dead"
    assert items[0]["projectedSurplus"] == 15
    assert items[0]["suggestedDiscountPct"] == 40


def test_surplus_skips_a_product_without_any_basis(client):
    """No basis (no learned level, no owner estimate) -> cannot judge risk, so skip."""
    unknown = {
        "productId": "p_new", "title": "صنف جديد", "category": "حلويات شرقية",
        "price": 30, "freshnessWindow": 2, "avgDailySales": None, "currentStock": 15,
    }
    r = client.post("/integration/restomind/surplus-offers",
                    json={"restaurantId": "R1", "stock": [unknown],
                          "timestamp": "2025-02-11T14:00:00", "closeHour": 22})
    assert r.status_code == 200
    assert r.json()["itemsAtRisk"] == []


def test_to_business_time_converts_an_aware_instant_and_passes_naive_through():
    """`close_hour` is a Cairo wall-clock hour, so the clock must be read in Cairo."""
    # Summer: Cairo is UTC+3.
    assert to_business_time(
        dt.datetime(2026, 7, 29, 19, 30, tzinfo=dt.timezone.utc)
    ) == dt.datetime(2026, 7, 29, 22, 30)
    # Winter: UTC+2. A fixed offset would get one of these two wrong.
    assert to_business_time(
        dt.datetime(2026, 1, 15, 19, 30, tzinfo=dt.timezone.utc)
    ) == dt.datetime(2026, 1, 15, 21, 30)
    # Date rollover: 21:30Z in July is already the next Cairo day.
    assert to_business_time(
        dt.datetime(2026, 7, 29, 21, 30, tzinfo=dt.timezone.utc)
    ).date() == dt.date(2026, 7, 30)
    naive = dt.datetime(2026, 7, 29, 22, 30)
    assert to_business_time(naive) == naive


def test_surplus_offers_reads_the_clock_in_cairo_not_utc(client):
    """An offset-aware UTC timestamp must be converted before `.hour` is read."""
    url = "/integration/restomind/surplus-offers"
    dead_slow = {
        "productId": "p_dead", "title": "بسبوسة قديمة", "category": "حلويات شرقية",
        "price": 30, "freshnessWindow": 2, "avgDailySales": 0.0, "currentStock": 15,
    }

    def scan(timestamp):
        r = client.post(url, json={"restaurantId": "R1", "stock": [dead_slow],
                                   "timestamp": timestamp, "closeHour": 22})
        assert r.status_code == 200
        items = r.json()["itemsAtRisk"]
        assert len(items) == 1
        return items[0]

    # 19:30Z is 22:30 in Cairo: past closing.
    aware_utc = scan("2026-07-29T19:30:00Z")
    naive_cairo = scan("2026-07-29T22:30:00")
    naive_utc_hour = scan("2026-07-29T19:30:00")

    assert aware_utc["hoursToClose"] == 0.0, "past closing -- nothing left to sell in"
    assert aware_utc["hoursToClose"] == naive_cairo["hoursToClose"]
    assert naive_utc_hour["hoursToClose"] == 2.5
    assert aware_utc["hoursToClose"] != naive_utc_hour["hoursToClose"]


def test_predict_matches_restomind_prediction_shape(client):
    """The /predict response must map onto their `predictions` document fields."""
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "P42", "title": "كنافة",
        "category": "حلويات شرقية", "targetWeek": "2025-03-10", "avgDailySales": 40,
    })
    assert r.status_code == 200
    b = r.json()
    for field in ("restaurantId", "productId", "modelVersionId", "targetWeek",
                  "predictedOrders", "featuresUsed"):
        assert field in b
    assert b["predictedOrders"] > 0
    assert len(b["dailyBreakdown"]) == 7
    # Post-rule, no invented Ramadan factor: the number is the basis level.
    assert b["factors"] == []


def test_predict_weekly_total_equals_daily_sum(client):
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "P1", "title": "كرواسون",
        "category": "معجنات", "targetWeek": "2025-02-10", "avgDailySales": 180,
    }).json()
    assert r["predictedOrders"] == sum(d["predictedQuantity"] for d in r["dailyBreakdown"])


def test_predict_uses_trained_week_for_sku_product(client):
    """A product with a trained SKU gets the calendar-aware trained week.

    The response is branded `calendar_decomposed/*`, carries real calendar factors
    (not []), and reconciles: predictedOrders == sum of the 7 days.
    """
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "Pk", "title": "كنافة",
        "category": "حلويات", "targetWeek": "2025-03-10", "avgDailySales": 40,
        "sku": "SWEET_KONAFA",
    }).json()
    assert r["modelVersionId"].startswith("calendar_decomposed/")
    assert r["featuresUsed"]["levelSource"] == "trained_model"
    assert r["featuresUsed"]["mode"] == "calendar_decomposed"
    assert r["confidence"] in ("high", "medium", "low")
    assert len(r["dailyBreakdown"]) == 7
    assert r["predictedOrders"] == sum(d["predictedQuantity"] for d in r["dailyBreakdown"])
    assert r["trainingMessage"] is None


def test_zero_avg_daily_sales_is_respected_not_replaced_by_default():
    """avgDailySales=0 means "sells nothing", not "give it a made-up level"."""
    from app.integration.restomind import ProductInput, basis_level

    zero = ProductInput(product_id="p0", title="Dead SKU", category="bread", avg_daily_sales=0.0)
    assert basis_level(zero) == 0.0

    unknown = ProductInput(product_id="p1", title="New SKU", category="bread", avg_daily_sales=None)
    assert basis_level(unknown) is None   # no basis -> still training


def test_predict_without_basis_reports_training(client):
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "p0", "title": "Dead SKU",
        "category": "bread", "targetWeek": "2025-02-10",
    })
    assert r.status_code == 200
    b = r.json()
    assert b["predictedOrders"] == 0
    assert b["trainingMessage"]
    assert b["featuresUsed"]["levelSource"] == "none"


def test_predict_daily_breakdown_repeats_the_basis_level(client):
    """Post-rule the week is the daily basis level; day count reconciles with total."""
    r = client.post("/integration/restomind/predict",
                    json={"restaurantId": "R1", "productId": "pflat", "title": "توست",
                          "category": "bread", "targetWeek": "2025-03-09",
                          "avgDailySales": 100})
    b = r.json()
    assert len(b["dailyBreakdown"]) == 7
    assert {d["predictedQuantity"] for d in b["dailyBreakdown"]} == {100}
    assert b["predictedOrders"] == 700
    assert b["featuresUsed"]["baseDailyLevel"] == 100


def test_predict_zero_avg_daily_sales_reports_true_base_level(client):
    """avgDailySales=0 is a real level, not a silently substituted default."""
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "p0", "title": "Dead SKU",
        "category": "bread", "targetWeek": "2025-02-10", "avgDailySales": 0,
    })
    assert r.status_code == 200
    assert r.json()["featuresUsed"]["baseDailyLevel"] == 0