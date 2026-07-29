"""RestoMind integration bridge: accepts their product shapes, returns model output.

Cold-start (rule-based) since no sales history exists yet -- the day-one path for the
Admin/Stores screens while the backend is still being built.
"""

from fastapi.testclient import TestClient
import pytest

from app.api.main import app
from app.integration.restomind import map_category

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


def test_category_mapping_handles_arabic_and_unknown():
    assert map_category("معجنات") == "pastry"
    assert map_category("حلويات شرقية") == "sweet"
    assert map_category("مخبوزات") == "bread"
    assert map_category("Something Random") == ""   # neutral fallback


def test_production_plan_returns_one_row_per_product(client):
    r = client.post("/integration/restomind/production-plan",
                    json={"restaurantId": "R1", "date": "2025-02-11", "products": PRODUCTS})
    assert r.status_code == 200
    b = r.json()
    assert len(b["items"]) == len(PRODUCTS)
    assert b["totalRecommendedQty"] == sum(i["recommendedQty"] for i in b["items"])
    assert all(i["source"] == "rule_based" for i in b["items"])


def test_production_plan_applies_ramadan_by_category(client):
    """Croissant (pastry) must be lower in Ramadan than on a normal day, from priors."""
    normal = client.post("/integration/restomind/production-plan",
                         json={"restaurantId": "R1", "date": "2025-02-11", "products": PRODUCTS}).json()
    ramadan = client.post("/integration/restomind/production-plan",
                         json={"restaurantId": "R1", "date": "2025-03-15", "products": PRODUCTS}).json()
    n = {i["title"]: i["recommendedQty"] for i in normal["items"]}
    rm = {i["title"]: i["recommendedQty"] for i in ramadan["items"]}
    assert rm["كرواسون"] < n["كرواسون"]     # breakfast pastry falls
    assert rm["كنافة"] > n["كنافة"]          # Ramadan sweet rises


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
        # low-price item must not collapse to "2 بدل 2"
        assert it["newPrice"] < it.get("price", 9e9) if "price" in it else True


def test_surplus_quiet_in_the_morning(client):
    stock = [{**PRODUCTS[0], "currentStock": 40}]
    r = client.post("/integration/restomind/surplus-offers",
                    json={"restaurantId": "R1", "stock": stock,
                          "timestamp": "2025-02-11T08:00:00", "closeHour": 22})
    assert r.json()["itemsAtRisk"] == []


def test_surplus_offers_respects_a_zero_avg_daily_sales(client):
    """avgDailySales=0 means "sells nothing" on the surplus path too.

    `surplus_offers` used a truthy `s.avg_daily_sales or DEFAULT_DAILY_LEVEL`,
    so a dead-slow SKU honestly reporting 0/day was credited with 40/day of
    expected sell-through. Its projected surplus collapsed to ~0 and it was
    skipped -- never flagged, never discounted -- which is precisely the stock
    most at risk of being thrown away.
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
    # Nothing is expected to sell -> the whole 15 units are surplus -> raw_risk 1.0
    # -> the top discount tier.
    assert items[0]["projectedSurplus"] == 15
    assert items[0]["suggestedDiscountPct"] == 40


def test_surplus_offers_still_substitutes_the_default_for_a_null_estimate(client):
    """None is "no estimate given" and must keep falling back to the default level.

    Guards the fix above from over-correcting into "treat missing as zero",
    which would flag every cold-start product as pure surplus.
    """
    unknown = {
        "productId": "p_new", "title": "صنف جديد", "category": "حلويات شرقية",
        "price": 30, "freshnessWindow": 2, "avgDailySales": None, "currentStock": 15,
    }
    r = client.post("/integration/restomind/surplus-offers",
                    json={"restaurantId": "R1", "stock": [unknown],
                          "timestamp": "2025-02-11T14:00:00", "closeHour": 22})
    assert r.status_code == 200
    # 40/day * remaining share comfortably covers 15 units, so there is no
    # projected surplus and nothing to discount.
    assert r.json()["itemsAtRisk"] == []


def test_predict_matches_restomind_prediction_shape(client):
    """The /predict response must map onto their `predictions` document fields."""
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "P42", "title": "كنافة",
        "category": "حلويات شرقية", "targetWeek": "2025-03-10", "avgDailySales": 40,
    })
    assert r.status_code == 200
    b = r.json()
    # Fields RestoMind's prediction.model.ts expects.
    for field in ("restaurantId", "productId", "modelVersionId", "targetWeek",
                  "predictedOrders", "featuresUsed"):
        assert field in b
    assert b["predictedOrders"] > 0
    assert len(b["dailyBreakdown"]) == 7
    # Ramadan week -> the calendar snapshot records it, and it drives the number up.
    assert b["featuresUsed"]["calendar"]["isRamadan"] is True
    assert any(f["factor"] == "Ramadan" and f["direction"] == "increase" for f in b["factors"])


def test_predict_weekly_total_equals_daily_sum(client):
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "P1", "title": "كرواسون",
        "category": "معجنات", "targetWeek": "2025-02-10", "avgDailySales": 180,
    }).json()
    assert r["predictedOrders"] == sum(d["qty"] for d in r["dailyBreakdown"])


def test_zero_avg_daily_sales_is_respected_not_replaced_by_default():
    """avgDailySales=0 means 'sells nothing', not 'no estimate given'."""
    from app.core.egypt_calendar import CALENDAR
    from app.integration.restomind import DEFAULT_DAILY_LEVEL, ProductInput, _forecast_one
    import datetime as dt

    feats = CALENDAR.features(dt.date(2025, 2, 11))
    zero = ProductInput(product_id="p0", title="Dead SKU", category="bread", avg_daily_sales=0.0)
    qty, _ = _forecast_one(zero, feats)
    assert qty == 0.0

    # None means "no estimate given" -> the default level is substituted, so it
    # must forecast identically to a product that explicitly states that level.
    unknown = ProductInput(product_id="p1", title="New SKU", category="bread", avg_daily_sales=None)
    explicit_default = ProductInput(
        product_id="p1b", title="New SKU", category="bread",
        avg_daily_sales=DEFAULT_DAILY_LEVEL,
    )
    qty_unknown, _ = _forecast_one(unknown, feats)
    qty_explicit, _ = _forecast_one(explicit_default, feats)
    assert qty_unknown > 0
    assert qty_unknown == qty_explicit


def test_fractional_avg_daily_sales_is_preserved():
    from app.core.egypt_calendar import CALENDAR
    from app.integration.restomind import ProductInput, _forecast_one
    import datetime as dt

    feats = CALENDAR.features(dt.date(2025, 2, 11))
    low = ProductInput(product_id="p2", title="Slow SKU", category="bread", avg_daily_sales=0.43)
    qty, _ = _forecast_one(low, feats)
    assert 0 < qty < 5


def test_predict_daily_breakdown_uses_predicted_quantity_key(client):
    r = client.post("/integration/restomind/predict",
                    json={"restaurantId": "R1", "productId": "p1", "title": "كرواسون",
                          "category": "معجنات", "targetWeek": "2025-03-09",
                          "avgDailySales": 180})
    assert r.status_code == 200
    b = r.json()
    assert len(b["dailyBreakdown"]) == 7
    for day in b["dailyBreakdown"]:
        assert "predictedQuantity" in day, "consumer reads predictedQuantity"
        assert day["predictedQuantity"] == day["qty"], "qty kept as deprecated alias"
    # The weekly total must reconcile with the daily rows.
    assert b["predictedOrders"] == sum(d["predictedQuantity"] for d in b["dailyBreakdown"])


def test_predict_daily_breakdown_is_not_flat_across_ramadan(client):
    """A week spanning Ramadan must vary day to day, not be a flat average."""
    r = client.post("/integration/restomind/predict",
                    json={"restaurantId": "R1", "productId": "p1", "title": "كرواسون",
                          "category": "معجنات", "targetWeek": "2025-02-27",
                          "avgDailySales": 180})
    qtys = [d["predictedQuantity"] for d in r.json()["dailyBreakdown"]]
    assert len(set(qtys)) > 1, "calendar signal must survive into the daily rows"


def test_predict_week_zero_avg_daily_sales_reports_true_base_level(client):
    """avgDailySales=0 means 'sells nothing' -- featuresUsed.baseDailyLevel must reflect
    that, not silently fall back to DEFAULT_DAILY_LEVEL (40)."""
    r = client.post("/integration/restomind/predict", json={
        "restaurantId": "R1", "productId": "p0", "title": "Dead SKU",
        "category": "bread", "targetWeek": "2025-02-10", "avgDailySales": 0,
    })
    assert r.status_code == 200
    b = r.json()
    assert b["featuresUsed"]["baseDailyLevel"] == 0
