"""End-to-end API tests: every endpoint driven for real."""

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


RAMADAN_2025 = "2025-03-15"      # mid-Ramadan
NORMAL_DAY = "2025-02-11"        # ordinary Tuesday
PRE_EID_2025 = "2025-03-27"      # kahk window


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data_source"] == "SIMULATED"


def test_daily_forecast_shape(client):
    r = client.post("/forecast/daily", json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY})
    assert r.status_code == 200
    b = r.json()
    assert b["recommended_quantity"] > 0
    assert b["lower_bound"] <= b["recommended_quantity"] <= b["upper_bound"]
    assert b["confidence"] in {"high", "medium", "low"}


def test_unknown_sku_is_rejected(client):
    r = client.post("/forecast/daily", json={"sku": "NOT_A_THING", "date": NORMAL_DAY})
    assert r.status_code == 422


def test_ramadan_suppresses_croissant_forecast(client):
    """The headline claim: the model knows Ramadan changes what Egyptians buy."""
    normal = client.post("/forecast/daily", json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY}).json()
    ramadan = client.post("/forecast/daily", json={"sku": "PASTRY_CROISSANT", "date": RAMADAN_2025}).json()
    assert ramadan["recommended_quantity"] < normal["recommended_quantity"]
    # The model must name Ramadan as a driver (phase label, e.g. "Ramadan (middle)").
    assert any("Ramadan" in f["factor"] for f in ramadan["factors"])
    ramadan_factor = next(f for f in ramadan["factors"] if "Ramadan" in f["factor"])
    assert ramadan_factor["direction"] == "decrease"


def test_ramadan_lifts_konafa_forecast(client):
    normal = client.post("/forecast/daily", json={"sku": "SWEET_KONAFA", "date": NORMAL_DAY}).json()
    ramadan = client.post("/forecast/daily", json={"sku": "SWEET_KONAFA", "date": RAMADAN_2025}).json()
    assert ramadan["recommended_quantity"] > normal["recommended_quantity"]


def test_weekly_forecast_returns_seven_days(client):
    r = client.post("/forecast/weekly", json={"sku": "CAKE_GATEAU", "start_date": NORMAL_DAY})
    assert r.status_code == 200
    b = r.json()
    assert len(b["days"]) == 7
    assert b["total_quantity"] == sum(d["recommended_quantity"] for d in b["days"])


def test_seasonality_adjustment_explains_ramadan(client):
    r = client.post("/forecast/seasonality-adjustment",
                    json={"sku": "SWEET_KONAFA", "date": RAMADAN_2025})
    assert r.status_code == 200
    b = r.json()
    assert b["calendar"]["is_ramadan"] is True
    assert b["calendar"]["ramadan_day"] is not None
    assert b["multiplier"] > 1.0


def test_waste_alert_fires_on_gross_overproduction(client):
    fc = client.post("/forecast/daily", json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY}).json()
    huge = fc["upper_bound"] * 3
    r = client.post("/alerts/waste-prevention",
                    json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY, "planned_quantity": huge})
    b = r.json()
    assert b["severity"] == "high"
    assert b["excess_qty"] > 0
    assert b["projected_waste_cost_egp"] > 0


def test_waste_alert_silent_when_plan_is_reasonable(client):
    fc = client.post("/forecast/daily", json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY}).json()
    r = client.post("/alerts/waste-prevention",
                    json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY,
                          "planned_quantity": fc["recommended_quantity"]})
    assert r.json()["severity"] == "none"


def test_surplus_detect_flags_stagnant_stock(client):
    r = client.post("/surplus/detect", json={
        "stock": {"CAKE_GATEAU": 60, "PASTRY_CROISSANT": 40, "BREAD_BALADI": 300},
        "timestamp": f"{NORMAL_DAY}T19:30:00",
        "close_hour": 22,
    })
    assert r.status_code == 200
    b = r.json()
    assert len(b["items_at_risk"]) > 0
    assert b["total_value_at_risk_egp"] > 0
    for item in b["items_at_risk"]:
        assert 5 <= item["suggested_discount_pct"] <= 70
        assert item["urgency"] in {"high", "medium", "low"}


def test_surplus_quiet_in_the_morning(client):
    """At 8am there is a whole day left to sell -- nothing should be at risk."""
    r = client.post("/surplus/detect", json={
        "stock": {"CAKE_GATEAU": 60},
        "timestamp": f"{NORMAL_DAY}T08:00:00",
        "close_hour": 22,
    })
    assert r.json()["items_at_risk"] == []


def test_generate_offer_returns_egyptian_arabic(client):
    r = client.post("/marketing/generate-offer",
                    json={"sku": "CAKE_GATEAU", "discount_pct": 30})
    assert r.status_code == 200
    b = r.json()
    assert "جاتوه" in b["copy_ar"]
    # Offer terms stated either as a percentage or as a price pair.
    assert "30" in b["copy_ar"] or ("24" in b["copy_ar"] and "35" in b["copy_ar"])
    assert b["new_price"] < b["old_price"]
    assert b["generator"] in {"llm", "template"}


def test_publish_defaults_to_dry_run(client):
    r = client.post("/marketing/publish",
                    json={"sku": "CAKE_GATEAU", "copy_ar": "عرض تجريبي",
                          "platforms": ["facebook"]})
    b = r.json()
    assert b["status"] == "preview"
    assert b["dry_run"] is True
    assert b["post_ids"] == {}


def test_publish_live_refuses_without_credentials(client):
    """Live publishing must fail closed, never silently no-op as if it worked."""
    r = client.post("/marketing/publish",
                    json={"sku": "CAKE_GATEAU", "copy_ar": "عرض تجريبي",
                          "platforms": ["facebook"], "dry_run": False})
    b = r.json()
    assert b["status"] == "failed"
    assert b["post_ids"] == {}


# -- batch endpoints ----------------------------------------------------------------


def test_daily_batch_returns_all_items(client):
    r = client.post("/forecast/daily-batch", json={"date": NORMAL_DAY})
    assert r.status_code == 200
    b = r.json()
    assert b["item_count"] == 11
    assert b["total_quantity"] == sum(i["recommended_quantity"] for i in b["items"])


def test_daily_batch_matches_single_endpoint(client):
    """A batch that disagrees with the per-item endpoint would be a silent bug."""
    batch = client.post("/forecast/daily-batch",
                        json={"date": RAMADAN_2025, "skus": ["PASTRY_CROISSANT", "SWEET_KONAFA"]}).json()
    by_sku = {i["sku"]: i["recommended_quantity"] for i in batch["items"]}
    for sku in ("PASTRY_CROISSANT", "SWEET_KONAFA"):
        single = client.post("/forecast/daily", json={"sku": sku, "date": RAMADAN_2025}).json()
        assert by_sku[sku] == single["recommended_quantity"]


def test_daily_batch_rejects_unknown_sku(client):
    r = client.post("/forecast/daily-batch", json={"date": NORMAL_DAY, "skus": ["NOPE"]})
    assert r.status_code == 422


def test_weekly_batch_returns_seven_days_per_item(client):
    r = client.post("/forecast/weekly-batch",
                    json={"start_date": NORMAL_DAY, "skus": ["CAKE_GATEAU", "BREAD_BALADI"]})
    assert r.status_code == 200
    b = r.json()
    assert b["item_count"] == 2
    for item in b["items"]:
        assert len(item["days"]) == 7
        assert item["total_quantity"] == sum(d["recommended_quantity"] for d in item["days"])
