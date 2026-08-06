"""End-to-end API tests: every endpoint driven for real."""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

import os

os.environ.setdefault("REGISTRY_STORE", "")   # in-memory registry for tests


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


def test_surplus_detect_reads_an_offset_aware_timestamp_as_cairo(client):
    """An aware instant must be read as the Cairo wall clock it represents.

    `close_hour` is a Cairo wall-clock hour, and so is the index of the
    sell-through curve `detect_surplus` reads. This endpoint passed the caller's
    timestamp straight through, so an offset-aware UTC instant had its UTC hour
    compared against a Cairo one. Its sibling
    `/integration/restomind/surplus-offers` was fixed to normalise; this one
    reads the same `close_hour` through the same curve and was not.

    Every other surplus test here sends a NAIVE timestamp, which is exactly why
    this stayed invisible.
    """
    stock = {"CAKE_GATEAU": 60, "PASTRY_CROISSANT": 40, "BREAD_BALADI": 300}

    aware_utc = dt.datetime.fromisoformat(f"{NORMAL_DAY}T19:30:00+00:00")
    cairo = aware_utc.astimezone(ZoneInfo("Africa/Cairo")).replace(tzinfo=None)
    # NORMAL_DAY is in February, so Cairo is UTC+2: 19:30Z is 21:30 local.
    assert cairo.hour == 21

    from_aware = client.post("/surplus/detect", json={
        "stock": stock, "timestamp": aware_utc.isoformat(), "close_hour": 22,
    }).json()
    from_cairo = client.post("/surplus/detect", json={
        "stock": stock, "timestamp": cairo.isoformat(), "close_hour": 22,
    }).json()

    # The same instant, so necessarily the same answer.
    assert from_aware["items_at_risk"] == from_cairo["items_at_risk"]

    # And genuinely normalised rather than passed through: reading 19:30 as a
    # Cairo wall clock leaves 2.5h to close instead of 0.5h, which the
    # sell-through curve prices very differently.
    naive_1930 = client.post("/surplus/detect", json={
        "stock": stock, "timestamp": f"{NORMAL_DAY}T19:30:00", "close_hour": 22,
    }).json()
    assert (
        from_aware["total_value_at_risk_egp"]
        != naive_1930["total_value_at_risk_egp"]
    )


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


def test_publish_requires_a_real_offer_id(client):
    """copy_ar is no longer accepted directly -- publishing has to reference an
    offer that already passed generate-offer's _validate() (OWASP LLM02)."""
    r = client.post("/marketing/publish",
                    json={"offerId": "not-a-real-id", "platforms": ["facebook"]})
    assert r.status_code == 404


def test_publish_defaults_to_dry_run(client):
    offer = client.post("/marketing/generate-offer",
                        json={"sku": "CAKE_GATEAU", "discount_pct": 30}).json()
    r = client.post("/marketing/publish",
                    json={"offerId": offer["offerId"], "platforms": ["facebook"]})
    b = r.json()
    assert b["status"] == "preview"
    assert b["dry_run"] is True
    assert b["post_ids"] == {}


def test_publish_live_refuses_without_credentials(client):
    """Live publishing must fail closed, never silently no-op as if it worked."""
    offer = client.post("/marketing/generate-offer",
                        json={"sku": "CAKE_GATEAU", "discount_pct": 30}).json()
    r = client.post("/marketing/publish",
                    json={"offerId": offer["offerId"],
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


# -- integration auth -----------------------------------------------------------------


def test_integration_routes_require_the_shared_secret(monkeypatch, client):
    """Anyone who can reach the port must not be able to poison another tenant's
    learned demand levels via /integration/restomind/* without the shared secret.

    The guard reads AI_SHARED_SECRET per request, not at app construction, so this
    reuses the module-level `app`/`client` fixture instead of reloading the module --
    reloading would reconstruct the shared FastAPI app singleton, which this suite
    already knows is ordering-sensitive (see Task 3's deferred-minor note about a test
    that re-runs the app's lifespan and mutates global registry state). monkeypatch
    also guarantees AI_SHARED_SECRET is unset again after this test regardless of
    whether the assertions below pass or fail, so nothing leaks into later tests.
    """
    monkeypatch.setenv("AI_SHARED_SECRET", "s3cret")

    payload = {"restaurantId": "R1", "productId": "p1", "title": "X",
               "targetWeek": "2025-03-09", "avgDailySales": 10}

    unauthenticated = client.post("/integration/restomind/predict", json=payload)
    assert unauthenticated.status_code == 401

    ok = client.post("/integration/restomind/predict", json=payload,
                     headers={"X-RestoMind-Key": "s3cret"})
    assert ok.status_code == 200

    # Non-integration routes stay open.
    assert client.get("/health").status_code == 200


def test_marketing_routes_also_require_the_shared_secret(monkeypatch, client):
    """/marketing joined the guard alongside /integration/restomind -- both
    /generate-offer and /publish drive real LLM/Meta calls an unauthenticated caller
    could otherwise trigger for free (OWASP LLM04) or use to bypass copy validation
    entirely by reaching /publish directly (OWASP LLM02)."""
    monkeypatch.setenv("AI_SHARED_SECRET", "s3cret")

    unauthenticated = client.post(
        "/marketing/generate-offer", json={"sku": "CAKE_GATEAU", "discount_pct": 30}
    )
    assert unauthenticated.status_code == 401

    ok = client.post(
        "/marketing/generate-offer", json={"sku": "CAKE_GATEAU", "discount_pct": 30},
        headers={"X-RestoMind-Key": "s3cret"},
    )
    assert ok.status_code == 200

    assert client.post(
        "/marketing/publish", json={"offerId": "whatever", "platforms": ["facebook"]}
    ).status_code == 401


def test_title_over_max_length_is_rejected(client):
    """OWASP LLM01 -- title is interpolated directly into an LLM prompt; unbounded
    length widens the injection surface with no legitimate use case for it."""
    r = client.post(
        "/integration/restomind/predict",
        json={"restaurantId": "R1", "productId": "p1", "title": "x" * 121,
              "targetWeek": "2025-03-09", "avgDailySales": 10},
    )
    assert r.status_code == 422


def test_surplus_stock_over_max_length_is_rejected(client):
    """OWASP LLM04 -- each stock item can trigger its own LLM call; an unbounded
    array lets one request trigger an unbounded number of them."""
    stock = [{"productId": f"p{i}", "title": "X", "currentStock": 5} for i in range(51)]
    r = client.post(
        "/integration/restomind/surplus-offers",
        json={"restaurantId": "R1", "stock": stock},
    )
    assert r.status_code == 422


def test_preflight_to_protected_route_still_gets_cors_headers(monkeypatch, client):
    """CORSMiddleware must be the outermost layer, not wrapped by the secret guard.

    Starlette builds the middleware stack so the LAST-registered middleware becomes
    OUTERMOST. If the secret guard is registered after CORSMiddleware (as a naive
    reading of "insert immediately after the CORSMiddleware block" would do), the
    guard runs first and can short-circuit an OPTIONS preflight with a 401 before
    CORSMiddleware ever gets a chance to attach Access-Control-Allow-* headers --
    which makes every cross-origin browser call to a protected route fail at
    preflight, correct secret or not, since the browser never gets to see the 401
    body or retry with credentials for a request CORS itself rejected.
    """
    monkeypatch.setenv("AI_SHARED_SECRET", "s3cret")

    preflight = client.options(
        "/integration/restomind/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" in {k.lower() for k in preflight.headers.keys()}
