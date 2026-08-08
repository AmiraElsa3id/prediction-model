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


# -- API key auth ----------------------------------------------------------------------
# docs/01-api-key-hardening.md: every route except /health requires X-API-Key once
# REQUIRE_API_KEY or API_KEY_HASH is set. auth.py reads its config at import time (so a
# misconfigured deploy fails at boot, not at the first request -- docs §2.4/§4), so tests
# that need the key enforced set the module's attributes directly via monkeypatch.setattr
# rather than the env vars, and rely on monkeypatch to restore them after each test.

import hashlib

from app.api import auth

TEST_KEY = "s3cret"
TEST_KEY_HASH = hashlib.sha256(TEST_KEY.encode()).hexdigest()


def test_protected_routes_require_the_api_key(monkeypatch, client):
    """Anyone who can reach the port must not be able to poison another tenant's
    learned demand levels, or use any other route, without the API key.
    """
    monkeypatch.setattr(auth, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(auth, "API_KEY_HASH", TEST_KEY_HASH)

    payload = {"restaurantId": "R1", "productId": "p1", "title": "X",
               "targetWeek": "2025-03-09", "avgDailySales": 10}

    unauthenticated = client.post("/integration/restomind/predict", json=payload)
    assert unauthenticated.status_code == 401

    wrong_key = client.post("/integration/restomind/predict", json=payload,
                            headers={"X-API-Key": "not-the-key"})
    assert wrong_key.status_code == 401

    ok = client.post("/integration/restomind/predict", json=payload,
                     headers={"X-API-Key": TEST_KEY})
    assert ok.status_code == 200

    # A non-integration route is protected too -- not just the old RestoMind-only scope.
    forecast_unauthenticated = client.post(
        "/forecast/daily", json={"sku": "PASTRY_CROISSANT", "date": NORMAL_DAY}
    )
    assert forecast_unauthenticated.status_code == 401

    # /health is the sole exemption.
    assert client.get("/health").status_code == 200


def test_refuses_to_boot_without_a_key_hash_when_required(monkeypatch):
    """A deploy that sets REQUIRE_API_KEY=true but forgets API_KEY_HASH must fail loudly
    at import/startup time, not silently serve every route as unauthenticated.
    """
    import importlib

    monkeypatch.setenv("REQUIRE_API_KEY", "true")
    monkeypatch.delenv("API_KEY_HASH", raising=False)
    try:
        with pytest.raises(RuntimeError):
            importlib.reload(auth)
    finally:
        # Restore the module to its normal (dev-mode) state so later tests -- which
        # import `auth` as the same shared module object `main.py` already holds a
        # reference to -- are unaffected by this test having reloaded it.
        monkeypatch.setenv("REQUIRE_API_KEY", "false")
        monkeypatch.delenv("API_KEY_HASH", raising=False)
        importlib.reload(auth)


# -- rate limiting -----------------------------------------------------------------------
# docs/03-cors-and-rate-limiting.md: a blunt cost guard, not per-user fairness -- there is
# exactly one legitimate caller (the backend), so this exists to cap the worst case (a
# retry loop, a leaked key) rather than to be fair to individual end users. Tests reset
# the shared in-memory counter and lower the limits via monkeypatch so they run fast and
# don't leak tripped state into other tests sharing this process.

from app.api import ratelimit


def test_default_tier_rate_limit_returns_429_with_retry_after(monkeypatch, client):
    monkeypatch.setattr(ratelimit, "_COUNTERS", {})
    monkeypatch.setattr(ratelimit, "DEFAULT_LIMIT_PER_MIN", 3)

    for _ in range(3):
        assert client.get("/model/status").status_code == 200

    limited = client.get("/model/status")
    assert limited.status_code == 429
    assert "retry-after" in {k.lower() for k in limited.headers.keys()}
    assert limited.json()["error"] == "rate_limited"


def test_marketing_tier_has_its_own_tighter_budget(monkeypatch, client):
    """The marketing tier's limit is tracked separately from the default tier, so heavy
    (legitimate) forecast traffic can't starve it, and vice versa.
    """
    monkeypatch.setattr(ratelimit, "_COUNTERS", {})
    monkeypatch.setattr(ratelimit, "MARKETING_LIMIT_PER_MIN", 2)

    payload = {"sku": "CAKE_GATEAU", "discount_pct": 30}
    for _ in range(2):
        assert client.post("/marketing/generate-offer", json=payload).status_code == 200

    limited = client.post("/marketing/generate-offer", json=payload)
    assert limited.status_code == 429

    # A different tier's budget is untouched by marketing's limit being tripped.
    assert client.get("/model/status").status_code == 200


def test_health_stays_exempt_from_rate_limiting(monkeypatch, client):
    monkeypatch.setattr(ratelimit, "_COUNTERS", {})
    monkeypatch.setattr(ratelimit, "DEFAULT_LIMIT_PER_MIN", 1)

    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_rate_limit_resets_after_the_window_elapses(monkeypatch, client):
    monkeypatch.setattr(ratelimit, "_COUNTERS", {})
    monkeypatch.setattr(ratelimit, "DEFAULT_LIMIT_PER_MIN", 1)

    fake_now = [1_000.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: fake_now[0])

    assert client.get("/model/status").status_code == 200
    assert client.get("/model/status").status_code == 429

    fake_now[0] += ratelimit.WINDOW_SECONDS + 1
    assert client.get("/model/status").status_code == 200
