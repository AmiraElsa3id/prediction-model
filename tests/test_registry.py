"""Per-restaurant registry: seeding real sales must change predictions, per tenant."""

import datetime as dt

from fastapi.testclient import TestClient
import pytest

from app.api.main import app
from app.integration.registry import MIN_DAYS_FOR_LEARNED

# ~55% of calendar days are "quiet" (weekends are excluded, plus Ramadan and the
# public holidays add ~6 weeks a year), so 2.2x the threshold always leaves a
# surplus of quiet days for a product meant to be learned.
_SEED_DAYS = int(MIN_DAYS_FOR_LEARNED * 2.2)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _sales(product_id, n, level, start="2025-01-01"):
    import random
    rng = random.Random(7)
    d0 = dt.date.fromisoformat(start)
    return [{"date": (d0 + dt.timedelta(days=i)).isoformat(),
             "productId": product_id, "salesQty": rng.randint(level - 12, level + 12)}
            for i in range(n)]


def test_seeding_sales_changes_the_prediction(client):
    rid, pid = "REG_R1", "REG_P1"
    prod = {"restaurantId": rid, "productId": pid, "title": "كرواسون",
            "category": "معجنات", "avgDailySales": 180, "targetWeek": "2025-02-10"}

    before = client.post("/integration/restomind/predict", json=prod).json()
    assert before["featuresUsed"]["levelSource"] == "owner_estimate"

    client.post("/integration/restomind/ingest", json={
        "restaurantId": rid, "records": _sales(pid, _SEED_DAYS, 110),
        "products": [{"productId": pid, "title": "كرواسون", "category": "معجنات"}],
    })

    after = client.post("/integration/restomind/predict", json=prod).json()
    assert after["featuresUsed"]["levelSource"] == "learned_from_sales"
    assert after["predictedOrders"] < before["predictedOrders"]  # learned 110 < guess 180


def test_registry_is_per_tenant(client):
    """Restaurant A's data must not leak into restaurant B."""
    pid = "SHARED_P"
    prod = {"productId": pid, "title": "كنافة", "category": "حلويات شرقية",
            "avgDailySales": 40, "targetWeek": "2025-02-10"}

    client.post("/integration/restomind/ingest", json={
        "restaurantId": "REG_A", "records": _sales(pid, _SEED_DAYS, 200),
        "products": [{"productId": pid, "title": "كنافة", "category": "حلويات شرقية"}],
    })
    a = client.post("/integration/restomind/predict", json={"restaurantId": "REG_A", **prod}).json()
    b = client.post("/integration/restomind/predict", json={"restaurantId": "REG_B", **prod}).json()

    assert a["featuresUsed"]["levelSource"] == "learned_from_sales"
    assert b["featuresUsed"]["levelSource"] == "owner_estimate"   # B never got data
    assert a["predictedOrders"] != b["predictedOrders"]


def test_too_little_data_keeps_owner_estimate(client):
    rid, pid = "REG_R3", "REG_P3"
    prod = {"restaurantId": rid, "productId": pid, "title": "دوناتس",
            "category": "معجنات", "avgDailySales": 100, "targetWeek": "2025-02-10"}
    client.post("/integration/restomind/ingest", json={
        "restaurantId": rid, "records": _sales(pid, 5, 60),   # below MIN_DAYS_FOR_LEARNED
        "products": [{"productId": pid, "title": "دوناتس", "category": "معجنات"}],
    })
    after = client.post("/integration/restomind/predict", json=prod).json()
    assert after["featuresUsed"]["levelSource"] == "owner_estimate"


def test_registry_status_reports_learned_products(client):
    rid, pid = "REG_R4", "REG_P4"
    client.post("/integration/restomind/ingest", json={
        "restaurantId": rid, "records": _sales(pid, _SEED_DAYS, 90),
        "products": [{"productId": pid, "title": "فطير", "category": "مالح"}],
    })
    st = client.get(f"/integration/restomind/status/{rid}").json()
    assert st["usingLearnedLevel"] == 1
    assert st["items"][0]["learnedLevel"] is not None


def test_status_reports_its_own_thresholds(client):
    """RestoMind draws a per-product progress bar from these.

    It used to hardcode `MIN_DAYS_FOR_LEARNED = 90` on its own side, so tuning the
    threshold here silently made that bar wrong. The numbers belong to this module,
    so the endpoint states them rather than leaving the caller to guess.
    """
    from app.integration.registry import (
        CONFIDENT_DAYS,
        MIN_DAYS_FOR_LEARNED,
        QUIET_WINDOW,
    )

    st = client.get("/integration/restomind/status/REG_THRESHOLDS").json()
    assert st["minDaysForLearned"] == MIN_DAYS_FOR_LEARNED
    assert st["quietWindowDays"] == QUIET_WINDOW
    assert st["confidentDays"] == CONFIDENT_DAYS
    # Reported even for a restaurant with no data at all -- the progress bar has
    # to render on day zero too.
    assert st["productsTracked"] == 0


def test_registry_survives_a_restart(tmp_path):
    """Learned levels must outlive the process, or a redeploy resets every tenant.

    Note: this pins round-trip / dtype fidelity (products, observed_days, learned_level,
    titles all surviving a fresh RestaurantRegistry over the same file) -- it does NOT
    discriminate pickle from JSON. It passes even against the old pickle-based store,
    because pickle round-trips correctly too. That discrimination is
    `test_registry_store_is_not_pickle`'s job; don't over-trust this one for that.
    """
    import pandas as pd
    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput

    store = tmp_path / "registry.json"
    rows = pd.DataFrame([
        {"date": d, "productId": "p1", "salesQty": 100}
        # _SEED_DAYS consecutive days guarantees the >= MIN_DAYS_FOR_LEARNED quiet
        # weekdays, so the learned level is set.
        for d in pd.date_range("2025-01-06", periods=_SEED_DAYS).strftime("%Y-%m-%d")
    ])

    first = RestaurantRegistry(persist_path=store)
    first.ingest("R1", rows, [ProductInput(product_id="p1", title="Bread", category="bread")])
    assert first.status("R1")["usingLearnedLevel"] == 1
    learned = first.status("R1")["items"][0]["learnedLevel"]

    # Simulate a restart: a brand-new registry over the same file.
    second = RestaurantRegistry(persist_path=store)
    status = second.status("R1")
    assert status["productsTracked"] == 1
    assert status["usingLearnedLevel"] == 1
    assert status["items"][0]["learnedLevel"] == learned
    assert status["items"][0]["title"] == "Bread"


def test_registry_store_is_not_pickle(tmp_path):
    """The store is loaded at startup; it must not be an executable format."""
    import json
    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput
    import pandas as pd

    store = tmp_path / "registry.json"
    reg = RestaurantRegistry(persist_path=store)
    reg.ingest("R1", pd.DataFrame([{"date": "2025-01-06", "productId": "p1", "salesQty": 5}]),
               [ProductInput(product_id="p1", title="Bread")])
    json.loads(store.read_text(encoding="utf-8"))  # must parse as JSON


def test_registry_tolerates_a_corrupt_store(tmp_path):
    from app.integration.registry import RestaurantRegistry

    store = tmp_path / "registry.json"
    store.write_text("{ not json", encoding="utf-8")
    reg = RestaurantRegistry(persist_path=store)  # must not raise
    assert reg.status("R1")["productsTracked"] == 0


def test_default_registry_store_path_is_used_when_unset(tmp_path, monkeypatch):
    """Pins main.py's actual production default: with REGISTRY_STORE genuinely UNSET
    (not just empty), the lifespan must persist under data/registry.json relative to
    the process cwd. Every other test module in this suite forces REGISTRY_STORE=""
    for isolation, so nothing else exercises this path -- this is the only test that
    proves the fix in app/api/main.py actually takes effect in production.
    """
    monkeypatch.delenv("REGISTRY_STORE", raising=False)
    monkeypatch.chdir(tmp_path)

    from app.api.main import STATE, app as real_app

    # This test runs a SECOND lifespan against the same app object, and shutdown
    # calls STATE.clear() -- which empties the registry the module-scoped `client`
    # fixture is still holding, so every client test after this one would 500 on a
    # missing STATE["registry"]. Snapshot and restore so ordering cannot matter.
    saved_state = dict(STATE)
    try:
        with TestClient(real_app) as c:
            rid, pid = "DEFAULT_R1", "DEFAULT_P1"
            resp = c.post("/integration/restomind/ingest", json={
                "restaurantId": rid,
                "records": [{"date": "2025-01-06", "productId": pid, "salesQty": 5}],
                "products": [{"productId": pid, "title": "Bread"}],
            })
            assert resp.status_code == 200
    finally:
        STATE.update(saved_state)

    assert (tmp_path / "data" / "registry.json").exists()


def test_upsert_merges_rather_than_replacing():
    """A later caller must not blank fields an earlier one populated.

    `/predict` sends one product with a category and no economics; an ingest sends the
    catalogue with price and freshness_window. Replacing wholesale meant whichever
    arrived last won, so a predict could wipe the price and shelf life -- and those two
    are exactly what the newsvendor q* is computed from.
    """
    from app.integration.registry import RestaurantState
    from app.integration.restomind import ProductInput

    state = RestaurantState(restaurant_id="R_MERGE")

    # Ingest registers the full record.
    state.upsert_products([
        ProductInput(
            product_id="P1", title="كرواسون", category="معجنات",
            price=18.0, freshness_window=2.0, avg_daily_sales=40.0,
        )
    ])

    # A predict-shaped payload: title + category only, economics unknown.
    state.upsert_products([
        ProductInput(product_id="P1", title="كرواسون", category="معجنات")
    ])

    kept = state.products["P1"].product
    assert kept.price == 18.0, "price was blanked by a caller that never knew it"
    assert kept.freshness_window == 2.0, "shelf life was blanked"
    assert kept.avg_daily_sales == 40.0


def test_upsert_applies_real_updates():
    """Merging must not freeze the record -- provided values still win."""
    from app.integration.registry import RestaurantState
    from app.integration.restomind import ProductInput

    state = RestaurantState(restaurant_id="R_MERGE2")
    state.upsert_products([
        ProductInput(product_id="P1", title="Old", category="bread", price=10.0)
    ])
    state.upsert_products([
        ProductInput(product_id="P1", title="New", category="pastry", price=12.5)
    ])

    kept = state.products["P1"].product
    assert kept.title == "New"          # title is always sent, so it always wins
    assert kept.category == "pastry"
    assert kept.price == 12.5


# -- P1: every endpoint must see ingested history, not just /predict ------------------


def test_production_plan_uses_ingested_sales(client):
    """The plan decides how much is baked. It must move when real sales arrive.

    Before this, /integration/restomind/production-plan called the stateless bridge
    and never consulted the registry, so a manager could upload a year of history,
    watch the prediction screen change, and see the production plan sit exactly where
    it was -- forecasting from the owner's estimate forever.
    """
    rid, pid = "PLAN_R1", "PLAN_P1"
    product = {
        "productId": pid, "title": "عيش بلدي", "category": "خبز",
        "price": 2.5, "freshnessWindow": 1, "avgDailySales": 200,
    }
    body = {"restaurantId": rid, "date": "2025-02-10", "products": [product]}

    before = client.post("/integration/restomind/production-plan", json=body).json()
    assert before["items"][0]["levelSource"] == "owner_estimate"

    # Real sales run far below the owner's 200/day guess.
    client.post("/integration/restomind/ingest", json={
        "restaurantId": rid,
        "records": _sales(pid, _SEED_DAYS, 60),
        "products": [product],
    })

    after = client.post("/integration/restomind/production-plan", json=body).json()
    assert after["items"][0]["levelSource"] == "learned_from_sales"
    assert after["items"][0]["recommendedQty"] < before["items"][0]["recommendedQty"], (
        "ingested sales did not reach the production plan"
    )


def test_production_plan_is_per_tenant(client):
    """One restaurant's history must not leak into another's plan."""
    product = {"productId": "ISO_P", "title": "كنافة", "category": "حلويات",
               "price": 45, "freshnessWindow": 2, "avgDailySales": 100}

    client.post("/integration/restomind/ingest", json={
        "restaurantId": "ISO_A",
        "records": _sales("ISO_P", _SEED_DAYS, 20),
        "products": [product],
    })

    body_b = {"restaurantId": "ISO_B", "date": "2025-02-10", "products": [product]}
    plan_b = client.post("/integration/restomind/production-plan", json=body_b).json()
    assert plan_b["items"][0]["levelSource"] == "owner_estimate"


def test_surplus_offers_uses_ingested_sales(client):
    """Expected sell-through decides whether stock is at risk, so it needs the level.

    A product that really sells 15/day, held against an owner estimate of 300/day,
    looks like it will clear everything on the shelf and is never flagged.
    """
    rid, pid = "SURP_R1", "SURP_P1"
    stock_item = {
        "productId": pid, "title": "بسبوسة", "category": "حلويات",
        "price": 20.0, "freshnessWindow": 1, "avgDailySales": 300,
        "currentStock": 60,
    }
    body = {
        "restaurantId": rid,
        "timestamp": "2025-02-10T20:00:00+02:00",
        "closeHour": 22,
        "stock": [stock_item],
    }

    before = client.post("/integration/restomind/surplus-offers", json=body).json()

    client.post("/integration/restomind/ingest", json={
        "restaurantId": rid,
        "records": _sales(pid, _SEED_DAYS, 15),
        "products": [{k: v for k, v in stock_item.items() if k != "currentStock"}],
    })

    after = client.post("/integration/restomind/surplus-offers", json=body).json()

    # With a true level of ~15/day this stock cannot clear, so it must now be flagged.
    assert len(after["itemsAtRisk"]) >= len(before["itemsAtRisk"])
    assert after["itemsAtRisk"], "learned level did not reach the surplus scan"


def test_plan_reports_the_level_it_used(client):
    """baseDailyLevel makes a plan auditable: which number produced this quantity."""
    rid, pid = "PLAN_R2", "PLAN_P2"
    product = {"productId": pid, "title": "فطير", "category": "معجنات",
               "avgDailySales": 77, "price": 30, "freshnessWindow": 2}
    plan = client.post("/integration/restomind/production-plan", json={
        "restaurantId": rid, "date": "2025-02-10", "products": [product],
    }).json()
    assert plan["items"][0]["baseDailyLevel"] == 77


# -- P2: a measured mean carries its window's calendar and must be stripped -----------


def test_ramadan_measured_mean_is_not_double_counted(client):
    """A mean measured INSIDE Ramadan must not get the Ramadan multiplier again.

    The caller's 14-day lookback is a raw mean over whatever days it covered. The
    bridge multiplies `base` by the target day's calendar multiplier, so `base` has
    to be an ordinary-day level. Feeding a Ramadan-inflated mean in as `base` applies
    Ramadan twice -- over-forecast, over-produce, waste, in the exact season the
    product exists to get right.
    """
    from app.core.egypt_calendar import CALENDAR

    # Find a Ramadan day and a run of days around it, from the calendar itself
    # rather than hardcoding a Hijri date.
    ramadan_day = next(
        d for d in (dt.date(2025, 3, 1) + dt.timedelta(days=i) for i in range(30))
        if CALENDAR.features(d)["is_ramadan"]
    )
    window_start = ramadan_day - dt.timedelta(days=13)

    product = {
        "productId": "DS_P1", "title": "كنافة", "category": "حلويات شرقية",
        "price": 45, "freshnessWindow": 2, "avgDailySales": 300,
    }
    body = {"restaurantId": "DS_R1", "date": ramadan_day.isoformat(),
            "products": [product]}

    # Same number, once declared as a raw measurement over a Ramadan window and
    # once as a plain ordinary-day estimate.
    without_window = client.post(
        "/integration/restomind/production-plan", json=body).json()
    with_window = client.post("/integration/restomind/production-plan", json={
        **body,
        "products": [{
            **product,
            "avgDailySalesWindow": {
                "from": window_start.isoformat(), "to": ramadan_day.isoformat(),
            },
        }],
    }).json()

    declared = with_window["items"][0]["recommendedQty"]
    undeclared = without_window["items"][0]["recommendedQty"]
    assert declared < undeclared, (
        "a Ramadan-measured mean was not deseasonalised, so Ramadan was applied twice"
    )


def test_deseasonalise_is_a_noop_on_an_ordinary_window():
    """A window of plain days must leave the mean essentially untouched."""
    from app.integration.restomind import deseasonalise

    # Mid-January 2025: no Ramadan, no Eid, no kahk season.
    start, end = dt.date(2025, 1, 13), dt.date(2025, 1, 26)
    out = deseasonalise(100.0, "حلويات شرقية", (start, end))
    # Weekends still sit inside any two-week window, so allow a modest band --
    # the point is that an ordinary window does not move the level much.
    assert 80.0 < out < 125.0


def test_deseasonalise_survives_a_reversed_or_degenerate_window():
    from app.integration.restomind import deseasonalise

    same_day = (dt.date(2025, 1, 15), dt.date(2025, 1, 15))
    assert deseasonalise(50.0, "خبز", same_day) > 0

    reversed_window = (dt.date(2025, 1, 26), dt.date(2025, 1, 13))
    forward_window = (dt.date(2025, 1, 13), dt.date(2025, 1, 26))
    assert deseasonalise(50.0, "خبز", reversed_window) == pytest.approx(
        deseasonalise(50.0, "خبز", forward_window)
    )


def test_owner_estimate_without_a_window_is_used_as_is(client):
    """No window means "this is already an ordinary-day figure" -- do not adjust it."""
    plan = client.post("/integration/restomind/production-plan", json={
        "restaurantId": "DS_R2", "date": "2025-01-15",
        "products": [{"productId": "DS_P2", "title": "فطير", "category": "معجنات",
                      "avgDailySales": 77, "price": 30, "freshnessWindow": 2}],
    }).json()
    assert plan["items"][0]["baseDailyLevel"] == 77
