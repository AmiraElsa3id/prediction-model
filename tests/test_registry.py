"""Per-restaurant registry: seeding real sales must change predictions, per tenant."""

import datetime as dt

from fastapi.testclient import TestClient
import pytest

from app.api.main import app


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
        "restaurantId": rid, "records": _sales(pid, 30, 110),
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
        "restaurantId": "REG_A", "records": _sales(pid, 30, 200),
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
        "restaurantId": rid, "records": _sales(pid, 30, 90),
        "products": [{"productId": pid, "title": "فطير", "category": "مالح"}],
    })
    st = client.get(f"/integration/restomind/status/{rid}").json()
    assert st["usingLearnedLevel"] == 1
    assert st["items"][0]["learnedLevel"] is not None


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
        # 30 consecutive days guarantees >= 14 quiet weekdays.
        for d in pd.date_range("2025-01-06", periods=30).strftime("%Y-%m-%d")
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

    from app.api.main import app as real_app

    with TestClient(real_app) as c:
        rid, pid = "DEFAULT_R1", "DEFAULT_P1"
        resp = c.post("/integration/restomind/ingest", json={
            "restaurantId": rid,
            "records": [{"date": "2025-01-06", "productId": pid, "salesQty": 5}],
            "products": [{"productId": pid, "title": "Bread"}],
        })
        assert resp.status_code == 200

    assert (tmp_path / "data" / "registry.json").exists()


# -- MongoDB-backed persistence -------------------------------------------------------
# Skipped, not failed, when no local Mongo is reachable -- these exercise a real
# connection rather than mocking pymongo, so CI/dev environments without one still
# pass the rest of the suite. A separate DB name from RestoMindAPI's own, so this
# never touches real seeded data.

MONGO_TEST_URL = "mongodb://127.0.0.1:27017/prediction_model_test"


def _mongo_reachable() -> bool:
    try:
        from pymongo import MongoClient

        MongoClient(MONGO_TEST_URL, serverSelectionTimeoutMS=1000).admin.command("ping")
        return True
    except Exception:
        return False


requires_mongo = pytest.mark.skipif(
    not _mongo_reachable(), reason="no local MongoDB reachable at 127.0.0.1:27017"
)


@pytest.fixture
def mongo_registry():
    """A RestaurantRegistry over a clean Mongo collection, cleaned up after too --
    tests must not leave state that affects a later run."""
    from pymongo import MongoClient

    from app.integration.registry import RestaurantRegistry

    MongoClient(MONGO_TEST_URL).get_default_database()["ai_registry_state"].delete_many({})
    reg = RestaurantRegistry(mongo_url=MONGO_TEST_URL)
    yield reg
    MongoClient(MONGO_TEST_URL).get_default_database()["ai_registry_state"].delete_many({})


@requires_mongo
def test_registry_survives_a_restart_via_mongo(mongo_registry):
    """The actual point of Part B: kill the process, a fresh RestaurantRegistry
    over the same MONGO_URL must see the same learned levels -- not the JSON-file
    path, a real second connection to a real database."""
    import pandas as pd

    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput

    rows = pd.DataFrame([
        {"date": d, "productId": "p1", "salesQty": 100}
        for d in pd.date_range("2025-01-06", periods=30).strftime("%Y-%m-%d")
    ])
    mongo_registry.ingest("R1", rows, [ProductInput(product_id="p1", title="Bread", category="bread")])
    assert mongo_registry.status("R1")["usingLearnedLevel"] == 1
    learned = mongo_registry.status("R1")["items"][0]["learnedLevel"]

    second = RestaurantRegistry(mongo_url=MONGO_TEST_URL)  # simulates a restart
    status = second.status("R1")
    assert status["usingLearnedLevel"] == 1
    assert status["items"][0]["learnedLevel"] == learned
    assert status["items"][0]["title"] == "Bread"


@requires_mongo
def test_mongo_registry_is_per_tenant(mongo_registry):
    """Same isolation guarantee the JSON-file/in-memory path already has -- must
    hold across a real database too, since it's now a shared resource multiple
    service instances could write to."""
    import pandas as pd

    from app.integration.restomind import ProductInput

    rows = pd.DataFrame([
        {"date": d, "productId": "shared", "salesQty": 200}
        for d in pd.date_range("2025-01-06", periods=30).strftime("%Y-%m-%d")
    ])
    mongo_registry.ingest("TENANT_A", rows, [ProductInput(product_id="shared", title="X")])

    assert mongo_registry.status("TENANT_A")["usingLearnedLevel"] == 1
    assert mongo_registry.status("TENANT_B")["productsTracked"] == 0


@requires_mongo
def test_mongo_unreachable_at_startup_starts_empty_not_crashing():
    """A bad/unreachable MONGO_URL must degrade to an empty registry, not take the
    whole service down at startup."""
    from app.integration.registry import RestaurantRegistry

    reg = RestaurantRegistry(mongo_url="mongodb://127.0.0.1:1/nonexistent")
    assert reg.status("R1")["productsTracked"] == 0
