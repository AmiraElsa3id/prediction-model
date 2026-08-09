"""MongoRegistryStore against a REAL local MongoDB -- not a mock.

Skipped automatically (not failed) when no MongoDB is reachable, so the rest of the
suite stays runnable without one. Uses a throwaway database, dropped in a fixture
teardown, so it never touches the RestoMind backend's own `restomind` database or
leaves state behind between runs.
"""

import datetime as dt
import uuid

import pandas as pd
import pytest

from app.integration.registry import MIN_DAYS_FOR_LEARNED

# Same sizing as test_registry.py: 2.2x the threshold guarantees a surplus of quiet
# (non-weekend, non-event) days so the learned level is actually reached.
_SEED_DAYS = int(MIN_DAYS_FOR_LEARNED * 2.2)

pymongo = pytest.importorskip("pymongo")

MONGO_URL = "mongodb://127.0.0.1:27017"


def _mongo_reachable() -> bool:
    try:
        client = pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=1000)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mongo_reachable(), reason="no local MongoDB reachable on 27017"
)


@pytest.fixture
def db_name():
    # Unique per test run so parallel runs and reruns never collide, and so a failed
    # teardown from a previous run cannot leak state into this one.
    name = f"prediction_model_test_{uuid.uuid4().hex[:10]}"
    yield name
    pymongo.MongoClient(MONGO_URL).drop_database(name)


def test_round_trips_through_a_real_mongo_document(db_name):
    from app.integration.mongo_store import MongoRegistryStore
    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput

    store = MongoRegistryStore(MONGO_URL, db_name)
    reg = RestaurantRegistry(store=store)

    rows = pd.DataFrame([
        {"date": d, "productId": "p1", "salesQty": 80}
        for d in pd.date_range("2025-01-06", periods=_SEED_DAYS).strftime("%Y-%m-%d")
    ])
    reg.ingest("R1", rows, [ProductInput(product_id="p1", title="Bread", category="bread")])

    assert reg.status("R1")["usingLearnedLevel"] == 1
    learned = reg.status("R1")["items"][0]["learnedLevel"]
    assert learned is not None


def test_survives_a_process_restart_via_mongo(db_name):
    """The actual point of P12: kill the process, come back, data is still there."""
    from app.integration.mongo_store import MongoRegistryStore
    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput

    rows = pd.DataFrame([
        {"date": d, "productId": "p1", "salesQty": 55}
        for d in pd.date_range("2025-01-06", periods=_SEED_DAYS).strftime("%Y-%m-%d")
    ])

    first_store = MongoRegistryStore(MONGO_URL, db_name)
    first = RestaurantRegistry(store=first_store)
    first.ingest("R1", rows, [ProductInput(product_id="p1", title="Bread")])
    learned = first.status("R1")["items"][0]["learnedLevel"]

    # A brand-new process would construct a brand-new client and registry against the
    # same database. Nothing here shares in-memory state with `first`.
    second_store = MongoRegistryStore(MONGO_URL, db_name)
    second = RestaurantRegistry(store=second_store)
    status = second.status("R1")

    assert status["productsTracked"] == 1
    assert status["usingLearnedLevel"] == 1
    assert status["items"][0]["learnedLevel"] == learned
    assert status["items"][0]["title"] == "Bread"


def test_two_restaurants_do_not_race_on_the_same_document(db_name):
    """Each restaurant is its own document -- unlike the JSON store, concurrent
    ingests for DIFFERENT restaurants cannot lose each other's writes."""
    from app.integration.mongo_store import MongoRegistryStore
    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput

    store = MongoRegistryStore(MONGO_URL, db_name)
    reg = RestaurantRegistry(store=store)

    rows_a = pd.DataFrame([{"date": "2025-01-06", "productId": "pa", "salesQty": 10}])
    rows_b = pd.DataFrame([{"date": "2025-01-06", "productId": "pb", "salesQty": 20}])

    reg.ingest("A", rows_a, [ProductInput(product_id="pa", title="A-item")])
    reg.ingest("B", rows_b, [ProductInput(product_id="pb", title="B-item")])

    fresh = RestaurantRegistry(store=MongoRegistryStore(MONGO_URL, db_name))
    assert fresh.status("A")["productsTracked"] == 1
    assert fresh.status("B")["productsTracked"] == 1
    assert fresh.status("A")["items"][0]["title"] == "A-item"
    assert fresh.status("B")["items"][0]["title"] == "B-item"


def test_predict_and_plan_calls_persist_economics_without_an_ingest(db_name):
    """Regression test for the bug found while wiring this in: production-plan and
    surplus-offers used to call `state.upsert_products` directly, bypassing save
    entirely, so a restaurant whose ONLY calls were /predict or /production-plan had
    its price/freshnessWindow held in memory only -- gone on restart."""
    from app.integration.mongo_store import MongoRegistryStore
    from app.integration.registry import RestaurantRegistry
    from app.integration.restomind import ProductInput

    store = MongoRegistryStore(MONGO_URL, db_name)
    reg = RestaurantRegistry(store=store)

    # upsert_products alone -- no ingest, no sales history.
    reg.upsert_products("R1", [
        ProductInput(product_id="p1", title="Croissant", price=18.0, freshness_window=2.0)
    ])

    fresh = RestaurantRegistry(store=MongoRegistryStore(MONGO_URL, db_name))
    state = fresh.get("R1")
    assert state.products["p1"].product.price == 18.0
    assert state.products["p1"].product.freshness_window == 2.0


def test_construction_fails_fast_when_mongo_is_unreachable():
    """An unreachable Mongo must not silently start the service with an empty store --
    that would make 'every restaurant's history just vanished' invisible."""
    from app.integration.mongo_store import MongoRegistryStore

    with pytest.raises(ConnectionError):
        MongoRegistryStore(
            "mongodb://127.0.0.1:1",  # nothing listens here
            "prediction_model_test_unreachable",
            server_selection_timeout_ms=300,
        )
