"""MongoDB-backed persistence for the per-restaurant registry.

WHY THIS EXISTS
----------------
`RestaurantRegistry` (app/integration/registry.py) already moved off pickle onto a
single local JSON file (`data/registry.json` by default) -- a real improvement, but
still not a database: one file on one machine's disk means no horizontal scaling (a
second instance of this service would not see the first's learned levels), no
external querying ("which restaurants have low-confidence levels" from outside the
process), and every `ingest()` call re-serialises every tenant's entire state to
rewrite the whole file.

`app/integration/connect_restomind.py` already connects directly to the same
MongoDB RestoMindAPI uses (`MongoClient(mongo_url).get_default_database()`) -- this
reuses that exact pattern rather than introducing a second datastore.

WHAT'S STORED
-------------
One document per restaurant in the `ai_registry_state` collection, `_id` = the
restaurant id, body = exactly what `RestaurantState.to_dict()` already produces (that
serialisation is reused unmodified -- it's already correct and already tested, see
tests/test_registry.py's round-trip tests).
"""

from __future__ import annotations

import os

COLLECTION_NAME = "ai_registry_state"


class MongoRegistryStore:
    """Thin persistence layer: load every restaurant's state at startup, write one
    restaurant's state through on every ingest. No query logic beyond that lives
    here -- RestaurantRegistry still owns the in-memory read-through cache and all
    the actual level-learning logic; this class only knows how to get a dict in and
    out of Mongo.
    """

    def __init__(self, mongo_url: str | None = None) -> None:
        from pymongo import MongoClient

        self.mongo_url = mongo_url or os.environ["MONGO_URL"]
        self._client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
        self._collection = self._client.get_default_database()[COLLECTION_NAME]
        self._collection.create_index("restaurantId", unique=True)

    def load_all(self) -> dict[str, dict]:
        """Every restaurant's raw state dict, keyed by restaurantId -- the same
        shape RestaurantState.from_dict() already knows how to consume."""
        return {
            doc["restaurantId"]: doc
            for doc in self._collection.find({})
        }

    def save(self, restaurant_id: str, state_dict: dict) -> None:
        """Upsert one restaurant's full state. `state_dict` is whatever
        RestaurantState.to_dict() returned -- stored as-is, `_id` set to the
        restaurant id so a second instance's `load_all()` sees this write."""
        payload = dict(state_dict)
        payload["_id"] = restaurant_id
        payload["restaurantId"] = restaurant_id
        self._collection.replace_one({"_id": restaurant_id}, payload, upsert=True)
