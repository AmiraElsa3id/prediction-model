"""MongoDB-backed persistence for the per-restaurant registry.

Replaces the in-memory-only / single-JSON-file store (`registry.py`'s previous only
option) with real durable storage: one document per restaurant, so a crash or redeploy
does not discard every tenant's learned demand levels and ingested history.

Deliberately narrow: this module knows how to load and save opaque state dicts (the
shape `RestaurantState.to_dict()` / `.from_dict()` already define) under a
`restaurant_id` key. It has no opinion on what is inside them -- that stays
`registry.py`'s job, so the storage backend can change again without touching the model
logic that already exists and is tested.

Uses a separate database from the RestoMind backend's own MongoDB (default
`restomind_ai`, distinct from their `restomind`). The model reads and writes only its
own database: it must not read the backend's `sales_transactions` directly, or every
future backend schema migration becomes a breaking change here too. The
`/integration/restomind/ingest` contract is what keeps the two services decoupled, and
that stays true with this store.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol


class RegistryStore(Protocol):
    """What `RestaurantRegistry` needs from a persistence backend.

    `JsonFileRegistryStore` (in registry.py) and `MongoRegistryStore` (here) both
    satisfy this. Anything else that can load and save per-restaurant state dicts can
    too -- the registry does not know or care which one it has.
    """

    def load_all(self) -> dict[str, dict]:
        """Every persisted restaurant's raw state dict, keyed by restaurantId."""
        ...

    def save_restaurant(self, restaurant_id: str, doc: dict) -> None:
        """Persist ONE restaurant's state. Must not touch any other restaurant's."""
        ...


class MongoRegistryStore:
    """One MongoDB document per restaurant, upserted on every save.

    This is the fix for the JSON store's write amplification: `JsonFileRegistryStore`
    must rewrite its entire file on every ingest because that is the nature of a single
    JSON file, so its cost grows with the whole corpus rather than the one restaurant
    that changed, and two restaurants saving concurrently race on the same file. A Mongo
    upsert touches exactly one document and is atomic per-document, so neither problem
    exists here.
    """

    def __init__(
        self,
        url: str,
        db_name: str = "restomind_ai",
        collection_name: str = "restaurant_registry_state",
        server_selection_timeout_ms: int = 5000,
    ) -> None:
        # Imported lazily: pymongo is only required when Mongo persistence is actually
        # requested (MONGO_URL set), so the JSON-file and in-memory paths -- including
        # the whole existing test suite, which forces REGISTRY_STORE="" -- never need
        # it installed.
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError

        self._PyMongoError = PyMongoError
        self._client: MongoClient = MongoClient(
            url, serverSelectionTimeoutMS=server_selection_timeout_ms
        )
        self._collection = self._client[db_name][collection_name]

        # Fail at construction, not on the first request. Starting up with an
        # unreachable Mongo and silently falling back to an empty in-memory store
        # would make "every restaurant's learned history just vanished" invisible --
        # exactly the failure mode this module exists to remove.
        try:
            self._client.admin.command("ping")
        except PyMongoError as exc:
            raise ConnectionError(
                f"MongoRegistryStore could not reach MongoDB at the configured "
                f"MONGO_URL ({db_name}.{collection_name}): {exc}"
            ) from exc

        # restaurantId is the natural key; _id IS restaurantId rather than a separate
        # indexed field, so lookups and upserts are both single-document point
        # operations with no secondary index to maintain.
        self._collection.create_index("updatedAt")

    def load_all(self) -> dict[str, dict]:
        return {doc["_id"]: doc["state"] for doc in self._collection.find()}

    def save_restaurant(self, restaurant_id: str, doc: dict) -> None:
        self._collection.replace_one(
            {"_id": restaurant_id},
            {
                "_id": restaurant_id,
                "state": doc,
                "updatedAt": dt.datetime.now(dt.timezone.utc),
            },
            upsert=True,
        )

    def close(self) -> None:
        self._client.close()
