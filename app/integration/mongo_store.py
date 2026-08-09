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
import logging
from typing import Protocol

_log = logging.getLogger(__name__)


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


class DualRegistryStore:
    """Writes every save to a primary store AND mirrors it to a secondary one.

    Satisfies `RegistryStore`, so `RestaurantRegistry` cannot tell it apart from a
    single backend. Wired up when both MONGO_URL and REGISTRY_STORE are configured
    (see `app/api/main.py`).

    The two halves are deliberately NOT equal partners:

    * **Primary is the source of truth.** A failed primary write propagates and fails
      the request -- silently losing a restaurant's learned levels is the exact failure
      this whole persistence layer exists to prevent.
    * **The mirror is best-effort.** A failed mirror write is logged and swallowed. A
      full disk or a locked file must not take down a service whose real store is
      healthy; the alternative trades one durable copy for zero availability.

    So the mirror is a convenience copy (inspect it by eye, keep a local snapshot),
    never a second authority. If you need both to be authoritative you need a
    distributed transaction, which is far more machinery than a dev-convenience mirror
    justifies.
    """

    def __init__(self, primary: "RegistryStore", mirror: "RegistryStore") -> None:
        self.primary = primary
        self.mirror = mirror

    def load_all(self) -> dict[str, dict]:
        """Primary wins. The mirror is read ONLY when the primary is entirely empty.

        That one case is the migration path: an existing JSON file's history is adopted
        by a fresh Mongo on first boot, and written back to Mongo on the next save.

        It is deliberately all-or-nothing rather than a per-restaurant merge. Merging
        would resurrect a restaurant deleted from the primary just because a stale
        mirror still listed it -- silently, and with no way to ever delete it again.
        """
        primary_state = self.primary.load_all()
        if primary_state:
            return primary_state

        mirror_state = self.mirror.load_all()
        if mirror_state:
            _log.warning(
                "Registry primary store is empty; adopting %d restaurant(s) from the "
                "mirror. This is expected on a first run against a fresh primary, and "
                "unexpected afterwards -- if you see it on every boot, the primary's "
                "writes are not landing.",
                len(mirror_state),
            )
        return mirror_state

    def save_restaurant(self, restaurant_id: str, doc: dict) -> None:
        # Primary first, and un-caught: if the durable copy did not land, the caller
        # must hear about it rather than be reassured by a successful mirror write.
        self.primary.save_restaurant(restaurant_id, doc)

        try:
            self.mirror.save_restaurant(restaurant_id, doc)
        except Exception as exc:
            # Broad on purpose. The mirror can be any RegistryStore, so the failures are
            # open-ended (OSError, PermissionError, a driver error), and every one of
            # them has the same correct response: the primary already succeeded, so the
            # data is safe -- say so and carry on.
            _log.warning(
                "Registry mirror write failed for restaurant %s (%s: %s). The primary "
                "store succeeded, so no data was lost; the mirror is now stale.",
                restaurant_id, type(exc).__name__, exc,
            )
