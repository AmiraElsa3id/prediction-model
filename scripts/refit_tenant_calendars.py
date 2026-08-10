"""Fit calendar effects for restaurants whose state predates them.

Run:  .venv/bin/python -m scripts.refit_tenant_calendars [--dry-run]

Restaurant state written before `app/integration/tenant_calendar.py` existed carries
history and learned levels but no calendar, so `/integration/restomind/production-plan`
keeps answering every date with the same flat quantity until the next `/ingest` call
refits it. That is invisible from the outside -- the response looks perfectly normal, it
just never changes -- so waiting for the next nightly ingest is a bad way to find out.

This replays each restaurant's ALREADY-PERSISTED history through `ingest`, which is the
same code path a real nightly call takes. Levels are recomputed identically (the history
is unchanged), and the calendar is fitted as a side effect. Nothing is downloaded, no
sales are invented, and a restaurant with too little history simply stays flat.

Reads the same store the API does (MONGO_URL / REGISTRY_STORE), so run it against the
same environment the service uses.
"""

from __future__ import annotations

import os
import sys

from app.integration.registry import RestaurantRegistry


def _open_registry() -> RestaurantRegistry:
    """The same three-way store selection `app/api/main.py`'s lifespan makes."""
    mongo_url = os.getenv("MONGO_URL")
    json_store_path = os.getenv("REGISTRY_STORE", "data/registry.json")
    if not mongo_url:
        return RestaurantRegistry(persist_path=json_store_path or None)

    from app.integration.mongo_store import DualRegistryStore, MongoRegistryStore
    from app.integration.registry import JsonFileRegistryStore

    mongo_store = MongoRegistryStore(mongo_url, os.getenv("MONGO_DB", "restomind_ai"))
    if json_store_path:
        return RestaurantRegistry(
            store=DualRegistryStore(mongo_store, JsonFileRegistryStore(json_store_path))
        )
    return RestaurantRegistry(store=mongo_store)


def main(dry_run: bool = False) -> int:
    registry = _open_registry()
    restaurant_ids = registry.restaurant_ids()
    if not restaurant_ids:
        print("No restaurants in the store -- nothing to refit.")
        return 0

    for rid in restaurant_ids:
        state = registry.get(rid)
        if state.history is None or state.history.empty:
            print(f"{rid}: no history, skipped")
            continue
        if state.calendar is not None:
            print(f"{rid}: already calendar-aware ({len(state.calendar.products)} products)")
            continue
        if dry_run:
            print(f"{rid}: would refit from {len(state.history)} rows")
            continue

        rows = state.history.copy()
        rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
        registry.ingest(rid, rows)

        calendar = registry.get(rid).calendar
        learned = len(calendar.products) if calendar else 0
        print(f"{rid}: refit from {len(rows)} rows -> {learned} products calendar-aware")

    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
