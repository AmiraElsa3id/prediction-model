"""Per-restaurant researched priors: storage + the pending_review -> approved gate.

Reuses the same MongoDB connection pattern as mongo_store.py / connect_restomind.py.
Deliberately separate from the forecast path's actual multiplier lookup
(app/integration/restomind.py's `researched_priors_for`) -- storing a result here
does NOT make it live. Only `approve()` does that, and only for the one
(restaurantId, category, event) triple explicitly approved -- see plan.md Part C's
guardrails and app/agents/market_research.py's module docstring for why.
"""

from __future__ import annotations

import datetime as dt
import os

COLLECTION_NAME = "ai_researched_priors"


class PriorsStore:
    def __init__(self, mongo_url: str | None = None) -> None:
        from pymongo import MongoClient

        self.mongo_url = mongo_url or os.environ["MONGO_URL"]
        self._client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
        self._collection = self._client.get_default_database()[COLLECTION_NAME]
        self._collection.create_index(
            [("restaurantId", 1), ("category", 1), ("event", 1)], unique=True
        )

    def save_researched(self, restaurant_id: str, result: dict) -> None:
        """Store one research_one() result as pending_review. A prior "approved"
        result for the same (category, event) is NOT overwritten by re-running
        research -- re-researching must not silently undo an explicit approval."""
        if result.get("status") != "researched":
            return  # only successfully-researched results are worth storing at all

        existing = self._collection.find_one({
            "restaurantId": restaurant_id,
            "category": result["category"],
            "event": result["event"],
        })
        if existing and existing.get("reviewStatus") == "approved":
            return

        self._collection.replace_one(
            {"restaurantId": restaurant_id, "category": result["category"], "event": result["event"]},
            {
                "restaurantId": restaurant_id,
                "category": result["category"],
                "event": result["event"],
                "multiplier": result["multiplier"],
                "confidence": result.get("confidence", "low"),
                "reasoning": result.get("reasoning", ""),
                "sources": result.get("sources", []),
                "reviewStatus": "pending_review",
                "researchedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            upsert=True,
        )

    def approve(self, restaurant_id: str, category: str, event: str) -> bool:
        """Explicitly gate one researched multiplier into the live forecast path.
        Returns False if there's nothing pending for that triple."""
        result = self._collection.update_one(
            {"restaurantId": restaurant_id, "category": category, "event": event,
             "reviewStatus": "pending_review"},
            {"$set": {"reviewStatus": "approved",
                      "approvedAt": dt.datetime.now(dt.timezone.utc).isoformat()}},
        )
        return result.modified_count > 0

    def approved_multipliers(self, restaurant_id: str, category: str) -> dict[str, float]:
        """What the forecast path actually reads: only APPROVED multipliers for this
        restaurant+category, as a plain {event: multiplier} dict ready to merge over
        the global category_priors() default."""
        docs = self._collection.find({
            "restaurantId": restaurant_id, "category": category, "reviewStatus": "approved",
        })
        return {d["event"]: d["multiplier"] for d in docs}

    def list_pending(self, restaurant_id: str) -> list[dict]:
        return list(self._collection.find(
            {"restaurantId": restaurant_id, "reviewStatus": "pending_review"},
            {"_id": 0},
        ))
