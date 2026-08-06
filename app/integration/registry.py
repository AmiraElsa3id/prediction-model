"""Per-restaurant state for the RestoMind bridge (multi-tenant).

Answers the owner's question: *"if I seed sales data, does it change the model?"*

The stateless bridge (`restomind.py`) forecasts every product purely from category priors
+ the owner's `avgDailySales` estimate — so seeding a MENU changes what gets predicted, but
not the numbers. This registry adds the missing piece: it stores each restaurant's sales
history separately and **learns the real demand level per product from it**. Once a product
has enough real days, its forecast uses the learned level instead of the owner's guess —
so seeding sales now visibly moves the prediction.

Scope of THIS layer: multi-tenant state + data-driven LEVEL (still rule-based for the
calendar shape). Wiring the full trained `CalendarDecomposed` model per restaurant (which
needs per-product economics generalised off the built-in catalogue) is the next step —
tracked in HANDOFF.md §9.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.integration.restomind import ProductInput, map_category, predict_week

# Real days a product needs before its learned level is trusted over the owner estimate.
MIN_DAYS_FOR_LEARNED = 14
# Days beyond which we call the level well-established (higher confidence).
CONFIDENT_DAYS = 90
# Window of recent quiet (non-event) days used to estimate the ordinary level.
QUIET_WINDOW = 42


@dataclass
class ProductState:
    product: ProductInput
    observed_days: int = 0
    learned_level: float | None = None


@dataclass
class RestaurantState:
    restaurant_id: str
    products: dict[str, ProductState] = field(default_factory=dict)
    history: pd.DataFrame | None = None

    def upsert_products(self, products: list[ProductInput]) -> None:
        for p in products:
            if p.product_id in self.products:
                self.products[p.product_id].product = p
            else:
                self.products[p.product_id] = ProductState(product=p)

    def ingest(self, records: pd.DataFrame) -> None:
        """Append sales rows and re-learn each product's ordinary-day level.

        `records` columns: date, productId, salesQty. The learned level is the mean of
        recent NON-EVENT days, so the calendar multiplier re-adds Ramadan/Eid on top at
        predict time rather than being baked into the level.
        """
        records = records.copy()
        records["date"] = pd.to_datetime(records["date"])
        self.history = records if self.history is None else pd.concat(
            [self.history, records], ignore_index=True
        ).drop_duplicates(subset=["date", "productId"], keep="last")

        cal = CALENDAR.feature_frame(self.history["date"].min(), self.history["date"].max())
        cal["date"] = pd.to_datetime(cal["date"])
        flags = ["date", "is_ramadan", "is_public_holiday", "is_kahk_window", "is_weekend"]
        merged = self.history.merge(cal[flags], on="date", how="left")

        for pid, grp in merged.groupby("productId"):
            st = self.products.get(pid)
            if st is None:
                continue
            st.observed_days = grp["date"].nunique()
            quiet = grp[
                (grp["is_ramadan"] == 0) & (grp["is_public_holiday"] == 0)
                & (grp["is_kahk_window"] == 0) & (grp["is_weekend"] == 0)
            ].sort_values("date").tail(QUIET_WINDOW)
            if len(quiet) >= MIN_DAYS_FOR_LEARNED:
                st.learned_level = float(quiet["salesQty"].mean())

    def _level_and_mode(self, pid: str) -> tuple[float | None, str, str]:
        st = self.products.get(pid)
        if st is None or st.learned_level is None:
            return None, "rule_based", "low"
        confidence = "medium" if st.observed_days >= CONFIDENT_DAYS else "low"
        return st.learned_level, "rule_based_learned", confidence

    def to_dict(self) -> dict:
        return {
            "restaurantId": self.restaurant_id,
            "products": {
                pid: {
                    "product": {
                        "product_id": st.product.product_id,
                        "title": st.product.title,
                        "category": st.product.category,
                        "price": st.product.price,
                        "freshness_window": st.product.freshness_window,
                        "avg_daily_sales": st.product.avg_daily_sales,
                    },
                    "observed_days": st.observed_days,
                    "learned_level": st.learned_level,
                }
                for pid, st in self.products.items()
            },
            "history": (
                []
                if self.history is None
                else [
                    {
                        "date": str(r["date"])[:10],
                        "productId": r["productId"],
                        "salesQty": int(r["salesQty"]),
                    }
                    for r in self.history.to_dict("records")
                ]
            ),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "RestaurantState":
        state = cls(restaurant_id=raw["restaurantId"])
        for pid, p in raw.get("products", {}).items():
            state.products[pid] = ProductState(
                product=ProductInput(**p["product"]),
                observed_days=p.get("observed_days", 0),
                learned_level=p.get("learned_level"),
            )
        rows = raw.get("history") or []
        if rows:
            hist = pd.DataFrame(rows)
            hist["date"] = pd.to_datetime(hist["date"])
            state.history = hist
        return state


class RestaurantRegistry:
    """Holds `RestaurantState` per restaurantId, backed by one of two persistence
    modes -- MongoDB (preferred; see mongo_store.py) or a local JSON file (kept for
    environments without a database configured). The in-memory `_states` dict is
    always a read-through cache regardless of mode: predictions never pay a DB
    round-trip, only `ingest()` (already an I/O-bound call) does.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        mongo_url: str | None = None,
    ) -> None:
        self._states: dict[str, RestaurantState] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        self._mongo = None

        if mongo_url:
            from app.integration.mongo_store import MongoRegistryStore

            try:
                self._mongo = MongoRegistryStore(mongo_url)
            except Exception:
                # Unreachable/misconfigured MONGO_URL at startup must degrade to an
                # empty, in-memory-only registry, not take the whole service down --
                # same "start empty rather than crash" discipline as a corrupt JSON
                # store already has below.
                self._mongo = None
            else:
                self._load()
        elif self.persist_path and self.persist_path.exists():
            self._load()

    def _load(self) -> None:
        if self._mongo is not None:
            try:
                raw = self._mongo.load_all()
            except Exception:
                self._states = {}  # Mongo unreachable at startup -> start empty, not crash
                return
            self._states = {rid: RestaurantState.from_dict(s) for rid, s in raw.items()}
            return

        # JSON, not pickle: this file is read at startup, and unpickling is
        # arbitrary code execution if anything can write to that path.
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self._states = {
                rid: RestaurantState.from_dict(s) for rid, s in raw.items()
            }
        except Exception:
            self._states = {}   # corrupt/old file -> start fresh rather than crash

    def _save(self, restaurant_id: str | None = None) -> None:
        """Persist. `restaurant_id`, when given, lets the Mongo path write through
        just the one tenant that changed instead of re-serialising every tenant on
        every ingest -- the exact per-write cost problem a single JSON file has."""
        if self._mongo is not None:
            if restaurant_id is not None:
                self._mongo.save(restaurant_id, self._states[restaurant_id].to_dict())
            else:
                for rid, st in self._states.items():
                    self._mongo.save(rid, st.to_dict())
            return

        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a crash mid-write must not leave a truncated store behind.
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        payload = {rid: st.to_dict() for rid, st in self._states.items()}
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.persist_path)

    def get(self, restaurant_id: str) -> RestaurantState:
        return self._states.setdefault(restaurant_id, RestaurantState(restaurant_id))

    def ingest(
        self, restaurant_id: str, records: pd.DataFrame,
        products: list[ProductInput] | None = None,
    ) -> dict:
        state = self.get(restaurant_id)
        if products:
            state.upsert_products(products)
        # Register any product seen in the data but not explicitly provided.
        for pid in records["productId"].unique():
            if pid not in state.products:
                state.upsert_products([ProductInput(product_id=pid, title=pid)])
        state.ingest(records)
        self._save(restaurant_id)
        return {
            "restaurantId": restaurant_id,
            "rowsIngested": int(len(records)),
            "productsTracked": len(state.products),
            "daysByProduct": {pid: st.observed_days for pid, st in state.products.items()},
            "learnedLevels": {
                pid: round(st.learned_level, 1)
                for pid, st in state.products.items() if st.learned_level is not None
            },
        }

    def predict_week(
        self, restaurant_id: str, product: ProductInput, week_start: dt.date,
        promotion_active: bool = False,
    ) -> dict:
        """Weekly prediction that uses the restaurant's learned level when available."""
        state = self.get(restaurant_id)
        state.upsert_products([product])
        level, mode, confidence = state._level_and_mode(product.product_id)
        return predict_week(
            restaurant_id, product, week_start,
            promotion_active=promotion_active, level=level, mode=mode, confidence=confidence,
        )

    def status(self, restaurant_id: str) -> dict:
        state = self.get(restaurant_id)
        items = []
        for pid, st in state.products.items():
            items.append({
                "productId": pid,
                "title": st.product.title,
                "observedDays": st.observed_days,
                "levelSource": "learned_from_sales" if st.learned_level is not None else "owner_estimate",
                "learnedLevel": round(st.learned_level, 1) if st.learned_level is not None else None,
            })
        return {
            "restaurantId": restaurant_id,
            "productsTracked": len(items),
            "usingLearnedLevel": sum(1 for i in items if i["levelSource"] == "learned_from_sales"),
            "items": items,
        }
