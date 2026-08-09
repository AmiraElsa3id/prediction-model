"""Per-restaurant state for the RestoMind bridge (multi-tenant).

Answers the owner's question: *"if I seed sales data, does it change the model?"*

The stateless bridge (`restomind.py`) forecasts every product from the owner's
`avgDailySales` estimate. This registry adds the missing piece: it stores each
restaurant's sales history separately and **learns the real demand level per product
from it**. Once a product has enough kept days, its forecast uses the learned level
instead of the owner's guess — so seeding sales now visually moves the prediction.

Scope of THIS layer: multi-tenant state + data-driven LEVEL. The rule-based calendar
multipliers the bridge once applied are gone (`rule_based.py` and `market_priors.py`
were removed; see `HANDOFF.md` §8). Products carrying a trained catalogue `sku` are now
forecast by the full trained `CalendarDecomposed` model (wired through the API);
generalising that to arbitrary per-restaurant products (economics off the built-in
catalogue) is tracked in `HANDOFF.md` §9.

Persistence is pluggable (see `RegistryStore` in `mongo_store.py`): a JSON file
(`JsonFileRegistryStore`, the original mechanism, kept for local dev and the
test suite) or MongoDB (`MongoRegistryStore` -- the durable option, one document per
restaurant, immune to the JSON file's whole-corpus rewrite on every save). Which one is
wired up is `app/api/main.py`'s decision; this module only knows the `RegistryStore`
protocol.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.integration.restomind import ProductInput, predict_week

if TYPE_CHECKING:
    from app.integration.mongo_store import RegistryStore
    from app.models.service import ForecastService

# Real days a product needs before its learned level is trusted over the owner estimate.
MIN_DAYS_FOR_LEARNED = 90
# Days beyond which we call the level well-established (higher confidence).
CONFIDENT_DAYS = 90
# Window of recent quiet (non-event) days used to estimate the ordinary level.
# Must not drop below MIN_DAYS_FOR_LEARNED: `.tail(QUIET_WINDOW)` caps the sample in
# ingest, so a smaller window would make the >= MIN_DAYS_FOR_LEARNED check impossible.
QUIET_WINDOW = 90


@dataclass
class ProductState:
    product: ProductInput
    observed_days: int = 0
    learned_level: float | None = None


# Fields where an absent value is genuinely "not provided by this caller" rather than
# a deliberate clear. `title` is excluded: it is required on every payload, so a change
# there is a real rename and must win. `price` defaults to 0.0 rather than None, so
# treat 0 as "not provided" too -- a product that is actually free is not a case the
# newsvendor economics can price anyway.
_MERGEABLE_FIELDS = ("category", "freshness_window", "avg_daily_sales", "sku","unit_cost")


def _merge_product(existing: ProductInput, incoming: ProductInput) -> ProductInput:
    """Overlay `incoming` onto `existing`, keeping known values the caller omitted."""
    merged = replace(existing, title=incoming.title)
    for name in _MERGEABLE_FIELDS:
        value = getattr(incoming, name)
        if value is not None:
            merged = replace(merged, **{name: value})
    if incoming.price:
        merged = replace(merged, price=incoming.price)
    return merged


@dataclass
class RestaurantState:
    restaurant_id: str
    products: dict[str, ProductState] = field(default_factory=dict)
    history: pd.DataFrame | None = None

    def upsert_products(self, products: list[ProductInput]) -> None:
        """Register or refresh products, MERGING rather than replacing.

        Callers do not all know the same things. `/predict` sends one product with a
        category and no economics; an ingest sends the catalogue with price and
        freshness_window. Replacing the stored ProductInput wholesale meant whichever
        call arrived last won, so a `/predict` could blank the price and shelf life a
        prior ingest had populated -- and those two fields are what the newsvendor
        service level q* is computed from.

        So a field is only overwritten when the incoming value actually says something.
        `None` means "I don't know", not "unset it". Clearing a field is therefore not
        expressible here, which is the right trade: nothing upstream ever needs to.
        """
        for p in products:
            existing = self.products.get(p.product_id)
            if existing is None:
                self.products[p.product_id] = ProductState(product=p)
                continue
            existing.product = _merge_product(existing.product, p)

    def ingest(self, records: pd.DataFrame) -> None:
        """Append sales rows and re-learn each product's ordinary-day level.

        `records` columns: date, productId, salesQty, and optionally productionQty /
        closingStock. The learned level is the mean of recent NON-EVENT, NON-STOCKOUT
        days, so the calendar multiplier re-adds Ramadan/Eid on top at predict time
        rather than being baked into the level.
        """
        records = records.copy()
        records["date"] = pd.to_datetime(records["date"])
        # Optional columns: a caller that only ever sends salesQty (the old minimal
        # shape) must keep working exactly as before -- missing columns become NaN,
        # which `is_stockout` below treats as "unknown, assume not stocked out".
        for col in ("productionQty", "closingStock"):
            if col not in records.columns:
                records[col] = pd.NA
        self.history = records if self.history is None else pd.concat(
            [self.history, records], ignore_index=True
        ).drop_duplicates(subset=["date", "productId"], keep="last")

        cal = CALENDAR.feature_frame(self.history["date"].min(), self.history["date"].max())
        cal["date"] = pd.to_datetime(cal["date"])
        flags = ["date", "is_ramadan", "is_public_holiday", "is_kahk_window", "is_weekend"]
        merged = self.history.merge(cal[flags], on="date", how="left")
        # A day the shelf sold out is supply-constrained: salesQty that day is a floor
        # on true demand, not the figure itself, so averaging it in like an ordinary
        # day would drag the learned level down. `closingStock` unknown (most callers,
        # today) defaults to "not a stockout" -- the pre-existing behaviour.
        merged["is_stockout"] = merged["closingStock"].fillna(1) == 0

        for pid, grp in merged.groupby("productId"):
            st = self.products.get(pid)
            if st is None:
                continue
            st.observed_days = grp["date"].nunique()
            quiet = grp[
                (grp["is_ramadan"] == 0) & (grp["is_public_holiday"] == 0)
                & (grp["is_kahk_window"] == 0) & (grp["is_weekend"] == 0)
                & (~grp["is_stockout"])
            ].sort_values("date").tail(QUIET_WINDOW)
            if len(quiet) >= MIN_DAYS_FOR_LEARNED:
                st.learned_level = float(quiet["salesQty"].mean())
            elif st.learned_level is not None:
                # Re-ingest re-evaluates every product: a level learned under an
                # older, lower threshold drops back to the estimate until the
                # product earns the new threshold again.
                st.learned_level = None

    def level_for(self, pid: str) -> tuple[float | None, str, str]:
        """`(level, mode, confidence)` for one product.

        Public because every endpoint that forecasts for a restaurant needs it, not
        just /predict. A `None` level means "no learned value, fall back to the
        owner's estimate" — the caller decides, this only reports. `mode` is
        `"training"` until the product has enough real days to earn a learned level,
        then `"learned"`.
        """
        st = self.products.get(pid)
        if st is None or st.learned_level is None:
            return None, "training", "low"
        confidence = "medium" if st.observed_days >= CONFIDENT_DAYS else "low"
        return st.learned_level, "learned", confidence

    def levels_for(self, pids: Iterable[str]) -> dict[str, tuple[float | None, str, str]]:
        """`level_for` across many products, for the batch endpoints."""
        return {pid: self.level_for(pid) for pid in pids}

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
                        "unit_cost": st.product.unit_cost,
                        "freshness_window": st.product.freshness_window,
                        "avg_daily_sales": st.product.avg_daily_sales,
                        "sku": st.product.sku,
                    },
                    "observed_days": st.observed_days,
                    "learned_level": st.learned_level,
                }
                for pid, st in self.products.items()
            },
            "history": (
                []
                if self.history is None
                else [_history_row_to_dict(r) for r in self.history.to_dict("records")]
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


def _history_row_to_dict(r: dict) -> dict:
    """One stored history row -- productionQty/closingStock only when actually known.

    `r` comes from `DataFrame.to_dict("records")`, so a column every row in this
    restaurant's history lacks a real value for reads back as NaN, not absent. Only
    write the key when there is a real value: an explicit `null` in the persisted doc
    would be indistinguishable from "the caller sent 0", which is a real, different
    answer (a genuinely sold-out day).
    """
    row = {
        "date": str(r["date"])[:10],
        "productId": r["productId"],
        "salesQty": int(r["salesQty"]),
    }
    for key in ("productionQty", "closingStock"):
        value = r.get(key)
        if pd.notna(value):
            row[key] = int(value)
    return row


class JsonFileRegistryStore:
    """The original persistence mechanism: every restaurant's state in one JSON file.

    Kept for local development (no MongoDB required) and for the test suite, which
    needs a store that does not depend on external infrastructure. `save_restaurant`
    still has to rewrite the whole file -- that is the nature of a single JSON file,
    not something this class can avoid -- which is exactly the cost `MongoRegistryStore`
    removes. Prefer Mongo for anything that is not local dev or tests.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Serialises read-modify-write: FastAPI runs /predict in a threadpool, so two
        # concurrent calls can otherwise both re-read the file, both re-write it (lost
        # update), and one `os.replace` lands while the other thread still has the file
        # open -- the Windows PermissionError this lock eliminates within a process.
        self._lock = threading.Lock()

    def load_all(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        # JSON, not pickle: this file is read at startup, and unpickling is
        # arbitrary code execution if anything can write to that path.
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}   # corrupt/old file -> start fresh rather than crash

    def save_restaurant(self, restaurant_id: str, doc: dict) -> None:
        # No partial write is possible in a single JSON file: read the current whole
        # state, replace one restaurant's entry, write the whole thing back. This is
        # the write-amplification MongoRegistryStore exists to remove.
        with self._lock:
            current = self.load_all()
            current[restaurant_id] = doc
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic: a crash mid-write must not leave a truncated store behind.
            # A unique temp name keeps concurrent writers (or a stale .tmp from a
            # crashed process) from sharing one lock target.
            tmp = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
            )
            try:
                tmp.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
                self._replace_with_retry(tmp, self.path)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    @staticmethod
    def _replace_with_retry(src: Path, dst: Path) -> None:
        # os.replace on Windows raises PermissionError when `src` or `dst` is
        # transiently held open (antivirus scan, a concurrent request's read).
        # python's file is already closed and flushed here, so the lock clears in
        # milliseconds; retry with a short backoff instead of failing the request.
        for attempt in range(10):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.02 * (attempt + 1))


class RestaurantRegistry:
    """Holds `RestaurantState` per restaurantId, durable across restarts.

    Pass `store=` a `RegistryStore` (typically `MongoRegistryStore`) for production
    persistence, or `persist_path=` for the JSON-file mechanism (local dev / tests).
    Passing neither keeps state in memory only, for tests that want isolation from any
    backing store at all.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        store: "RegistryStore | None" = None,
    ) -> None:
        self._states: dict[str, RestaurantState] = {}
        self.store = store or (JsonFileRegistryStore(persist_path) if persist_path else None)
        if self.store:
            self._load()

    def _load(self) -> None:
        raw = self.store.load_all()
        self._states = {rid: RestaurantState.from_dict(s) for rid, s in raw.items()}

    def _save(self, restaurant_id: str) -> None:
        """Persist ONE restaurant. Every method that mutates a RestaurantState calls
        this with that restaurant's id -- so a metadata-only update (a /predict or
        /production-plan call upserting a product, with no new sales rows) is durable
        too, not just a full /ingest. Before this, only `ingest()` saved, so calling
        /production-plan or /surplus-offers for a restaurant that had never called
        /ingest left its economics unpersisted -- present in memory, gone on restart.
        """
        if not self.store:
            return
        self.store.save_restaurant(restaurant_id, self._states[restaurant_id].to_dict())

    def get(self, restaurant_id: str) -> RestaurantState:
        return self._states.setdefault(restaurant_id, RestaurantState(restaurant_id))

    def upsert_products(self, restaurant_id: str, products: list[ProductInput]) -> None:
        """Register or refresh products for a restaurant, and persist the change.

        The entry point `/predict`, `/production-plan` and `/surplus-offers` should all
        use instead of reaching into `get(...).upsert_products(...)` directly -- that
        form mutates in memory only, silently skipping persistence.
        """
        state = self.get(restaurant_id)
        state.upsert_products(products)
        self._save(restaurant_id)

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
        promotion_active: bool = False, model: "ForecastService | None" = None,
    ) -> dict:
        """Weekly prediction that uses the restaurant's learned level when available.

        `model`, when given, lets the bridge route products with a trained catalogue
        `sku` link through the trained CalendarDecomposed model instead of the level.
        """
        self.upsert_products(restaurant_id, [product])
        state = self.get(restaurant_id)
        level, mode, confidence = state.level_for(product.product_id)
        return predict_week(
            restaurant_id, product, week_start,
            promotion_active=promotion_active, level=level, mode=mode, confidence=confidence,
            model=model,
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
            # Reported, not assumed by the caller: these thresholds belong to this
            # module, and RestoMind's progress bar is derived from them.
            "minDaysForLearned": MIN_DAYS_FOR_LEARNED,
            "quietWindowDays": QUIET_WINDOW,
            "confidentDays": CONFIDENT_DAYS,
            "items": items,
        }
