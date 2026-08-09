"""Forecast service: the layer the API talks to.

Wraps model training, caching, prediction intervals, and -- importantly for adoption --
explanations. A bare integer ("make 137 croissants") gets overridden by the manager the
first time it looks odd. "137, down 35% because Ramadan starts tomorrow" gets followed.

Holds state in memory and on disk via pickle. That is a deliberate POC choice: a real
deployment needs a model registry and a per-tenant store, which is scaffolding that
would not change anything an investor sees.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.core.features import build_features
from app.core.generate import generate
from app.core.items import Item, Catalogue
from app.models.forecaster import CalendarDecomposed

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
LOW_Q, HIGH_Q = 0.1, 0.9

# Days of history an item needs before its trained model is trusted. Below this there
# is NO forecast: the rule-based cold-start layer was removed (see HANDOFF.md), so an
# item under the threshold has no forecaster at all until it accumulates the data.
TRAIN_THRESHOLD_DAYS = 90


class ModelNotReadyError(RuntimeError):
    """Raised when a forecast is requested for an item with no trained model yet.

    This is the post-rule cold-start contract: below ``TRAIN_THRESHOLD_DAYS`` the model
    admits it has nothing to say rather than inventing a number from hand-written
    priors. The API maps this to a "still training" response; consumers fall back to
    their own estimator until the item's own model exists.
    """


@dataclass
class ForecastResult:
    sku: str
    date: dt.date
    quantity: int
    lower: int
    upper: int
    confidence: str
    source: str
    factors: list[dict] = field(default_factory=list)


class ForecastService:
    """Trains and serves the demand forecaster."""

    def __init__(self, horizon: int = 1, train_threshold: int = TRAIN_THRESHOLD_DAYS) -> None:
        self.horizon = horizon
        self.train_threshold = train_threshold
        # The DYNAMIC item registry: populated only from uploaded data (see `Catalogue`).
        # Empty catalogue => nothing is known, nothing forecastable -- that is the honest
        # answer for a brand-new bakery with no uploaded items.
        self.catalogue = Catalogue()
        # ML models start empty. Without the rule-based layer there is no afternoon
        # fallback: an item only has a forecast once its model exists and is over the
        # training threshold.
        self.point_model: CalendarDecomposed | None = None
        self.low_model: CalendarDecomposed | None = None
        self.high_model: CalendarDecomposed | None = None
        self.history: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.observed_days: dict[str, int] = {}
        self.observed_events: set[str] = set()
        self.trained_at: dt.datetime | None = None
        # Provenance for /health: SIMULATED base comes from generate(); real rows are
        # appended by ingest(). Tracks both halves so health can say which is true.
        self.base_data_source: str = "none"   # "simulated" | "real" | "none"
        self.real_ingest_rows: int = 0

    # -- data & training ----------------------------------------------------------

    # Major calendar events the trained model may still be weak on until it has seen
    # them in its training window. When a date carries one it has never observed, the
    # forecast is still made from the model, but the interval widens and confidence
    # drops -- there is no rule layer left to route to instead.
    MAJOR_EVENTS = [
        "is_ramadan", "is_eid_fitr", "is_eid_adha", "is_kahk_window", "is_sham_el_nessim",
    ]

    def _recount_days(self) -> None:
        """Per-item day counts and the set of major events seen in the training window."""
        if self.history is None or self.history.empty:
            self.observed_days = {}
            self.observed_events = set()
            return
        h = self.history.copy()
        h["date"] = pd.to_datetime(h["date"])
        self.observed_days = h.groupby("sku")["date"].nunique().to_dict()

        # Which major events does the accumulated history actually contain?
        cal = CALENDAR.feature_frame(h["date"].min(), h["date"].max())
        self.observed_events = {e for e in self.MAJOR_EVENTS if cal[e].sum() > 0}

    def _target_unseen_event(self, target_date: dt.date) -> bool:
        """True if the date carries a major event the model has never trained on."""
        cal = CALENDAR.features(target_date)
        active = {e for e in self.MAJOR_EVENTS if cal.get(e)}
        return bool(active - getattr(self, "observed_events", set()))

    def _fit_ml(self) -> None:
        """(Re)fit the trained model on all accumulated history.

        Called when at least one item has crossed the threshold. Items still under it
        keep having no forecast -- routing is per item, in :meth:`_use_ml`, and the
        rule-based fallback no longer exists.
        """
        self.features = build_features(self.history, horizon=self.horizon)
        self.point_model = CalendarDecomposed(horizon=self.horizon).fit(self.features)
        self.low_model = CalendarDecomposed(quantile=LOW_Q, horizon=self.horizon).fit(self.features)
        self.high_model = CalendarDecomposed(quantile=HIGH_Q, horizon=self.horizon).fit(self.features)
        self.trained_at = dt.datetime.now()

    def train(self, raw: pd.DataFrame | None = None) -> "ForecastService":
        """Load history and fit the trained model where there is enough of it.

        With the default full synthetic dataset every item has years of history, so all
        of them route to the trained `CalendarDecomposed` model. With a small slice (or
        nothing), items below the threshold have NO forecast -- this is the cold-start
        path a real new bakery follows until its data arrives.
        """
        # If no raw data was passed we built the history from the simulator; that is
        # the honesty signal /health reports. Real data passed in is marked as such.
        _generated = raw is None
        raw = generate() if raw is None else raw
        self.history = raw.copy()
        self.base_data_source = "simulated" if _generated else "real"
        # Items are whatever the data contains; the runtime catalogue never uses the
        # hardcoded simulation fixtures.
        self.catalogue.register_from_data(self.history)
        self._recount_days()

        # Fit the ML model only if some item has enough history to justify it.
        if self.observed_days and max(self.observed_days.values()) >= self.train_threshold:
            self._fit_ml()
        return self

    def start_cold(self) -> "ForecastService":
        """Begin with zero history: no model, no items, so nothing is forecastable yet."""
        self.history = None
        self.features = None
        self.observed_days = {}
        # Critical: this is what the whole reset is for. If we only cleared the models
        # and observed_days, a previously-trained instance would still carry its old
        # observed_events, so the unseen-event safety net (_confidence_for_date) would
        # wrongly trust the model on events it has never actually seen this run.
        self.observed_events = set()
        self.point_model = self.low_model = self.high_model = None
        self.trained_at = None
        self.base_data_source = "none"
        self.real_ingest_rows = 0
        # The dynamic catalogue resets with the history: nothing uploaded => nothing known.
        self.catalogue = Catalogue()
        return self

    def register_item(
        self,
        sku: str,
        *,
        unit_price: float = 0.0,
        unit_cost: float = 0.0,
        shelf_life_days: int = 1,
        name_ar: str = "",
        name_en: str = "",
        category: str = "general",
    ) -> None:
        """Register/refresh one item in the dynamic catalogue without any sales data.

        This is how a backend syncs its product menu to the model before or alongside
        posting history. Registering alone does NOT train anything -- the item still
        needs cross-threshold history before a forecast exists.
        """
        self.catalogue.register(
            sku, unit_price=unit_price, unit_cost=unit_cost,
            shelf_life_days=shelf_life_days, name_ar=name_ar,
            name_en=name_en, category=category,
        )

    def ingest(self, records: pd.DataFrame) -> dict:
        """Append end-of-day actuals and retrain if an item crosses the threshold.

        This is what the API calls each night. It accumulates real sales, refreshes the
        trained model whenever an item reaches `train_threshold` days.
        """
        records = records.copy()
        records["date"] = pd.to_datetime(records["date"])
        # New SKUs (with their economics) become known the moment their data arrives.
        self.catalogue.register_from_data(records)
        self.history = records if self.history is None else pd.concat(
            [self.history, records], ignore_index=True
        ).drop_duplicates(subset=["date", "sku"], keep="last")

        # This method is the real-data append endpoint. Record provenance so /health
        # stops claiming SIMULATED once actuals have been posted on top of (or instead
        # of) the generated base.
        self.real_ingest_rows += len(records)

        before = {s for s, n in self.observed_days.items() if n >= self.train_threshold}
        self._recount_days()
        after = {s for s, n in self.observed_days.items() if n >= self.train_threshold}

        newly_ready = sorted(after - before)
        # Retrain when the eligible set grows, or when we already have a model to refresh.
        if after and (newly_ready or self.point_model is not None):
            self._fit_ml()

        return {
            "rows_ingested": len(records),
            "total_days_by_item": {k: int(v) for k, v in self.observed_days.items()},
            "newly_switched_to_ml": newly_ready,
            "model_retrained": bool(after and (newly_ready or self.point_model is not None)),
        }

    def _use_ml(self, sku: str) -> bool:
        """Whether this item has a trained model it can serve a forecast for.

        The gate that used to be "rule-based below, trained above" is now simply "not
        below": no model exists and no rules exist, so anything under the threshold is
        simply not forecastable. An unseen-event date no longer routes away from the
        model — see `_confidence_for_date` for how that case is handled instead.
        """
        if self.point_model is None or self.observed_days.get(sku, 0) < self.train_threshold:
            return False
        return True

    def data_source(self) -> str:
        """Honest label for /health: which data actually feeds the model.

        The service is truthful rather than aspirational here. A purely
        simulated base reports SIMULATED; once real actuals are appended on top
        it reports SIMULATED + REAL; a cold start fed only real actuals reports
        REAL. The old always-SIMULATED label made /data/ingest a lie the moment
        real data arrived. getattr guards survive pickles written before the
        provenance fields existed.
        """
        real = getattr(self, "real_ingest_rows", 0)
        base = getattr(self, "base_data_source", "none")
        if real > 0 and base == "simulated":
            return "SIMULATED + REAL"
        if real > 0:
            return "REAL"
        if base == "simulated":
            return "SIMULATED"
        return "NONE"

    def status(self) -> dict:
        """Per-item mode report: which items have a trained model, and how close to one.

        Lets the frontend show a progress bar per item ("42 / 90 days"): before the
        threshold an item is not forecastable at all (no rule layer remains); at or
        above it the trained model serves it.
        """
        items = []
        for sku in self.catalogue.skus():
            days = int(self.observed_days.get(sku, 0))
            ml = self._use_ml(sku)
            items.append({
                "sku": sku,
                "mode": "trained_model" if ml else "untrained",
                "observed_days": days,
                "days_until_switch": max(0, self.train_threshold - days) if not ml else 0,
                "progress": round(min(1.0, days / self.train_threshold), 3),
            })
        return {
            "train_threshold_days": self.train_threshold,
            "model_trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "items_untrained": sum(1 for i in items if i["mode"] == "untrained"),
            "items_trained": sum(1 for i in items if i["mode"] == "trained_model"),
            "items": items,
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_DIR / "forecaster.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ForecastService":
        path = path or MODEL_DIR / "forecaster.pkl"
        with path.open("rb") as fh:
            return pickle.load(fh)

    # -- prediction ---------------------------------------------------------------

    def _row_for(self, sku: str, target_date: dt.date) -> pd.DataFrame:
        """Assemble a feature row for a future date.

        Lag features come from the tail of observed history. Because everything is
        shifted by at least the horizon, a date beyond the data still has all the lags
        it needs -- which is exactly what the horizon shift was for.
        """
        if self.features is None:
            raise RuntimeError("service not trained")

        hist = self.features[self.features["sku"] == sku].sort_values("date")
        if hist.empty:
            raise KeyError(f"no history for SKU {sku!r}")

        # Prefer the real feature row for this date. Reusing the last row of history and
        # only swapping the calendar columns would leave the lag features describing a
        # completely different time of year, which silently swamps the calendar signal.
        target_ts = pd.Timestamp(target_date)
        exact = hist[hist["date"] == target_ts]
        row = (exact.iloc[[-1]] if not exact.empty else hist.iloc[[-1]]).copy()
        row["date"] = target_ts

        # Overwrite calendar columns with the target date's real calendar.
        cal = CALENDAR.features(target_date)
        for key, value in cal.items():
            if key in row.columns and key != "date":
                row[key] = value
        return row

    def _explain(self, sku: str, row: pd.DataFrame) -> list[dict]:
        """Attribute the forecast to calendar drivers.

        Delegates to the decomposed model's ridge attribution. Because that fit is
        linear in log space and uses no lag features, each active calendar term's
        contribution is an exact `exp(coefficient)` -- not a counterfactual estimate.
        This is what the single-model approach could never produce: with lag features
        present, switching the Ramadan flag off barely moved the prediction because the
        lags already carried the effect.
        """
        if self.point_model is None:
            return []
        return self.point_model.explain(row)

    def _confidence(self, sku: str) -> str:
        """How much history backs this forecast, as a coarse label."""
        n = self.observed_days.get(sku, 0)
        if n >= 365:
            return "high"
        if n >= self.train_threshold:
            return "medium"
        return "low"

    def _confidence_for_date(self, sku: str, target_date: dt.date) -> tuple[str, float]:
        """(confidence, interval_scale) for an item on a date.

        An item that is otherwise trained but has never seen this day's major event
        (first Ramadan, first kahk season) cannot be trusted as usual: the model's
        learned effect for it is ~zero. With no rule layer to fall back to, we still
        forecast, but mark it `low` confidence and widen the interval.
        """
        if self._target_unseen_event(target_date):
            return "low", 1.35
        return self._confidence(sku), 1.0

    def forecast(self, sku: str, target_date: dt.date) -> ForecastResult:
        if sku not in self.catalogue:
            raise KeyError(f"unknown SKU {sku!r}")

        # The gate: no trained model, no rules, no forecast. The caller shows
        # "model still training" and falls back to whatever it controls.
        if not self._use_ml(sku):
            raise ModelNotReadyError(
                f"no trained model for {sku!r} yet "
                f"({self.observed_days.get(sku, 0)} of {self.train_threshold} days)"
            )

        confidence, scale = self._confidence_for_date(sku, target_date)
        row = self._row_for(sku, target_date)
        qty = float(self.point_model.predict(row)[0])
        low = float(self.low_model.predict(row)[0])
        high = float(self.high_model.predict(row)[0])

        return ForecastResult(
            sku=sku,
            date=target_date,
            quantity=int(round(qty)),
            lower=int(round(min(low * scale, qty))),
            upper=int(round(max(high * scale, qty))),
            confidence=confidence,
            source="batch",
            factors=self._explain(sku, row),
        )

    def forecast_week(self, sku: str, start: dt.date) -> list[ForecastResult]:
        return [self.forecast(sku, start + dt.timedelta(days=i)) for i in range(7)]

    def forecast_all(
        self, target_date: dt.date, skus: list[str] | None = None,
    ) -> list[ForecastResult]:
        """Forecast every item for one day in a single model pass.

        This is what the POS/e-commerce backend should call each morning: one request
        for the whole production plan instead of one round-trip per SKU. Any item that
        has not reached the training threshold has NO forecast (the rule layer is gone)
        and raises `ModelNotReadyError` -- the caller should surface it as "still
        training" rather than plan from nothing.
        """
        skus = skus or self.catalogue.skus()
        unknown = [s for s in skus if s not in self.catalogue]
        if unknown:
            raise KeyError(f"unknown SKU(s): {', '.join(unknown)}")

        not_ready = [s for s in skus if not self._use_ml(s)]
        if not_ready:
            first = not_ready[0]
            raise ModelNotReadyError(
                f"no trained model for {first!r} yet "
                f"({self.observed_days.get(first, 0)} of {self.train_threshold} days)"
            )

        results: dict[str, ForecastResult] = {}

        rows = pd.concat([self._row_for(s, target_date) for s in skus], ignore_index=True)
        qty = self.point_model.predict(rows)
        low = self.low_model.predict(rows)
        high = self.high_model.predict(rows)
        for i, sku in enumerate(skus):
            confidence, scale = self._confidence_for_date(sku, target_date)
            q = float(qty[i])
            results[sku] = ForecastResult(
                sku=sku, date=target_date, quantity=int(round(q)),
                lower=int(round(min(float(low[i]) * scale, q))),
                upper=int(round(max(float(high[i]) * scale, q))),
                confidence=confidence, source="batch",
                factors=self._explain(sku, rows.iloc[[i]]),
            )

        return [results[s] for s in skus]

    def forecast_week_all(
        self, start: dt.date, skus: list[str] | None = None,
    ) -> dict[str, list[ForecastResult]]:
        """Seven-day plan for every item, keyed by SKU. One model pass per day."""
        skus = skus or self.catalogue.skus()
        by_sku: dict[str, list[ForecastResult]] = {s: [] for s in skus}
        for i in range(7):
            for r in self.forecast_all(start + dt.timedelta(days=i), skus):
                by_sku[r.sku].append(r)
        return by_sku

    # -- seasonality ---------------------------------------------------------------

    def seasonality_adjustment(self, sku: str, target_date: dt.date) -> dict:
        """Calendar-only view: how much does this date differ from a neutral one?

        The multiplier comes from the decomposed model's learned calendar component.
        An item with no trained model raises `ModelNotReadyError` -- there is no
        rule-based priors layer left to answer for it.
        """
        cal = CALENDAR.features(target_date)
        if not self._use_ml(sku):
            raise ModelNotReadyError(
                f"no trained model for {sku!r} yet "
                f"({self.observed_days.get(sku, 0)} of {self.train_threshold} days)"
            )

        row = self._row_for(sku, target_date)
        qty = float(self.point_model.predict(row)[0])
        multiplier = float(self.point_model.calendar_multiplier(row)[0])
        factors = self._explain(sku, row)

        base = qty / multiplier if multiplier > 0 else qty

        return {
            "sku": sku,
            "date": target_date,
            "baseline_quantity": int(round(base)),
            "adjusted_quantity": int(round(qty)),
            "multiplier": round(multiplier, 3),
            "factors": factors,
            "calendar": {
                "is_ramadan": bool(cal["is_ramadan"]),
                "ramadan_day": int(cal["ramadan_day_index"]) or None,
                "holiday": cal["holiday_name"],
                "is_weekend": bool(cal["is_weekend"]),
                "is_school_term": bool(cal["is_school_term"]),
                "days_to_eid_fitr": (
                    int(cal["days_to_eid_fitr"]) if cal["days_to_eid_fitr"] < 999 else None
                ),
            },
        }

    # -- waste prevention ----------------------------------------------------------

    def waste_alert(self, sku: str, target_date: dt.date, planned_qty: int) -> dict:
        """Compare a manager's manual entry against the forecast interval.

        Alerting on any deviation would train managers to ignore the system, so the
        threshold is the upper bound of the prediction interval, not the point
        forecast: we only object when the plan is outside what the model considers
        plausible at all.
        """
        item: Item = self.catalogue.get(sku)
        fc = self.forecast(sku, target_date)

        excess = planned_qty - fc.upper
        if excess <= 0:
            severity, message = "none", "Planned quantity is within the forecast range."
        else:
            ratio = excess / max(fc.upper, 1)
            severity = "high" if ratio > 0.5 else "medium" if ratio > 0.2 else "low"
            message = (
                f"Planned {planned_qty} exceeds the forecast upper bound of {fc.upper} "
                f"by {excess} units."
            )

        projected_waste = max(excess, 0)
        return {
            "sku": sku,
            "date": target_date,
            "planned_qty": planned_qty,
            "forecast_qty": fc.quantity,
            "forecast_upper": fc.upper,
            "excess_qty": max(excess, 0),
            "severity": severity,
            "message": message,
            "projected_waste_cost_egp": round(
                projected_waste * item.unit_cost * item.spoilage_severity, 2
            ),
            "factors": fc.factors,
        }
