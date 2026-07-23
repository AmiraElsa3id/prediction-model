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
from app.core.items import BY_SKU, Item
from app.models.forecaster import CalendarDecomposed
from app.models.rule_based import RuleBasedForecaster

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
LOW_Q, HIGH_Q = 0.1, 0.9

# Days of history an item needs before we trust the trained model over the rules.
# Below this the forecast is rule-based (owner priors + calendar rules); at or above it,
# the item switches to the trained CalendarDecomposed model. Confirmed with the user.
TRAIN_THRESHOLD_DAYS = 90


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
        # ML models start empty: a fresh bakery is entirely rule-based until data arrives.
        self.point_model: CalendarDecomposed | None = None
        self.low_model: CalendarDecomposed | None = None
        self.high_model: CalendarDecomposed | None = None
        self.rule_model = RuleBasedForecaster()
        self.history: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.observed_days: dict[str, int] = {}
        self.observed_events: set[str] = set()
        self.trained_at: dt.datetime | None = None

    # -- data & training ----------------------------------------------------------

    # Major calendar events. The trained model may only be trusted for one of these once
    # it has actually appeared in the training window -- otherwise its learned effect is
    # ~zero and it would forecast a Ramadan day as if it were ordinary.
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

        Only called when at least one item has crossed the threshold. Items still under
        it keep using the rules even after this runs -- routing is per item, in
        :meth:`_use_ml`.
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
        nothing), items below the threshold stay rule-based -- this is the cold-start
        path a real new bakery follows.
        """
        raw = generate() if raw is None else raw
        self.history = raw.copy()
        self._recount_days()
        self.rule_model.update_baseline(self.history)

        # Fit the ML model only if some item has enough history to justify it.
        if self.observed_days and max(self.observed_days.values()) >= self.train_threshold:
            self._fit_ml()
        return self

    def start_cold(self) -> "ForecastService":
        """Begin with zero history: every item is rule-based until data is ingested."""
        self.history = None
        self.features = None
        self.observed_days = {}
        self.point_model = self.low_model = self.high_model = None
        self.trained_at = None
        return self

    def ingest(self, records: pd.DataFrame) -> dict:
        """Append end-of-day actuals and retrain if an item crosses the threshold.

        This is what the POS/e-commerce backend calls each night. It accumulates real
        sales, refreshes the rule-based baselines immediately, and promotes items to the
        trained model once they reach `train_threshold` days.
        """
        records = records.copy()
        records["date"] = pd.to_datetime(records["date"])
        self.history = records if self.history is None else pd.concat(
            [self.history, records], ignore_index=True
        ).drop_duplicates(subset=["date", "sku"], keep="last")

        before = {s for s, n in self.observed_days.items() if n >= self.train_threshold}
        self._recount_days()
        self.rule_model.update_baseline(self.history)
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

    def _use_ml(self, sku: str, target_date: dt.date | None = None) -> bool:
        """Route to the trained model for this item (and, if given, this date).

        Two conditions, both required:
          * the item has at least `train_threshold` days of history, and
          * if a date is given, it carries no major calendar event the model has never
            trained on -- otherwise the rules, which know that event, are safer.
        """
        if self.point_model is None or self.observed_days.get(sku, 0) < self.train_threshold:
            return False
        if target_date is not None and self._target_unseen_event(target_date):
            return False
        return True

    def status(self) -> dict:
        """Per-item mode report: which items are rule-based vs trained, and how close.

        Lets the frontend show a progress bar per item ("42 / 90 days until the AI takes
        over") -- the cold-start story made visible.
        """
        items = []
        for sku in BY_SKU:
            days = int(self.observed_days.get(sku, 0))
            ml = self._use_ml(sku)
            items.append({
                "sku": sku,
                "mode": "trained_model" if ml else "rule_based",
                "observed_days": days,
                "days_until_switch": max(0, self.train_threshold - days) if not ml else 0,
                "progress": round(min(1.0, days / self.train_threshold), 3),
            })
        return {
            "train_threshold_days": self.train_threshold,
            "model_trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "items_rule_based": sum(1 for i in items if i["mode"] == "rule_based"),
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

    def _rule_forecast(self, sku: str, target_date: dt.date) -> ForecastResult:
        """Cold-start path: owner priors + calendar rules, no trained model involved."""
        out = self.rule_model.forecast(sku, target_date)
        return ForecastResult(
            sku=sku,
            date=target_date,
            quantity=out["quantity"],
            lower=out["lower"],
            upper=out["upper"],
            confidence="low",
            source="rule_based",
            factors=out["factors"],
        )

    def forecast(self, sku: str, target_date: dt.date) -> ForecastResult:
        if sku not in BY_SKU:
            raise KeyError(f"unknown SKU {sku!r}")

        # Route per item and date: rule-based until it has enough history, and still
        # rule-based on a major event the model has not yet trained through.
        if not self._use_ml(sku, target_date):
            return self._rule_forecast(sku, target_date)

        row = self._row_for(sku, target_date)
        qty = float(self.point_model.predict(row)[0])
        low = float(self.low_model.predict(row)[0])
        high = float(self.high_model.predict(row)[0])

        return ForecastResult(
            sku=sku,
            date=target_date,
            quantity=int(round(qty)),
            lower=int(round(min(low, qty))),
            upper=int(round(max(high, qty))),
            confidence=self._confidence(sku),
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
        for the whole production plan instead of one round-trip per SKU. Items that have
        reached the training threshold run through the model in a single pass (rebuilding
        the deseasonalised history only once); items still in cold start are answered
        from the rules. Results keep the requested order.
        """
        skus = skus or list(BY_SKU)
        unknown = [s for s in skus if s not in BY_SKU]
        if unknown:
            raise KeyError(f"unknown SKU(s): {', '.join(unknown)}")

        ml_skus = [s for s in skus if self._use_ml(s, target_date)]
        results: dict[str, ForecastResult] = {}

        # Trained items: one batched model pass over all of them together.
        if ml_skus:
            rows = pd.concat([self._row_for(s, target_date) for s in ml_skus], ignore_index=True)
            qty = self.point_model.predict(rows)
            low = self.low_model.predict(rows)
            high = self.high_model.predict(rows)
            for i, sku in enumerate(ml_skus):
                q = float(qty[i])
                results[sku] = ForecastResult(
                    sku=sku, date=target_date, quantity=int(round(q)),
                    lower=int(round(min(float(low[i]), q))),
                    upper=int(round(max(float(high[i]), q))),
                    confidence=self._confidence(sku), source="batch",
                    factors=self._explain(sku, rows.iloc[[i]]),
                )

        # Cold-start items: answered from the rules.
        for sku in skus:
            if sku not in results:
                results[sku] = self._rule_forecast(sku, target_date)

        return [results[s] for s in skus]

    def forecast_week_all(
        self, start: dt.date, skus: list[str] | None = None,
    ) -> dict[str, list[ForecastResult]]:
        """Seven-day plan for every item, keyed by SKU. One model pass per day."""
        skus = skus or list(BY_SKU)
        by_sku: dict[str, list[ForecastResult]] = {s: [] for s in skus}
        for i in range(7):
            for r in self.forecast_all(start + dt.timedelta(days=i), skus):
                by_sku[r.sku].append(r)
        return by_sku

    # -- seasonality ---------------------------------------------------------------

    def seasonality_adjustment(self, sku: str, target_date: dt.date) -> dict:
        """Calendar-only view: how much does this date differ from a neutral one?

        For trained items the multiplier comes straight from the decomposed model's
        calendar component; for cold-start items it comes from the rule-based priors. In
        both cases it is exactly the factor the forecast applies.
        """
        cal = CALENDAR.features(target_date)

        if self._use_ml(sku, target_date):
            row = self._row_for(sku, target_date)
            qty = float(self.point_model.predict(row)[0])
            multiplier = float(self.point_model.calendar_multiplier(row)[0])
            factors = self._explain(sku, row)
        else:
            from app.models.rule_based import rule_multiplier
            multiplier, factors = rule_multiplier(self.rule_model.priors[sku], cal)
            qty = float(self._rule_forecast(sku, target_date).quantity)

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
        item: Item = BY_SKU[sku]
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
