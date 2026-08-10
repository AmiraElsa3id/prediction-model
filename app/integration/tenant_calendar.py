"""Per-restaurant calendar effects, learned from that restaurant's own sales.

WHY THIS EXISTS
---------------
`registry.py` learns one number per product: the mean of recent quiet days. Quiet means
non-Ramadan, non-holiday, non-kahk and non-weekend, deliberately, so that seasonality is
NOT baked into the level -- the design always assumed something would re-apply it at
predict time. The rule-based layer that used to do that was removed (HANDOFF.md §8) and
nothing replaced it for products outside the trained catalogue.

The result was a production plan frozen at one number: `/integration/restomind/
production-plan` returned the same quantity for a Tuesday, a Friday and a day in Ramadan,
because `target_date` never entered the arithmetic. Meanwhile `/forecast/daily-batch`,
running the trained model, moved with the calendar as it should. Same data, two answers.

WHAT THIS DOES
--------------
Fits the SAME calendar model the trained path uses (`CalendarEffects`: ridge in log
space, calendar features only, one fit per product) to the restaurant's OWN ingested
history, keyed by `productId` instead of catalogue `sku`. Products get their real
weekday shape, and their real Ramadan/holiday shape once they have lived through one.

An effect the history has never contained stays at zero -- ridge shrinks an all-zero
dummy to no effect -- so a restaurant with five months of data gets its weekday profile
and honestly nothing for Ramadan. That is the point: this learns from data or says
nothing, it never falls back to hand-written priors.

NORMALISATION
-------------
`CalendarEffects` measures demand against a rolling clean baseline, whose reference day
is not the same thing as the registry's learned level (a quiet-weekday mean). Multiplying
one by the other directly would shift every quantity by a constant. So each product
stores a `reference`: its mean raw multiplier over the exact days that produced its
learned level. Dividing by it makes the multiplier 1.0 on an average learned-from day, so
`level x multiplier` reproduces today's behaviour on an ordinary weekday and only departs
from it where the calendar says it should.

Persistence: plain floats (coefficients, intercepts, references), so a restaurant's
learned calendar survives a restart in the same JSON/Mongo document as its levels. No
pickled estimators -- that file is read at startup and unpickling is code execution.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.core.features import build_features
from app.models.seasonality import (
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    CalendarEffects,
    align,
    attribute,
    design_matrix,
)


def _calendar_row(day: dt.date) -> pd.DataFrame:
    """A one-row frame carrying `day`'s calendar, shaped for `design_matrix`."""
    return pd.DataFrame([CALENDAR.features(day)])


@dataclass
class TenantCalendar:
    """One restaurant's learned calendar effects, per product.

    `coefficients`/`intercepts` are the ridge fit in log space; `references` is the
    normalisation described in the module docstring. A product absent from the fit gets
    a neutral 1.0 multiplier and no factors -- the caller keeps the flat level.
    """

    columns: list[str]
    coefficients: dict[str, list[float]]
    intercepts: dict[str, float]
    references: dict[str, float]

    def __contains__(self, product_id: str) -> bool:
        return product_id in self.coefficients

    @property
    def products(self) -> list[str]:
        return sorted(self.coefficients)

    def multiplier(self, product_id: str, day: dt.date) -> float:
        """How much this product's demand moves on `day`, relative to its learned level.

        1.0 means "an ordinary day for this product" -- and is what an unlearned product
        always gets, so the caller's arithmetic is unchanged for it.
        """
        coefs = self.coefficients.get(product_id)
        if coefs is None:
            return 1.0
        X = align(design_matrix(_calendar_row(day)), self.columns)
        raw = float(np.exp(self.intercepts[product_id] + float(X.to_numpy()[0] @ np.array(coefs))))
        reference = self.references.get(product_id) or 1.0
        return float(np.clip(raw / reference, MIN_MULTIPLIER, MAX_MULTIPLIER))

    def explain(self, product_id: str, day: dt.date) -> list[dict]:
        """The drivers behind `multiplier`, in the same shape every other endpoint uses."""
        coefs = self.coefficients.get(product_id)
        if coefs is None:
            return []
        return attribute(self.columns, coefs, align(design_matrix(_calendar_row(day)), self.columns))

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "coefficients": {k: [float(c) for c in v] for k, v in self.coefficients.items()},
            "intercepts": {k: float(v) for k, v in self.intercepts.items()},
            "references": {k: float(v) for k, v in self.references.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "TenantCalendar":
        return cls(
            columns=list(raw["columns"]),
            coefficients={k: list(v) for k, v in raw.get("coefficients", {}).items()},
            intercepts=dict(raw.get("intercepts", {})),
            references=dict(raw.get("references", {})),
        )


def fit_tenant_calendar(
    history: pd.DataFrame, quiet_days: dict[str, list],
) -> TenantCalendar | None:
    """Fit calendar effects for the products that earned a learned level.

    `history` is the registry's own frame (`date`, `productId`, `salesQty`, optionally
    `productionQty`/`closingStock`). `quiet_days` maps productId -> the dates whose mean
    IS that product's learned level; those same dates define the normalisation reference,
    which is what keeps `level x multiplier` consistent with the level it multiplies.

    Only products in `quiet_days` are fitted. A product still on the owner's estimate has
    no history to learn a calendar from, and inventing one for it is exactly the
    hand-written-priors behaviour that was removed.

    Returns `None` when nothing could be fitted, so the caller stays on the flat level.
    """
    if not quiet_days or history is None or history.empty:
        return None

    # The pipeline groups by a string `sku`, so key everything by the string form of the
    # id up front. Otherwise a non-string productId (nothing in the API can send one, but
    # `RestaurantRegistry` is callable directly) would silently match nothing and drop
    # every product back to the flat level.
    quiet_days = {str(pid): days for pid, days in quiet_days.items()}

    rows = history[history["productId"].astype(str).isin(quiet_days)]
    if rows.empty:
        return None

    # Rename onto the pipeline's schema so the trained path's cleaning, calendar join and
    # clean_baseline all apply unchanged -- the fit is only as good as that baseline, and
    # reimplementing it here would be a second definition free to drift.
    frame = pd.DataFrame({
        "date": pd.to_datetime(rows["date"]),
        "sku": rows["productId"].astype(str),
        "sales_qty": pd.to_numeric(rows["salesQty"], errors="coerce").astype(float),
        "production_qty": pd.to_numeric(rows.get("productionQty"), errors="coerce").astype(float),
    })
    # `reconcile_stock` rebuilds leftovers from production - sales anyway; an unknown
    # production quantity simply leaves it NaN, which no downstream step here reads.
    frame["leftover_qty"] = (frame["production_qty"] - frame["sales_qty"]).clip(lower=0)

    features = build_features(frame, horizon=1)
    effects = CalendarEffects().fit(features)
    if not effects.models:
        return None

    coefficients: dict[str, list[float]] = {}
    intercepts: dict[str, float] = {}
    references: dict[str, float] = {}

    for pid, model in effects.models.items():
        wanted = pd.to_datetime(pd.Series(quiet_days.get(pid, [])))
        sample = features[(features["sku"] == pid) & features["date"].isin(wanted)]
        if sample.empty:
            continue
        reference = float(np.mean(effects.multiplier(sample)))
        if not np.isfinite(reference) or reference <= 0:
            continue
        coefficients[pid] = [float(c) for c in model.coef_]
        intercepts[pid] = float(model.intercept_)
        references[pid] = reference

    if not coefficients:
        return None

    return TenantCalendar(
        columns=list(effects.columns),
        coefficients=coefficients,
        intercepts=intercepts,
        references=references,
    )
