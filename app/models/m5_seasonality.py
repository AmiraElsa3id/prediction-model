"""M5 counterpart to `app/models/seasonality.py` -- same Ridge-in-log-space,
per-series calendar decomposition, built on `m5_calendar.py`'s real event columns
instead of Egypt's.

The reasoning for Ridge over OLS carries over unchanged: M5's real events are also
sparse per series (SuperBowl, Thanksgiving et al. appear once a year), so an
unregularised fit would still produce wild coefficients on a couple of noisy
observations. Same fix, same `alpha=1.0` default, same bounded output range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

RIDGE_ALPHA = 1.0
MIN_MULTIPLIER, MAX_MULTIPLIER = 0.05, 25.0


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """M5 calendar-only design matrix. No history features, same discipline as
    egypt_calendar's design_matrix -- the calendar coefficient must not have lags
    available to hide behind."""
    X = pd.DataFrame(index=df.index)

    for d in range(1, 7):
        X[f"dow_{d}"] = (df["day_of_week"] == d).astype(float)

    for col in ("is_sporting", "is_cultural", "is_national", "is_religious"):
        X[col] = df[col].astype(float)

    # SNAP windows shift which days benefit purchases land on -- relevant for FOODS.
    X["snap_ca"] = df["snap_ca"].astype(float)

    return X


class M5CalendarEffects:
    """Per-series multiplicative calendar effects, fitted in log space -- identical
    mechanism to `CalendarEffects`, different (real, recorded) input columns."""

    def __init__(self, alpha: float = RIDGE_ALPHA, min_rows: int = 60) -> None:
        self.alpha = alpha
        self.min_rows = min_rows
        self.models: dict[str, Ridge] = {}
        self.columns: list[str] = []

    def fit(self, df: pd.DataFrame, target: str = "sales_qty") -> "M5CalendarEffects":
        usable = df[
            df[target].notna()
            & df["clean_baseline"].notna()
            & (df["clean_baseline"] > 0)
        ]

        X_all = design_matrix(usable)
        self.columns = list(X_all.columns)

        for sku, grp in usable.groupby("sku", sort=False):
            if len(grp) < self.min_rows:
                continue
            X = X_all.loc[grp.index]
            y = np.log((grp[target] + 1.0) / (grp["clean_baseline"] + 1.0))
            model = Ridge(alpha=self.alpha)
            model.fit(X, y)
            self.models[sku] = model

        return self

    def multiplier(self, df: pd.DataFrame) -> np.ndarray:
        X = design_matrix(df)
        for col in self.columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.columns]

        out = np.ones(len(df), dtype=float)
        skus = df["sku"].to_numpy()
        for sku, model in self.models.items():
            sel = skus == sku
            if sel.any():
                out[sel] = np.exp(model.predict(X[sel]))

        return np.clip(out, MIN_MULTIPLIER, MAX_MULTIPLIER)
