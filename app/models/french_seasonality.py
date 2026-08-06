"""France counterpart to `m5_seasonality.py`/`favorita_seasonality.py` -- same
Ridge-in-log-space, per-series calendar decomposition, built on `french_calendar.py`'s
real (`holidays` library) public holidays and August-closure signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

RIDGE_ALPHA = 1.0
MIN_MULTIPLIER, MAX_MULTIPLIER = 0.05, 25.0


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for d in range(1, 7):
        X[f"dow_{d}"] = (df["day_of_week"] == d).astype(float)
    X["is_public_holiday"] = df["is_public_holiday"].astype(float)
    X["is_august"] = df["is_august"].astype(float)
    return X


class FrenchCalendarEffects:
    """Per-series multiplicative calendar effects, fitted in log space -- identical
    mechanism to CalendarEffects/M5CalendarEffects/FavoritaCalendarEffects."""

    def __init__(self, alpha: float = RIDGE_ALPHA, min_rows: int = 60) -> None:
        self.alpha = alpha
        self.min_rows = min_rows
        self.models: dict[str, Ridge] = {}
        self.columns: list[str] = []

    def fit(self, df: pd.DataFrame, target: str = "sales_qty") -> "FrenchCalendarEffects":
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
