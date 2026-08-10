"""Explicit calendar effect estimation, separated from the autoregressive model.

WHY THIS MODULE EXISTS
----------------------
Handing calendar features and lag features to one gradient booster does not work, and
fails silently. Measured on this data, a LightGBM model given both learns to predict
Ramadan almost entirely from `lag_1` and `roll_mean_7`: on 364 days out of 365 yesterday
already reflects whatever the calendar is doing, so the lags minimise average loss and
the calendar columns are left with nothing to explain.

The consequence only shows up where it hurts most. On the first day of Ramadan 2025 the
combined model forecast croissants *up* (334 vs 289 the week before) when true demand
falls 55% overnight. It was tracking the event by following sales downward a few days
late. Adding four more years of history did not help -- with six Ramadan onsets it
forecast 390. This is structural, not a sample-size problem.

THE FIX
-------
Decompose, the way Prophet does, and for the same reason -- Prophet has no
autoregressive terms, so seasonality is forced to carry the signal:

    demand  =  baseline level  x  calendar multiplier

The calendar multiplier is fitted here by ridge regression in log space, using calendar
features ONLY. With no lags available, the Ramadan coefficient has to absorb the Ramadan
effect. The autoregressive model then trains on deseasonalised demand, where its job is
just to track the slow-moving level.

The multiplier is applied from the calendar, so it lands in full on day one of Ramadan
rather than being discovered a week later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Ridge, not OLS: rare events (Sham El-Nessim appears twice in two years) would
# otherwise get wild coefficients fitted to a couple of noisy observations. The penalty
# shrinks thinly-observed effects toward "no effect", which is the right prior.
RIDGE_ALPHA = 1.0

# Multipliers outside this range are almost certainly artefacts of a tiny sample.
MIN_MULTIPLIER, MAX_MULTIPLIER = 0.05, 25.0

# Human labels for the design matrix's columns. Several columns share a label on
# purpose (the two kahk ramp terms are one driver to a reader, as are the six weekday
# dummies); `attribute` sums their contributions before reporting.
FACTOR_LABELS = {
    "ramadan_early": "Ramadan (first third)",
    "ramadan_mid": "Ramadan (middle)",
    "ramadan_late": "Ramadan (last third)",
    "eid_fitr": "Eid al-Fitr",
    "eid_adha": "Eid al-Adha",
    "kahk_ramp": "Kahk season",
    "kahk_ramp_sq": "Kahk season",
    "is_sham_el_nessim": "Sham El-Nessim",
    "is_coptic_christmas": "Coptic Christmas",
    "is_public_holiday": "Public holiday",
    "is_school_term": "School term",
    "is_payday_window": "Payday period",
    **{f"dow_{d}": "Day of week" for d in range(1, 7)},
}

# Drivers moving the number by less than this are noise to a manager reading the screen.
MIN_REPORTABLE_EFFECT = 0.02


def attribute(columns: list[str], coefficients, X_row: pd.DataFrame) -> list[dict]:
    """Attribute one row's multiplier to individual calendar drivers.

    Because the fit is linear in log space, each active term's contribution is exactly
    `exp(coefficient x value)` -- a decomposition, not an approximation. Shared by the
    trained per-SKU model (`CalendarEffects.explain`) and the per-restaurant fit in
    `app.integration.tenant_calendar`, so both screens explain a number the same way.

    Effects are relative to the fit's reference day: a Monday outside Ramadan and any
    holiday (see `design_matrix` for why Monday). That is the same convention the trained
    path reports, so a "Day of week" factor always means "compared with a Monday".
    """
    contributions: dict[str, float] = {}
    for col, coef in zip(columns, coefficients):
        value = float(X_row.iloc[0][col])
        if value == 0.0:
            continue
        label = FACTOR_LABELS.get(col, col)
        contributions[label] = contributions.get(label, 0.0) + float(coef) * value

    factors = []
    for label, log_effect in contributions.items():
        change = float(np.exp(log_effect) - 1.0)
        if abs(change) >= MIN_REPORTABLE_EFFECT:
            factors.append({
                "factor": label,
                "impact_pct": round(change * 100, 1),
                "direction": "increase" if change > 0 else "decrease",
            })

    return sorted(factors, key=lambda f: abs(f["impact_pct"]), reverse=True)


def align(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Force a design matrix onto `columns`, filling anything absent with zero.

    Guards against a frame built from a different feature set than the fit saw.
    """
    for col in columns:
        if col not in X.columns:
            X[col] = 0.0
    return X[columns]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-only design matrix. Deliberately contains no history features."""
    X = pd.DataFrame(index=df.index)

    # Weekday as dummies. `day_of_week` is Monday-zero (Friday 4, Saturday 5 -- Egypt's
    # weekend), so omitting index 0 makes MONDAY the reference level, and every reported
    # weekday effect is a deviation from a Monday. An earlier comment here claimed Sunday.
    for d in range(1, 7):
        X[f"dow_{d}"] = (df["day_of_week"] == d).astype(float)

    # Ramadan split into phases -- the month is not homogeneous. The first days are an
    # adjustment shock, the last ten carry both Laylat al-Qadr and pre-Eid shopping.
    ram = df["ramadan_day_index"]
    X["ramadan_early"] = ((ram >= 1) & (ram <= 10)).astype(float)
    X["ramadan_mid"] = ((ram >= 11) & (ram <= 20)).astype(float)
    X["ramadan_late"] = (ram >= 21).astype(float)

    X["eid_fitr"] = (df["eid_fitr_day_index"] > 0).astype(float)
    X["eid_adha"] = (df["eid_adha_day_index"] > 0).astype(float)

    # Kahk ramps convexly into Eid; the raw ramp and its square let the fit bend.
    X["kahk_ramp"] = df["kahk_ramp"].astype(float)
    X["kahk_ramp_sq"] = df["kahk_ramp"].astype(float) ** 2

    for col in (
        "is_sham_el_nessim", "is_coptic_christmas", "is_public_holiday",
        "is_school_term", "is_payday_window",
    ):
        if col in df.columns:
            X[col] = df[col].astype(float)

    return X


class CalendarEffects:
    """Per-item multiplicative calendar effects, fitted in log space.

    One ridge model per SKU. Items differ qualitatively -- Ramadan halves croissants and
    quadruples konafa -- so pooling them would average the effects to nothing.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA, min_rows: int = 60) -> None:
        self.alpha = alpha
        self.min_rows = min_rows
        self.models: dict[str, Ridge] = {}
        self.columns: list[str] = []

    def fit(self, df: pd.DataFrame, target: str = "sales_qty") -> "CalendarEffects":
        """Fit log(demand / baseline) ~ calendar, per SKU."""
        usable = df[
            df[target].notna()
            & (df["is_closed"] == 0)
            & (df["is_outlier"] == 0)
            & df["clean_baseline"].notna()
            & (df["clean_baseline"] > 0)
            # NOTE: stockout days are deliberately KEPT here. They are censored, so
            # they understate demand -- but excluding them is worse. Items with the
            # largest jumps (konafa quadruples overnight) sell out on almost every
            # event day, so dropping stockouts removes the entire effect: konafa's
            # true 4.5x was estimated at 2.0x. Keeping the capped values yields a
            # conservative multiplier rather than almost none.
        ]

        X_all = design_matrix(usable)
        self.columns = list(X_all.columns)

        for sku, grp in usable.groupby("sku", sort=False):
            if len(grp) < self.min_rows:
                continue
            X = X_all.loc[grp.index]
            # +1 keeps zero-demand days finite without distorting busy ones.
            y = np.log((grp[target] + 1.0) / (grp["clean_baseline"] + 1.0))
            model = Ridge(alpha=self.alpha)
            model.fit(X, y)
            self.models[sku] = model

        return self

    def multiplier(self, df: pd.DataFrame) -> np.ndarray:
        """Calendar multiplier for each row. Unknown SKUs get a neutral 1.0."""
        X = align(design_matrix(df), self.columns)

        out = np.ones(len(df), dtype=float)
        skus = df["sku"].to_numpy()
        for sku, model in self.models.items():
            sel = skus == sku
            if sel.any():
                out[sel] = np.exp(model.predict(X[sel]))

        return np.clip(out, MIN_MULTIPLIER, MAX_MULTIPLIER)

    def explain(self, df_row: pd.DataFrame) -> list[dict]:
        """Attribute a single date's multiplier to individual calendar drivers.

        This is the attribution the combined model could not produce. The arithmetic
        lives in `attribute` so the per-restaurant fit explains itself identically.
        """
        model = self.models.get(df_row.iloc[0]["sku"])
        if model is None:
            return []
        return attribute(self.columns, model.coef_, align(design_matrix(df_row), self.columns))
