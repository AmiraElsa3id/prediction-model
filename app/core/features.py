"""Cleaning and feature engineering for the demand forecaster.

Two things here are easy to get wrong and expensive when wrong:

LEAKAGE
    Every lag and rolling statistic is shifted by at least the forecast horizon. When
    predicting 7 days out we must not use a 1-day lag, because in production that value
    does not exist yet. `build_features(horizon=h)` enforces this by construction rather
    than by discipline.

CLOSED vs ZERO
    A day the bakery was shut and a day nothing sold look identical after a naive
    `reindex().fillna(0)`. They are opposites: one is missing data, the other is a real
    observation of zero demand. Conflating them drags the weekday profile down and
    teaches the model that Eid is a dead day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.core.items import newsvendor_q_from_row

# Lags and rolling windows offered to the model, subject to the horizon constraint.
LAG_DAYS = [1, 2, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]

# Rows this far outside the rolling median (in MADs) are treated as suspect --
# unless a calendar event explains them.
OUTLIER_MAD_THRESHOLD = 6.0

CALENDAR_FEATURES = [
    "day_of_week", "is_weekend", "day_of_month", "month", "week_of_year",
    "is_ramadan", "ramadan_day_index", "ramadan_progress", "is_last_ten_of_ramadan",
    "days_to_eid_fitr", "is_kahk_window", "kahk_ramp",
    "is_eid_fitr", "eid_fitr_day_index", "is_eid_adha", "eid_adha_day_index",
    "is_sham_el_nessim", "is_coptic_christmas", "is_public_holiday",
    "is_school_term", "is_payday_window",
]


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------


def reconcile_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where production - sales != leftover.

    An inventory miscount means at least one of the three numbers is wrong and we
    cannot tell which. Sales come from the POS and are the most trustworthy, so we
    keep sales and flag the row rather than silently "fixing" leftovers.
    """
    df = df.copy()
    implied = df["production_qty"] - df["sales_qty"]
    df["stock_mismatch"] = (implied != df["leftover_qty"]).astype(int)
    # Trust the POS: rebuild leftovers from the identity, keeping the flag for audit.
    df["leftover_qty_raw"] = df["leftover_qty"]
    df["leftover_qty"] = implied.clip(lower=0)
    return df


def mark_closed_days(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a full daily grid per SKU, distinguishing closed days from zero sales.

    Missing rows become `is_closed = 1` with NaN targets, so they are excluded from
    training instead of being read as genuine zero demand.
    """
    frames = []
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    for sku, grp in df.groupby("sku", sort=False):
        g = grp.set_index("date").reindex(full_range)
        g.index.name = "date"
        g["sku"] = sku
        g["is_closed"] = g["sales_qty"].isna().astype(int)
        # Static per-item attributes survive the reindex.
        for col in ("branch", "item_name_ar", "category", "unit_price", "unit_cost",
                    "shelf_life_days"):
            if col in g.columns:
                g[col] = g[col].ffill().bfill()
        frames.append(g.reset_index())

    return pd.concat(frames, ignore_index=True).sort_values(["sku", "date"])


def flag_outliers(df: pd.DataFrame, target: str = "sales_qty") -> pd.DataFrame:
    """Hampel-style outlier flag that spares calendar-explained spikes.

    A 40x jump in kahk sales the day before Eid is the single most important signal in
    the dataset. A naive outlier filter would erase exactly the peaks we are paid to
    predict, so any day carrying a calendar event is exempt from flagging.
    """
    df = df.copy()
    df["is_outlier"] = 0

    explained = (
        (df.get("is_public_holiday", 0) == 1)
        | (df.get("is_ramadan", 0) == 1)
        | (df.get("is_kahk_window", 0) == 1)
        | (df.get("is_sham_el_nessim", 0) == 1)
    )

    for sku, grp in df.groupby("sku", sort=False):
        s = grp[target]
        med = s.rolling(28, center=True, min_periods=7).median()
        mad = (s - med).abs().rolling(28, center=True, min_periods=7).median()
        # 1.4826 scales MAD to a standard-deviation equivalent for normal data.
        scaled = 1.4826 * mad.replace(0, np.nan)
        deviation = (s - med).abs() / scaled
        flagged = (deviation > OUTLIER_MAD_THRESHOLD).fillna(False)
        df.loc[grp.index, "is_outlier"] = (flagged & ~explained.loc[grp.index]).astype(int)

    return df


# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Join Egyptian calendar features onto a dated frame."""
    cal = CALENDAR.feature_frame(df["date"].min(), df["date"].max())
    cal["date"] = pd.to_datetime(cal["date"])
    return df.merge(cal.drop(columns=["holiday_name"]), on="date", how="left")


def add_lags(df: pd.DataFrame, horizon: int, target: str = "sales_qty") -> pd.DataFrame:
    """Lag and rolling features, all shifted by at least `horizon` days.

    Shifting by the horizon is what makes the backtest honest: predicting a week ahead
    with yesterday's sales would score beautifully and be unusable in production.
    """
    df = df.sort_values(["sku", "date"]).copy()
    g = df.groupby("sku", sort=False)[target]

    for lag in LAG_DAYS:
        if lag < horizon:
            continue  # not yet observable at prediction time
        df[f"lag_{lag}"] = g.shift(lag)

    for win in ROLL_WINDOWS:
        base = g.shift(horizon)
        df[f"roll_mean_{win}"] = base.rolling(win, min_periods=2).mean()
        df[f"roll_std_{win}"] = base.rolling(win, min_periods=2).std()

    # Same weekday, previous weeks -- the strongest single predictor for bakeries.
    same_dow = max(7, int(np.ceil(horizon / 7) * 7))
    df["lag_same_dow"] = g.shift(same_dow)
    df["mean_last_4_same_dow"] = (
        g.shift(same_dow).groupby(df["sku"]).rolling(28, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Short-term momentum: is this item trending up or down going into the horizon?
    df["trend_7_28"] = df["roll_mean_7"] / df["roll_mean_28"].replace(0, np.nan)

    # Reference level used by ratio-mode training (see forecaster.LightGBMQuantile).
    # A 28-day window is deliberately long: it is slow enough that a week of Ramadan
    # does not drag the baseline down with it, which would hide the very effect we want
    # the calendar features to explain.
    baseline = df["roll_mean_28"]
    for fallback in ("roll_mean_14", "roll_mean_7", "lag_same_dow"):
        if fallback in df.columns:
            baseline = baseline.fillna(df[fallback])
    df["baseline_level"] = baseline.clip(lower=0.0)

    # A second, EVENT-FREE baseline used only for estimating calendar multipliers.
    #
    # `baseline_level` above is contaminated for this purpose: by the middle of Ramadan
    # its 28-day window is mostly Ramadan days, so demand/baseline drifts back toward 1
    # and the estimated effect collapses. Measured here, konafa's true 4.5x came out as
    # 1.38x. Averaging only over quiet days keeps the reference at the item's normal
    # level, so the ratio recovers the real multiplier.
    if {"is_ramadan", "is_public_holiday", "is_kahk_window"} <= set(df.columns):
        quiet = (
            (df["is_ramadan"] == 0)
            & (df["is_public_holiday"] == 0)
            & (df["is_kahk_window"] == 0)
        )
        quiet_target = df[target].where(quiet)
        # Longer window than usual, since dropping event days thins the sample.
        clean = (
            quiet_target.groupby(df["sku"], sort=False)
            .transform(lambda s: s.shift(horizon).rolling(70, min_periods=5).mean())
        )
        clean = clean.groupby(df["sku"], sort=False).transform(lambda s: s.ffill().bfill())
        df["clean_baseline"] = clean.fillna(df["baseline_level"]).clip(lower=0.0)
    else:
        df["clean_baseline"] = df["baseline_level"]

    # Scale-free versions of every history feature.
    #
    # Ratio-mode training only works if the FEATURES are level-free as well as the
    # target. Given raw `lag_1` alongside a ratio target, the model simply reconstructs
    # the level from the lags and recovers the same shortcut we were trying to remove.
    # Dividing each history feature by the same baseline leaves momentum and volatility
    # information intact while stripping out the level.
    safe_base = df["baseline_level"].clip(lower=1.0)
    for col in [c for c in df.columns if c.startswith(("lag_", "roll_mean_", "roll_std_"))]:
        df[f"{col}_rel"] = (df[col] / safe_base).clip(0.0, 10.0)

    return df


def build_features(
    df: pd.DataFrame,
    horizon: int = 1,
    target: str = "sales_qty",
) -> pd.DataFrame:
    """Full pipeline: clean -> calendar -> lags -> outlier flags.

    Returns every row; callers decide what to drop. `training_mask` encodes which rows
    are safe to fit on.
    """
    out = reconcile_stock(df)
    out = mark_closed_days(out)
    out = add_calendar(out)
    out = add_lags(out, horizon=horizon, target=target)
    out = flag_outliers(out, target=target)

    out["sku_code"] = out["sku"].astype("category").cat.codes
    # Newsvendor q* comes from the data's own economics (unit_price/unit_cost/shelf
    # life), not a fixed catalogue -- every SKU in the frame, known or brand new, gets
    # a per-item service level. Defaults to 0.5 when economics are missing.
    for col in ("shelf_life_days", "unit_price", "unit_cost"):
        if col not in out.columns:
            out[col] = 0 if col != "shelf_life_days" else 1
    out["newsvendor_q"] = out[["sku", "unit_price", "unit_cost", "shelf_life_days"]].apply(
        lambda r: newsvendor_q_from_row(r.to_dict()),
        axis=1,
    )
    return out.reset_index(drop=True)


def training_mask(df: pd.DataFrame, exclude_stockouts: bool = True) -> pd.Series:
    """Rows safe to train on.

    Stockout days are right-censored: sales were capped by supply, so the recorded
    value is a lower bound on demand rather than demand itself. Fitting on them teaches
    the model that the ceiling it hit *was* the demand, which drives production down,
    which causes more stockouts. Excluding them breaks that feedback loop.
    """
    mask = (
        df["sales_qty"].notna()
        & (df["is_closed"] == 0)
        & (df["is_outlier"] == 0)
        & df["lag_same_dow"].notna()
    )
    if exclude_stockouts:
        mask &= df["is_stockout"].fillna(0) == 0
    return mask


def feature_columns(df: pd.DataFrame, mode: str = "level") -> list[str]:
    """Model input columns: calendar + engineered history, never leaky same-day fields.

    `mode="ratio"` swaps the raw history features for their scale-free `_rel` versions,
    so the model cannot recover the demand level and must explain deviation from the
    calendar instead.
    """
    calendar = [c for c in CALENDAR_FEATURES if c in df.columns]

    if mode == "ratio":
        history = [c for c in df.columns if c.endswith("_rel")]
        history += [c for c in ("trend_7_28",) if c in df.columns]
    else:
        history = [
            c for c in df.columns
            if (c.startswith(("lag_", "roll_", "mean_last_4")) or c == "trend_7_28")
            and not c.endswith("_rel")
        ]

    return calendar + history + ["sku_code"]
