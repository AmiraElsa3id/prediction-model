"""Metrics and rolling-origin backtesting.

WHY NOT MAPE
    MAPE is the default in demand forecasting and is wrong here. Bakery items hit zero
    (kahk for 11 months of the year), making the percentage undefined or explosive. It
    also punishes over-forecasting more than under-forecasting, which quietly biases
    model selection towards under-production. WAPE is the headline instead: same
    interpretation, defined at zero, symmetric.

WHY PINBALL
    We train a quantile, so we must score a quantile. MAE would reward the conditional
    median and silently undo the newsvendor logic.

WHY ROLLING ORIGIN
    A single train/test cut gives one noisy number that depends entirely on where the
    cut landed. Expanding-window backtesting scores across many origins and mirrors how
    the model is actually retrained in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|. Defined at zero."""
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error. Positive = under-forecasting, negative = over-forecasting.

    Reported because direction matters more than magnitude here: systematic
    over-forecasting is exactly the waste we are trying to remove.
    """
    return float((y_true - y_pred).mean())


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Quantile loss. The metric the LightGBM objective is actually optimising."""
    delta = y_true - y_pred
    return float(np.maximum(q * delta, (q - 1) * delta).mean())


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 7) -> float:
    """Mean absolute scaled error against an in-sample seasonal-naive benchmark.

    Scale-free, so it can be averaged across items whose volumes differ by 100x.
    MASE < 1 means the model beats seasonal naive.
    """
    if len(y_train) <= season:
        return float("nan")
    scale = np.abs(y_train[season:] - y_train[:-season]).mean()
    if scale == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).mean() / scale)


def business_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, unit_cost: float,
    unit_price: float, spoilage_severity: float = 1.0,
) -> dict:
    """Translate forecast error into money -- the only metric a bakery owner cares about.

    Assumes production is set to the forecast, so over-forecasting creates waste and
    under-forecasting creates lost sales.
    """
    over = np.maximum(y_pred - y_true, 0)
    under = np.maximum(y_true - y_pred, 0)
    waste_cost = float((over * unit_cost * spoilage_severity).sum())
    lost_margin = float((under * (unit_price - unit_cost)).sum())
    return {
        "units_wasted": float(over.sum()),
        "units_short": float(under.sum()),
        "waste_cost_egp": round(waste_cost, 2),
        "lost_margin_egp": round(lost_margin, 2),
        "total_cost_egp": round(waste_cost + lost_margin, 2),
        "stockout_rate": float((under > 0).mean()),
    }


def score_all(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None = None,
    q: float = 0.5,
) -> dict:
    out = {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "pinball": pinball_loss(y_true, y_pred, q),
        "n": int(len(y_true)),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train)
    return out


# --------------------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------------------


@dataclass
class Fold:
    """One expanding-window split."""

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(
    dates: pd.Series, n_folds: int = 6, test_days: int = 28, horizon: int = 1,
) -> list[Fold]:
    """Expanding-window folds walking forward through time.

    A `horizon` gap sits between train_end and test_start: when forecasting h days
    ahead, the h days immediately before the target are not yet observable. Without the
    gap the backtest is optimistic in a way production never will be.
    """
    dmin, dmax = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    folds: list[Fold] = []

    for i in range(n_folds, 0, -1):
        test_end = dmax - pd.Timedelta(days=(i - 1) * test_days)
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        train_end = test_start - pd.Timedelta(days=horizon)
        if train_end <= dmin + pd.Timedelta(days=60):
            continue  # not enough history to fit anything meaningful
        folds.append(Fold(train_end=train_end, test_start=test_start, test_end=test_end))

    return folds


def backtest(
    df: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    target: str = "sales_qty",
    eval_target: str | None = None,
    n_folds: int = 6,
    test_days: int = 28,
    horizon: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Run a rolling-origin backtest.

    `fit_predict(train_df, test_df) -> predictions` keeps this agnostic to the model,
    so the naive baseline, LightGBM, and the online learner all run through the exact
    same harness and are directly comparable.

    `eval_target` allows scoring against a different column than the one trained on.
    On synthetic data we train on censored `sales_qty` but score against `true_demand`,
    which is the only honest way to test whether the censoring correction worked.
    """
    eval_target = eval_target or target
    folds = make_folds(df["date"], n_folds=n_folds, test_days=test_days, horizon=horizon)
    if not folds:
        raise ValueError("no valid folds -- not enough history for this configuration")

    records: list[pd.DataFrame] = []

    for k, fold in enumerate(folds):
        train = df[df["date"] <= fold.train_end]
        test = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
        if train.empty or test.empty:
            continue

        preds = fit_predict(train, test)
        rec = test[["date", "sku", target]].copy()
        rec["y_true"] = test[eval_target].to_numpy()
        rec["y_pred"] = np.asarray(preds, dtype=float)
        rec["fold"] = k
        records.append(rec)

    results = pd.concat(records, ignore_index=True)
    results = results.dropna(subset=["y_true", "y_pred"])

    summary = score_all(
        results["y_true"].to_numpy(),
        results["y_pred"].to_numpy(),
        y_train=df[df["date"] <= folds[0].train_end][target].dropna().to_numpy(),
    )
    return results, summary


def per_item_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Break a backtest down by SKU -- aggregate numbers hide items that are failing."""
    rows = []
    for sku, g in results.groupby("sku"):
        s = score_all(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        s["sku"] = sku
        rows.append(s)
    return pd.DataFrame(rows).set_index("sku").sort_values("wape", ascending=False)
