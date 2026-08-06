"""
Walk-forward accuracy check of the RestoMind bridge against the Egyptian data the user
just imported through the CSV upload UI (Mongo `salestransactions`/`products`), using
the exact same held-out methodology as backtest_kaggle.py: predict each week using only
data strictly before it, compare to what actually happened.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from pymongo import MongoClient

from app.core.egypt_calendar import CALENDAR
from app.core.market_priors import category_priors
from app.models.rule_based import rule_multiplier

N_BACKTEST_WEEKS = 12
QUIET_WINDOW = 42
MIN_DAYS_FOR_LEARNED = 14


def learned_level(history: pd.DataFrame, pid, as_of: dt.date) -> float | None:
    sub = history[(history["productId"] == pid) & (history["date"] < pd.Timestamp(as_of))]
    if sub.empty:
        return None
    cal = CALENDAR.feature_frame(sub["date"].min(), sub["date"].max())
    cal["date"] = pd.to_datetime(cal["date"])
    merged = sub.merge(
        cal[["date", "is_ramadan", "is_public_holiday", "is_kahk_window", "is_weekend"]],
        on="date", how="left",
    )
    quiet = merged[
        (merged["is_ramadan"] == 0) & (merged["is_public_holiday"] == 0)
        & (merged["is_kahk_window"] == 0) & (merged["is_weekend"] == 0)
    ].sort_values("date").tail(QUIET_WINDOW)
    if len(quiet) < MIN_DAYS_FOR_LEARNED:
        return None
    return float(quiet["qty"].mean())


def bridge_predict_week(level: float, category: str, week_start: dt.date) -> float:
    priors = category_priors(category)
    total = 0.0
    for i in range(7):
        d = week_start + dt.timedelta(days=i)
        feats = CALENDAR.features(d)
        mult, _ = rule_multiplier(priors, feats)
        total += level * mult
    return total


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / actual.sum()) if actual.sum() > 0 else float("nan")


def main() -> None:
    db = MongoClient("mongodb://127.0.0.1:27017/restomind").get_default_database()

    products = {p["_id"]: p for p in db.products.find({})}
    categories = {c["_id"]: c["name"] for c in db.categories.find({})}
    cat_priors_map = {
        "bread": "bread", "pastry": "pastry", "cake": "cake", "sweet": "sweet",
        "savoury": "savoury", "seasonal": "seasonal", "dry": "dry",
    }

    sales = pd.DataFrame(list(db.salestransactions.find(
        {}, {"productId": 1, "date": 1, "quantitySold": 1}
    )))
    sales = sales.rename(columns={"quantitySold": "qty"})
    sales["date"] = pd.to_datetime(sales["date"])

    last_date = sales["date"].max()
    first_week_start = last_date.normalize() - pd.Timedelta(days=int(last_date.dayofweek))
    week_starts = [
        (first_week_start - pd.Timedelta(weeks=k)).date()
        for k in range(N_BACKTEST_WEEKS, 0, -1)
    ]

    rows = []
    for pid, p in products.items():
        category = categories.get(p.get("category"), "")
        category = cat_priors_map.get(category, category)
        for ws in week_starts:
            we = ws + dt.timedelta(days=6)
            actual = sales[
                (sales["productId"] == pid)
                & (sales["date"] >= pd.Timestamp(ws)) & (sales["date"] <= pd.Timestamp(we))
            ]["qty"].sum()

            level = learned_level(sales, pid, ws)
            if level is None:
                continue

            bridge_pred = bridge_predict_week(level, category, ws)
            prev_week_actual = sales[
                (sales["productId"] == pid)
                & (sales["date"] >= pd.Timestamp(ws) - pd.Timedelta(days=7))
                & (sales["date"] < pd.Timestamp(ws))
            ]["qty"].sum()
            flat_pred = level * 7

            rows.append({
                "title": p["description"], "week": str(ws), "actual": actual,
                "bridge_pred": bridge_pred, "prev_week_pred": prev_week_actual,
                "flat_pred": flat_pred,
            })

    df = pd.DataFrame(rows)
    print(f"Backtest rows: {len(df)}  ({df['title'].nunique()} products x up to {N_BACKTEST_WEEKS} weeks)\n")

    overall = {
        "bridge (level x Egyptian calendar)": wape(df["actual"].values, df["bridge_pred"].values),
        "naive: same qty as last week":       wape(df["actual"].values, df["prev_week_pred"].values),
        "naive: flat learned level x 7":      wape(df["actual"].values, df["flat_pred"].values),
    }
    print("Overall WAPE (lower is better):")
    for name, v in overall.items():
        print(f"  {name:38s} {v:.3f}")

    print("\nPer-product WAPE (bridge vs naive-last-week):")
    for title, g in df.groupby("title"):
        b = wape(g["actual"].values, g["bridge_pred"].values)
        n = wape(g["actual"].values, g["prev_week_pred"].values)
        flag = "  <-- bridge WORSE than naive" if b > n else ""
        print(f"  {title:18s} bridge={b:.3f}  naive={n:.3f}  (n={len(g)} weeks){flag}")


if __name__ == "__main__":
    main()
