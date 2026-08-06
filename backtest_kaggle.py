"""
Walk-forward backtest of the RestoMind bridge (rule_based_learned) against the real
Kaggle French bakery data, to answer: does this forecast beat a naive baseline?

Replicates the exact production logic:
  registry.RestaurantState.ingest()   -- learned level = mean of quiet weekdays, 42-day window
  restomind._forecast_one()           -- level * rule_multiplier(category_priors, calendar_features)

For each product and each of the last N weeks, "predict" using only data strictly
BEFORE that week (no leakage), then compare to the actual total for that week.
Reports WAPE for: the bridge, and two naive baselines.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.core.market_priors import category_priors
from app.models.rule_based import rule_multiplier

N_BACKTEST_WEEKS = 8
QUIET_WINDOW = 42
MIN_DAYS_FOR_LEARNED = 14
TOP_N_PRODUCTS = 20  # by total volume, so results aren't dominated by rare noisy items

CATEGORY_PRIOR_MAP = {
    "Bread": "bread", "Pastry": "pastry", "Cake": "cake",
    "Dry Biscuit": "dry", "Savoury": "savoury", "Bakery Bread Other": "bread",
}


def learned_level(history: pd.DataFrame, article: str, as_of: dt.date) -> float | None:
    """Exact registry.ingest() logic: mean of quiet (non-weekend/event) days, last 42,
    using only rows strictly before `as_of` -- no leakage of the week being predicted."""
    sub = history[(history["article"] == article) & (history["date"] < pd.Timestamp(as_of))]
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
    """Exact restomind.predict_week() logic: sum of 7 daily point estimates."""
    priors = category_priors(CATEGORY_PRIOR_MAP.get(category, ""))
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
    data = json.load(open("../bakery_aggregated.json", encoding="utf-8"))
    sales = pd.DataFrame(data["sales"])
    sales["date"] = pd.to_datetime(sales["date"])
    products = {p["article"]: p["category"] for p in data["products"]}

    last_date = sales["date"].max()
    # Align backtest weeks to the data's own weekly grid, walking back from the end.
    first_week_start = last_date.normalize() - pd.Timedelta(days=int(last_date.dayofweek))
    week_starts = [
        (first_week_start - pd.Timedelta(weeks=k)).date()
        for k in range(N_BACKTEST_WEEKS, 0, -1)
    ]

    top_articles = (
        sales.groupby("article")["qty"].sum().sort_values(ascending=False).head(TOP_N_PRODUCTS).index.tolist()
    )

    rows = []
    for article in top_articles:
        category = products.get(article, "")
        for ws in week_starts:
            we = ws + dt.timedelta(days=6)
            actual = sales[
                (sales["article"] == article)
                & (sales["date"] >= pd.Timestamp(ws)) & (sales["date"] <= pd.Timestamp(we))
            ]["qty"].sum()

            level = learned_level(sales, article, ws)
            if level is None:
                continue  # not enough history yet -- matches production behaviour (falls back to owner estimate)

            bridge_pred = bridge_predict_week(level, category, ws)

            # Naive baseline 1: same week last year (if available), else previous week.
            prev_week_actual = sales[
                (sales["article"] == article)
                & (sales["date"] >= pd.Timestamp(ws) - pd.Timedelta(days=7))
                & (sales["date"] < pd.Timestamp(ws))
            ]["qty"].sum()

            # Naive baseline 2: flat level (no calendar multiplier at all).
            flat_pred = level * 7

            rows.append({
                "article": article, "week": str(ws), "actual": actual,
                "bridge_pred": bridge_pred, "prev_week_pred": prev_week_actual,
                "flat_pred": flat_pred,
            })

    df = pd.DataFrame(rows)
    print(f"Backtest rows: {len(df)}  (products x weeks with enough history)\n")

    overall = {
        "bridge (level x Egyptian calendar)": wape(df["actual"].values, df["bridge_pred"].values),
        "naive: same qty as last week":       wape(df["actual"].values, df["prev_week_pred"].values),
        "naive: flat learned level x 7":      wape(df["actual"].values, df["flat_pred"].values),
    }
    print("Overall WAPE (lower is better, 0 = perfect, 1.0 = off by 100% on average):")
    for name, v in overall.items():
        print(f"  {name:38s} {v:.3f}")

    print("\nPer-product WAPE (bridge vs naive-last-week):")
    for article, g in df.groupby("article"):
        b = wape(g["actual"].values, g["bridge_pred"].values)
        n = wape(g["actual"].values, g["prev_week_pred"].values)
        flag = "  <-- bridge WORSE than naive" if b > n else ""
        print(f"  {article:28s} bridge={b:.3f}  naive={n:.3f}{flag}")


if __name__ == "__main__":
    main()
