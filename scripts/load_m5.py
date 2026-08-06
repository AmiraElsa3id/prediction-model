"""
Load a bounded subset of the real M5 (Walmart) dataset, reshaped to the same
`date, sku, sales_qty` long format `app/core/generate.py` produces for the synthetic
Egyptian data -- so the rest of the pipeline (features, backtest, models) can treat
both interchangeably.

Run:  .venv/Scripts/python.exe scripts/load_m5.py

Scope: the full M5 set is 30,490 item-store series over 1,969 days -- too large to run
the Ridge+LightGBM decomposition over in reasonable time (see plan). This samples a
fixed, seeded subset of FOODS-category series from one store (CA_1) -- closest category
analog to a bakery, and comparable in scale (~30 series) to the 11-SKU Egyptian
catalogue every other backtest this session has used.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "m5"
STORE = "CA_1"
CATEGORY = "FOODS"
N_SERIES = 30
RANDOM_STATE = 42


def load_calendar() -> pd.DataFrame:
    """d_N -> real calendar date, plus the columns m5_calendar.py needs."""
    cal = pd.read_csv(DATA_DIR / "calendar.csv")
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def load_m5_subset(
    store: str = STORE, category: str = CATEGORY, n_series: int = N_SERIES,
) -> pd.DataFrame:
    """Long-format subset: date, sku, sales_qty, store_id, cat_id.

    `sku` is M5's `id` column (item_id + store_id + '_evaluation') -- unique per series,
    playing the same role `sku` plays for the Egyptian catalogue.
    """
    cal = load_calendar()
    d_to_date = dict(zip(cal["d"], cal["date"]))

    wide = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    subset = wide[(wide["store_id"] == store) & (wide["cat_id"] == category)]
    if len(subset) > n_series:
        subset = subset.sample(n=n_series, random_state=RANDOM_STATE)

    d_cols = [c for c in wide.columns if c.startswith("d_")]
    long = subset.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=d_cols, var_name="d", value_name="sales_qty",
    )
    long["date"] = long["d"].map(d_to_date)
    long = long.rename(columns={"id": "sku"})
    long = long[["date", "sku", "sales_qty", "store_id", "cat_id", "item_id"]]
    return long.sort_values(["sku", "date"]).reset_index(drop=True)


def main() -> None:
    cal = load_calendar()
    df = load_m5_subset()

    print(f"Store: {STORE}   Category: {CATEGORY}")
    print(f"Series selected: {df['sku'].nunique()} (of "
          f"{cal.shape[0]} calendar days available, seed={RANDOM_STATE})")
    print(f"Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Total rows: {len(df):,}")
    print()
    print("Sample SKUs:")
    for sku in df["sku"].unique()[:5]:
        print(f"  {sku}")
    print()
    print("Overall daily total, first 5 days:")
    print(df.groupby("date")["sales_qty"].sum().head())
    events = cal[cal["event_name_1"].notna()]
    print(f"\nCalendar events in range: {len(events)}")
    print(events[["date", "event_name_1", "event_type_1"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
