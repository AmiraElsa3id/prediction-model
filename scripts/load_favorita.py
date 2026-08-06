"""
Load a bounded subset of the real Favorita (Corporación Favorita, Ecuador) grocery
dataset, reshaped to the same `date, sku, sales_qty` long format used for M5 and the
synthetic Egyptian data.

Run:  .venv/Scripts/python.exe scripts/load_favorita.py

Unlike M5, `train.csv` (~5 GB, ~125M rows) is already long-format (one row per
date/store/item observation) rather than a wide day-columns grid, and it is NOT dense --
a missing (date, item) row means no recorded sale that day, not "unknown". Read
chunked, filtering to one store x the `BREAD/BAKERY` item family as we go (the closest
category match to this project's actual domain -- better than M5's generic FOODS), then
reindexed to a full daily grid per item so absent days become explicit zeros.

`unit_sales` can be negative in this dataset (returns) -- dropped, same treatment as
the negative-quantity rows filtered out of the Kaggle French bakery CSV earlier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "favorita"
STORE = 44          # Quito, type A -- a large, representative store (stores.csv)
FAMILY = "BREAD/BAKERY"
N_SERIES = 30
RANDOM_STATE = 42
CHUNK_SIZE = 2_000_000


def _bakery_item_ids() -> set[int]:
    items = pd.read_csv(DATA_DIR / "items.csv")
    return set(items.loc[items["family"] == FAMILY, "item_nbr"])


def load_favorita_subset(
    store: int = STORE, family: str = FAMILY, n_series: int = N_SERIES,
) -> pd.DataFrame:
    item_ids = _bakery_item_ids()

    dtypes = {"store_nbr": "int32", "item_nbr": "int32", "unit_sales": "float32"}
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        dtype=dtypes, parse_dates=["date"], chunksize=CHUNK_SIZE,
    ):
        match = chunk[(chunk["store_nbr"] == store) & chunk["item_nbr"].isin(item_ids)]
        if not match.empty:
            chunks.append(match)

    raw = pd.concat(chunks, ignore_index=True)
    raw = raw[raw["unit_sales"] > 0]  # drop returns/corrections (negative rows)

    available = raw["item_nbr"].unique()
    rng_items = pd.Series(available).sample(
        n=min(n_series, len(available)), random_state=RANDOM_STATE
    ).tolist()
    raw = raw[raw["item_nbr"].isin(rng_items)]

    # Dense daily grid per item -- an absent row means zero sales that day, not
    # missing data (Favorita's train.csv only records days something sold).
    daily = raw.groupby(["item_nbr", "date"])["unit_sales"].sum().reset_index()
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

    frames = []
    for item, grp in daily.groupby("item_nbr"):
        g = grp.set_index("date").reindex(full_range, fill_value=0.0)
        g["item_nbr"] = item
        g.index.name = "date"
        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"unit_sales": "sales_qty"})
    out["sku"] = out["item_nbr"].apply(lambda i: f"FAVORITA_{i}_S{store}")
    return out[["date", "sku", "sales_qty"]].sort_values(["sku", "date"]).reset_index(drop=True)


def main() -> None:
    df = load_favorita_subset()
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv")

    print(f"Store: {STORE}   Family: {FAMILY}")
    print(f"Series selected: {df['sku'].nunique()} (seed={RANDOM_STATE})")
    print(f"Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Total rows: {len(df):,}")
    print()
    print("Sample SKUs:")
    for sku in df["sku"].unique()[:5]:
        print(f"  {sku}")
    print()
    print("Overall daily total, first 5 days:")
    print(df.groupby("date")["sales_qty"].sum().head())
    print(f"\nHoliday/event rows in holidays_events.csv: {len(holidays)}")
    print(holidays.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
