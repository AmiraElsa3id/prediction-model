"""
Load the real French bakery Kaggle dataset (already used once this session, via the
rule-based bridge with Egypt's calendar -- that lost to naive, WAPE 0.605). This time:
same real data, no date shift (real 2021-2022 dates, matched against France's own real
public holidays), through the actual CalendarDecomposed architecture instead of the
rule-based bridge.

Run:  .venv/Scripts/python.exe scripts/load_french.py

Reuses the exact cleaning logic from `seed_bakery_excel.py` (price parsing for the
"0,90 €" format, dropping non-positive quantities) but skips the Excel/JSON export --
this only needs the long dataframe for the backtest.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[2] / "kaggle-raw" / "Bakery sales.csv"
N_SERIES = 30
RANDOM_STATE = 42

_NUMERIC_CHARS = re.compile(r"[^0-9,.\-]")


def _parse_price(val) -> float:
    """Same parser as seed_bakery_excel.py -- French comma-decimal, currency symbol."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = _NUMERIC_CHARS.sub("", str(val)).replace(",", ".")
    if not cleaned or cleaned in {"-", "."}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def load_french_subset(n_series: int = N_SERIES) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]

    df["unit_price"] = df["unit_price"].apply(_parse_price)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["article"] = df["article"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["date", "Quantity", "article"])
    df = df[(df["Quantity"] > 0) & (df["article"].str.len() > 0)]  # drop returns

    daily = (
        df.groupby(["date", "article"])["Quantity"].sum().reset_index()
        .rename(columns={"Quantity": "sales_qty", "article": "sku"})
    )

    # Dense grid per article -- a day the article doesn't appear means zero sold,
    # not missing data (same distinction made for M5/Favorita).
    top_articles = (
        daily.groupby("sku")["sales_qty"].sum().sort_values(ascending=False)
    )
    available = top_articles.index.tolist()
    chosen = (
        pd.Series(available).sample(n=min(n_series, len(available)), random_state=RANDOM_STATE)
        .tolist()
    )
    daily = daily[daily["sku"].isin(chosen)]

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    frames = []
    for sku, grp in daily.groupby("sku"):
        g = grp.set_index("date")["sales_qty"].reindex(full_range, fill_value=0.0)
        g = g.to_frame()
        g["sku"] = sku
        g.index.name = "date"
        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    return out[["date", "sku", "sales_qty"]].sort_values(["sku", "date"]).reset_index(drop=True)


def main() -> None:
    df = load_french_subset()
    print(f"Series selected: {df['sku'].nunique()} (seed={RANDOM_STATE})")
    print(f"Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Total rows: {len(df):,}")
    print()
    print("Sample SKUs:")
    for sku in df["sku"].unique()[:8]:
        print(f"  {sku}")
    print()
    print("Overall daily total, first 5 days:")
    print(df.groupby("date")["sales_qty"].sum().head())


if __name__ == "__main__":
    main()
