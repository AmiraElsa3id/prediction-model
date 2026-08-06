"""
Export the model's own synthetic Egyptian bakery data (app/core/generate.py + items.py)
as CSVs shaped for RestoMind's CSV import feature -- so the CSV-import path can be
exercised with Egyptian-calendar-shaped demand instead of the French Kaggle set.

    .venv/Scripts/python.exe export_egyptian_import_csvs.py

Writes to seeds/egyptian-import/ (repo root, alongside seeds/kaggle-import/):
  1_menu_items.csv        11 products, Arabic titles (matches seed.ts / seed_bakery_history.py
                           convention), category = the model's own category string so
                           map_category() in the bridge resolves it correctly.
  2_sales_history.csv     2 years of daily sales_qty per product (what a POS actually
                           records -- already stockout-censored, same as real data would be).

IMPORTANT CAVEAT: the CSV import schema (title, category, price, freshnessWindow,
description) has no `sku` field. Products created this way will NOT be linked to the
trained CalendarDecomposed model the way `seed_bakery_history.py`'s direct-to-Mongo
insert does (that script sets `sku` explicitly). They will use the rule-based bridge
(`rule_based_learned` once enough history exists) -- the same mechanism already
validated against the French data, just with the correct calendar this time.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from app.core.generate import generate
from app.core.items import CATALOGUE

OUT_DIR = Path(__file__).resolve().parents[1] / "seeds" / "egyptian-import"
MENU_HEADERS = ["title", "category", "price", "freshnessWindow", "description"]
SALES_HEADERS = ["date", "productId", "quantitySold", "sellingPrice", "basePrice"]


def week_aligned_shift(last_date: dt.date, target: dt.date) -> int:
    """Same technique used for the Kaggle conversion: shift by a whole number of
    weeks so every SKU keeps its real day-of-week profile (weekend uplift etc.)."""
    raw = (target - last_date).days
    return round(raw / 7) * 7


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate()
    orig_min, orig_max = df["date"].min().date(), df["date"].max().date()
    shift_days = week_aligned_shift(orig_max, dt.date.today())
    df["date"] = df["date"] + pd.Timedelta(days=shift_days)
    new_min, new_max = df["date"].min().date(), df["date"].max().date()

    print(f"Original range : {orig_min} .. {orig_max}")
    print(f"Shifted +{shift_days} days ({shift_days // 7} weeks) -> {new_min} .. {new_max}")

    # -- menu items ----------------------------------------------------------------
    menu_path = OUT_DIR / "1_menu_items.csv"
    with menu_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=MENU_HEADERS)
        w.writeheader()
        for item in CATALOGUE:
            w.writerow({
                "title": item.name_ar,
                "category": item.category,  # bread/pastry/cake/... -- resolves via map_category
                "price": f"{item.unit_price:.2f}",
                "freshnessWindow": item.shelf_life_days,
                "description": item.name_en,
            })
    print(f"\n  {menu_path}  ({len(CATALOGUE)} products)")

    # -- sales history ---------------------------------------------------------------
    sku_to_name = {i.sku: i.name_ar for i in CATALOGUE}
    sku_to_price = {i.sku: i.unit_price for i in CATALOGUE}

    sales_path = OUT_DIR / "2_sales_history.csv"
    rows = df[df["sales_qty"] > 0].sort_values(["date", "sku"])
    with sales_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SALES_HEADERS)
        w.writeheader()
        for r in rows.itertuples():
            price = sku_to_price[r.sku]
            w.writerow({
                "date": r.date.date().isoformat(),
                "productId": sku_to_name[r.sku],
                "quantitySold": int(r.sales_qty),
                "sellingPrice": f"{price:.2f}",
                "basePrice": f"{price:.2f}",
            })
    print(f"  {sales_path}  ({len(rows):,} rows)")
    print(f"\nDate range: {new_min} .. {new_max}")
    print("\nImport order (menu items MUST be first):")
    print("  1. 1_menu_items.csv     importType=menu_items")
    print("  2. 2_sales_history.csv  importType=sales_history")


if __name__ == "__main__":
    main()
