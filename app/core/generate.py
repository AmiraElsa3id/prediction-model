"""Synthetic Egyptian bakery POS data.

WHY THIS EXISTS
---------------
The client is pre-launch: there is no real sales history, and no public dataset
contains Egyptian calendar signal (Ramadan, Eid, Coptic feasts, Fri/Sat weekend).
So we generate history with *known* effect sizes taken from `items.CATALOGUE`.

That gives two things a real dataset could not:

1. A recovery test. Because we know the true Ramadan/weekend/kahk multipliers, we can
   assert the model rediscovers them. This proves the pipeline is *correct*.
2. True demand. Real POS data only records sales, which are censored by whatever was
   produced. Here we keep `true_demand` alongside `sales_qty`, so the censoring
   correction can be validated against ground truth.

IMPORTANT: numbers derived from this data are simulations, not measurements. Anything
shown to stakeholders must be labelled as such.

THE PRODUCTION POLICY
---------------------
`sales` alone would be an unrealistically clean signal. Real bakeries decide production
by gut feel, so we simulate a manager who averages the last week and pads for safety --
and who has no idea Ramadan is coming. Their misses on event days are precisely the
waste the AI is meant to remove, and they form the baseline for the business simulation.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.core.items import CATALOGUE, Item

# Manager pads the weekly average by this much, fearing empty shelves more than waste.
MANAGER_SAFETY_MARGIN = 1.25
MANAGER_LOOKBACK_DAYS = 7
# Production happens in trays/batches, so quantities are rounded up to these.
BATCH_SIZES = {"bread": 50, "pastry": 12, "cake": 6, "sweet": 10,
               "savoury": 6, "seasonal": 20, "dry": 5}


def _ramadan_multiplier(item: Item, feats: dict) -> float:
    """Ramadan effect as a curve over the month, not a flat flag.

    Daylight fasting suppresses breakfast items and lifts iftar desserts, and the
    effect intensifies over the month as the last ten days approach.
    """
    if not feats["is_ramadan"]:
        return 1.0
    mult = item.ramadan_mult
    if feats["is_last_ten_of_ramadan"]:
        mult *= item.ramadan_late_boost
    # Gentle intensification across the month (0.92x at day 1 -> 1.08x at the end).
    mult *= 0.92 + 0.16 * feats["ramadan_progress"]
    return mult


def _kahk_multiplier(item: Item, feats: dict) -> float:
    """Kahk ramps steeply in the ~10 days before Eid al-Fitr, then collapses."""
    if item.kahk_peak_mult <= 1.0:
        return 1.0
    if feats["is_kahk_window"]:
        # Convex ramp: demand accelerates as Eid closes in.
        return 1.0 + (item.kahk_peak_mult - 1.0) * (feats["kahk_ramp"] ** 1.8)
    return 1.0


def _demand_multiplier(item: Item, feats: dict, rng: np.random.Generator) -> float:
    """Compose all calendar effects multiplicatively."""
    m = 1.0
    if feats["is_weekend"]:
        m *= item.weekend_mult
    m *= _ramadan_multiplier(item, feats)
    m *= _kahk_multiplier(item, feats)
    if feats["is_eid_fitr"] or feats["is_eid_adha"]:
        m *= item.eid_mult
    if feats["is_school_term"]:
        m *= item.school_mult
    if feats["is_payday_window"]:
        m *= item.payday_mult
    if feats["is_sham_el_nessim"]:
        m *= item.sham_mult
    # Generic holiday lift, but don't double-count Eid/Sham which have their own terms.
    if feats["is_public_holiday"] and not (
        feats["is_eid_fitr"] or feats["is_eid_adha"] or feats["is_sham_el_nessim"]
    ):
        m *= item.holiday_mult
    return m


def _round_to_batch(qty: float, category: str) -> int:
    batch = BATCH_SIZES.get(category, 1)
    return int(np.ceil(max(qty, 0) / batch) * batch)


def generate(
    start: str = "2023-07-01",
    end: str = "2025-06-30",
    seed: int = 42,
    branch: str = "CAIRO_MAADI",
) -> pd.DataFrame:
    """Generate two years of daily per-item POS + inventory records.

    Returns one row per (date, sku) with the columns a real POS/inventory export
    would give us -- plus `true_demand`, which reality would not.
    """
    rng = np.random.default_rng(seed)
    cal = CALENDAR.feature_frame(start, end)
    cal_by_date = {row["date"]: row for row in cal.to_dict("records")}
    dates = sorted(cal_by_date)
    n_days = len(dates)

    rows: list[dict] = []

    for item in CATALOGUE:
        # Mild organic growth over the two years.
        trend = np.linspace(1.0, 1.22, n_days)
        recent_sales: list[float] = []

        for i, d in enumerate(dates):
            feats = cal_by_date[d]

            base = item.base_daily_demand * trend[i]
            mult = _demand_multiplier(item, feats, rng)

            in_season = not item.seasonal_only or mult >= 2.0
            if not in_season:
                # Kahk barely sells outside its window; keep a trickle, not a flatline.
                base *= 0.04

            mean_demand = base * mult
            # Lognormal multiplicative noise then Poisson counts: overdispersed,
            # non-negative, and integer -- like real unit sales.
            noisy = mean_demand * rng.lognormal(-0.5 * item.noise_sigma**2, item.noise_sigma)
            true_demand = int(rng.poisson(max(noisy, 0.01)))

            # --- manager decides production, knowing only recent sales ---
            if not in_season:
                # Nobody bakes kahk in October. Seasonal lines are simply not run
                # out of season, so there is no production and no waste.
                production = 0
            else:
                trailing = (
                    np.mean(recent_sales[-MANAGER_LOOKBACK_DAYS:])
                    if len(recent_sales) >= MANAGER_LOOKBACK_DAYS
                    else 0.0
                )
                if trailing < 1.0:
                    # No usable recent history: opening week, or a seasonal line
                    # restarting after months of zeros. Manager falls back to
                    # experience of what this item normally does.
                    planned = mean_demand * MANAGER_SAFETY_MARGIN
                else:
                    planned = trailing * MANAGER_SAFETY_MARGIN
                production = _round_to_batch(planned, item.category)

            sales = min(true_demand, production)
            leftover = production - sales
            # Right-censored: we sold everything and demand was still unmet.
            is_stockout = int(sales == production and true_demand > production)

            rows.append({
                "date": pd.Timestamp(d),
                "branch": branch,
                "sku": item.sku,
                "item_name_ar": item.name_ar,
                "category": item.category,
                "production_qty": production,
                "sales_qty": sales,
                "leftover_qty": leftover,
                "closing_stock": leftover,
                "is_stockout": is_stockout,
                "unit_price": item.unit_price,
                "unit_cost": item.unit_cost,
                "shelf_life_days": item.shelf_life_days,
                "revenue": round(sales * item.unit_price, 2),
                "waste_cost": round(leftover * item.unit_cost, 2),
                # Ground truth -- available here, never in production.
                "true_demand": true_demand,
            })
            recent_sales.append(sales)

    df = pd.DataFrame(rows).sort_values(["date", "sku"]).reset_index(drop=True)
    return _inject_data_quality_issues(df, rng)


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the messiness a real POS export always has.

    Without this the cleaning pipeline would be untested decoration. Two failure modes
    that genuinely occur and that the pipeline must survive:

      * closed days   -- rows missing entirely (must NOT be read as zero demand)
      * miscounts     -- inventory that violates production - sales = leftover
    """
    df = df.copy()

    # Bakery shuts on the first day of each Eid al-Fitr: drop those rows entirely.
    cal = CALENDAR.feature_frame(df["date"].min(), df["date"].max())
    closed = {
        pd.Timestamp(r["date"])
        for r in cal.to_dict("records")
        if r["eid_fitr_day_index"] == 1
    }
    df = df[~df["date"].isin(closed)].copy()

    # ~0.5% of rows carry an inventory miscount, breaking the stock identity.
    n_bad = max(1, int(len(df) * 0.005))
    bad_idx = rng.choice(df.index, size=n_bad, replace=False)
    df.loc[bad_idx, "leftover_qty"] = (
        df.loc[bad_idx, "leftover_qty"] + rng.integers(1, 8, size=n_bad)
    )
    df.loc[bad_idx, "closing_stock"] = df.loc[bad_idx, "leftover_qty"]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[2] / "data" / "synthetic_pos.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_parquet(out, index=False)

    print(f"SIMULATED DATA -- not measured from a real bakery")
    print(f"wrote {len(data):,} rows -> {out}")
    print(f"range: {data['date'].min().date()} .. {data['date'].max().date()}")
    print(f"items: {data['sku'].nunique()}")
    print(f"stockout rate: {data['is_stockout'].mean():.1%}")
    print(f"waste rate: {data['leftover_qty'].sum() / data['production_qty'].sum():.1%}")
    print(f"total waste cost: {data['waste_cost'].sum():,.0f} EGP over 2 years")
