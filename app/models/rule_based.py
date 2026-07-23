"""Rule-based cold-start forecaster.

WHY THIS EXISTS
---------------
A brand-new bakery has no sales history, so there is nothing to train a model on. But it
still needs a production number on day one. This forecaster provides that using two
things that exist before any data does:

  1. A per-item baseline the owner gives at onboarding (typical units/day), refined by
     whatever few days of real sales have trickled in.
  2. Hand-encoded Egyptian calendar rules -- "croissants roughly halve in Ramadan",
     "kahk only sells in the ten days before Eid al-Fitr" -- taken from the item
     catalogue's prior multipliers.

It is deliberately NOT a model: no fitting, no lags, no risk of overfitting two data
points. It is the honest answer to "what should I bake before I know anything", and it
hands over to the trained model (`CalendarDecomposed`) once an item has ~90 days of
history. See `ForecastService` for the switch.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from app.core.egypt_calendar import CALENDAR
from app.core.items import BY_SKU, Item
from app.core.market_priors import load_priors

# Days of recent normal-day sales used to refine the owner's baseline estimate.
RECENT_WINDOW = 28
# Below this many observed days, trust the owner's prior more than the thin sample.
MIN_DAYS_FOR_DATA_BASELINE = 7
# Rough prediction interval around the point estimate (cold start is uncertain).
INTERVAL_LOW, INTERVAL_HIGH = 0.65, 1.40


def rule_multiplier(priors: dict, feats: dict) -> tuple[float, list[dict]]:
    """Calendar multiplier from the market priors, plus a human-readable breakdown.

    `priors` is a resolved per-item dict from `market_priors.load_priors` (LLM/market
    analysis, corrected by data over time). Returns the combined multiplier and the list
    of factors that moved it, so the API explains the number exactly as it does for the
    trained model.
    """
    m = 1.0
    factors: list[dict] = []

    def apply(mult: float, label: str) -> None:
        nonlocal m
        if abs(mult - 1.0) < 1e-9:
            return
        m *= mult
        factors.append({
            "factor": label,
            "impact_pct": round((mult - 1.0) * 100, 1),
            "direction": "increase" if mult > 1 else "decrease",
        })

    if feats["is_weekend"]:
        apply(priors["weekend"], "Weekend")

    if feats["is_ramadan"]:
        ram = priors["ramadan"]
        if feats["is_last_ten_of_ramadan"]:
            ram *= priors["ramadan_late"]
        apply(ram, "Ramadan")

    if feats["is_kahk_window"] and priors["kahk_peak"] > 1.0:
        apply(1.0 + (priors["kahk_peak"] - 1.0) * (feats["kahk_ramp"] ** 1.8), "Kahk season")

    if feats["is_eid_fitr"] or feats["is_eid_adha"]:
        apply(priors["eid"], "Eid")

    if feats["is_school_term"]:
        apply(priors["school"], "School term")

    if feats["is_payday_window"]:
        apply(priors["payday"], "Payday period")

    if feats["is_sham_el_nessim"]:
        apply(priors["sham_el_nessim"], "Sham El-Nessim")

    if feats["is_public_holiday"] and not (
        feats["is_eid_fitr"] or feats["is_eid_adha"] or feats["is_sham_el_nessim"]
    ):
        apply(priors["holiday"], "Public holiday")

    factors.sort(key=lambda f: abs(f["impact_pct"]), reverse=True)
    return m, factors


class RuleBasedForecaster:
    """Cold-start forecaster driven by owner priors and calendar rules."""

    def __init__(self, priors: dict[str, dict] | None = None) -> None:
        # Per-item baseline "normal day" level. Seeded from catalogue priors, refined by
        # observed sales as they arrive.
        self.baselines: dict[str, float] = {
            sku: item.base_daily_demand for sku, item in BY_SKU.items()
        }
        self.observed_days: dict[str, int] = {sku: 0 for sku in BY_SKU}
        # Calendar sensitivities from market analysis (LLM/file), correctable by data.
        self.priors = priors or load_priors()

    def update_baseline(self, history: pd.DataFrame) -> "RuleBasedForecaster":
        """Refine each item's baseline from recent normal-day sales.

        Uses only quiet days (no event, no weekend) so the baseline stays a true
        "ordinary day" level; the calendar multiplier re-adds event effects at predict
        time. Stockout and closed days are skipped -- they misrepresent demand.
        """
        if history is None or history.empty:
            return self

        df = history.copy()
        df["date"] = pd.to_datetime(df["date"])
        cal = CALENDAR.feature_frame(df["date"].min(), df["date"].max())
        cal["date"] = pd.to_datetime(cal["date"])
        keep = ["date", "is_ramadan", "is_public_holiday", "is_kahk_window", "is_weekend"]
        df = df.merge(cal[keep], on="date", how="left")

        for sku, grp in df.groupby("sku"):
            self.observed_days[sku] = grp["date"].nunique()
            quiet = grp[
                (grp.get("is_ramadan", 0) == 0)
                & (grp.get("is_public_holiday", 0) == 0)
                & (grp.get("is_kahk_window", 0) == 0)
                & (grp.get("is_weekend", 0) == 0)
                & (grp.get("is_stockout", 0) == 0)
            ]
            recent = quiet.sort_values("date").tail(RECENT_WINDOW)
            if len(recent) >= MIN_DAYS_FOR_DATA_BASELINE:
                self.baselines[sku] = float(recent["sales_qty"].mean())
            elif len(recent) > 0 and sku in BY_SKU:
                # Blend the thin sample with the prior rather than trusting either alone.
                w = len(recent) / MIN_DAYS_FOR_DATA_BASELINE
                self.baselines[sku] = float(
                    w * recent["sales_qty"].mean() + (1 - w) * BY_SKU[sku].base_daily_demand
                )
        return self

    def forecast(self, sku: str, target_date: dt.date) -> dict:
        """Point forecast + interval + factors for one item/day."""
        if sku not in BY_SKU:
            raise KeyError(f"unknown SKU {sku!r}")
        item = BY_SKU[sku]
        feats = CALENDAR.features(target_date)

        # Seasonal items barely sell outside their window; don't bake a full baseline.
        base = self.baselines.get(sku, item.base_daily_demand)
        mult, factors = rule_multiplier(self.priors[sku], feats)
        if item.seasonal_only and mult < 2.0:
            base *= 0.04

        qty = base * mult
        return {
            "sku": sku,
            "date": target_date,
            "quantity": int(round(max(qty, 0))),
            "lower": int(round(max(qty * INTERVAL_LOW, 0))),
            "upper": int(round(qty * INTERVAL_HIGH)),
            "factors": factors,
            "observed_days": self.observed_days.get(sku, 0),
        }
