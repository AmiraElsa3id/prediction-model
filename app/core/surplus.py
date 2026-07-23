"""Near-closing surplus detection and discount sizing.

Deliberately rule-based, not learned. Discount sizing is a price-elasticity problem,
and elasticity cannot be estimated from data that does not exist yet -- the bakery has
never run a discount, so there is nothing to learn from. A model fitted here would be
inventing its own training signal.

The rules encode the actual economics: a unit sold at any price above marginal cost
beats a unit in the bin, but discounting more than needed to clear the shelf is just
giving away margin. So discount depth scales with genuine risk, not with stock level.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.core.items import BY_SKU, Item

# Typical Egyptian bakery trading day.
DEFAULT_OPEN_HOUR = 7
DEFAULT_CLOSE_HOUR = 22

# Share of a day's sales that has typically happened by a given hour. Bakeries are
# front-loaded: the morning bread rush is most of the day's volume.
SELL_THROUGH_CURVE = {
    7: 0.05, 8: 0.15, 9: 0.27, 10: 0.37, 11: 0.45, 12: 0.53, 13: 0.60,
    14: 0.66, 15: 0.71, 16: 0.76, 17: 0.81, 18: 0.86, 19: 0.91,
    20: 0.95, 21: 0.98, 22: 1.00,
}

DISCOUNT_TIERS = [
    (0.80, 40, "خصم كبير"),
    (0.60, 30, "خصم كويس"),
    (0.40, 20, "خصم بسيط"),
    (0.20, 10, "خصم خفيف"),
]


def expected_sell_through(hour: int) -> float:
    """Fraction of the day's demand expected to have arrived by `hour`."""
    if hour <= DEFAULT_OPEN_HOUR:
        return 0.0
    if hour >= DEFAULT_CLOSE_HOUR:
        return 1.0
    return SELL_THROUGH_CURVE.get(hour, min(1.0, (hour - 7) / 15))


@dataclass
class SurplusItem:
    sku: str
    name_ar: str
    current_stock: int
    expected_remaining_sales: float
    projected_surplus: int
    risk_score: float
    urgency: str
    suggested_discount_pct: int
    value_at_risk_egp: float
    hours_to_close: float


def detect_surplus(
    stock: dict[str, int],
    daily_forecast: dict[str, float],
    now: dt.datetime,
    close_hour: int = DEFAULT_CLOSE_HOUR,
    min_risk: float = 0.15,
) -> list[SurplusItem]:
    """Find items at genuine risk of going to waste tonight.

    Risk is the share of current stock we do NOT expect to sell before closing, scaled
    by how perishable the item is. A long shelf life is not an emergency: 20 leftover
    petit fours keep for two weeks and should not trigger a discount, while 20 leftover
    baladi loaves are worthless tomorrow morning.
    """
    hours_left = max(0.0, close_hour - (now.hour + now.minute / 60))
    results: list[SurplusItem] = []

    for sku, current in stock.items():
        if sku not in BY_SKU or current <= 0:
            continue
        item: Item = BY_SKU[sku]
        day_forecast = float(daily_forecast.get(sku, 0.0))

        # How much of today's demand is still to come?
        remaining_share = max(0.0, 1.0 - expected_sell_through(now.hour))
        expected_remaining = day_forecast * remaining_share

        projected_surplus = max(0.0, current - expected_remaining)
        if current <= 0:
            continue

        # Fraction of stock we expect to be stuck with, weighted by perishability.
        raw_risk = projected_surplus / current
        risk = raw_risk * item.spoilage_severity

        if risk < min_risk or projected_surplus < 1:
            continue

        discount = 0
        label = ""
        for threshold, pct, tier_label in DISCOUNT_TIERS:
            if raw_risk >= threshold:
                discount, label = pct, tier_label
                break

        # Never discount below marginal cost -- a sale that loses money is worse
        # than waste for a long-shelf-life item that can simply be sold tomorrow.
        floor_pct = int((1 - item.unit_cost / item.unit_price) * 100)
        discount = min(discount, max(floor_pct - 5, 0))

        if discount <= 0:
            continue

        results.append(SurplusItem(
            sku=sku,
            name_ar=item.name_ar,
            current_stock=int(current),
            expected_remaining_sales=round(expected_remaining, 1),
            projected_surplus=int(round(projected_surplus)),
            risk_score=round(min(risk, 1.0), 3),
            urgency="high" if risk > 0.6 else "medium" if risk > 0.3 else "low",
            suggested_discount_pct=discount,
            value_at_risk_egp=round(projected_surplus * item.unit_cost * item.spoilage_severity, 2),
            hours_to_close=round(hours_left, 1),
        ))

    return sorted(results, key=lambda r: r.value_at_risk_egp, reverse=True)
