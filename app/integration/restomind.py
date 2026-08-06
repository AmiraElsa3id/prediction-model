"""RestoMind integration bridge.

Speaks the RestoMind backend's language: it takes that system's own product/restaurant
shapes (`productId`, `restaurantId`, `category`, `price`, `freshnessWindow`) and returns
model output ready for the Admin/Stores UI -- without touching the backend, which is
still being built.

Because there is no `sales_transactions` history yet, this runs fully **rule-based**
(cold start): calendar sensitivities come from the product's category (market priors),
the demand level comes from the owner's estimate (`avgDailySales`) or a default. When
real sales exist, the same products flow into the trained model instead -- this bridge is
the day-one path, not a replacement for it.

Two outputs, matching the two screens:
  * production_plan  -> Admin: how much of each product to make on a date, and why.
  * surplus_offers   -> Stores: which products are at risk near closing, with a discount
                        and Egyptian-Arabic offer copy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from app.core.egypt_calendar import CALENDAR
from app.core.market_priors import category_priors
from app.core.surplus import SELL_THROUGH_CURVE, expected_sell_through
from app.marketing.copy import OfferService
from app.models.rule_based import rule_multiplier

# Every wall-clock quantity this bridge reasons about -- `close_hour`, the
# sell-through curve's hour index, the calendar's date -- is Cairo local time.
BUSINESS_TIMEZONE = ZoneInfo("Africa/Cairo")

# Fallback daily level when the owner gives no estimate for a product.
DEFAULT_DAILY_LEVEL = 40.0
# Cold-start interval width (same spirit as RuleBasedForecaster).
INTERVAL_LOW, INTERVAL_HIGH = 0.65, 1.40

# Map a RestoMind category name (Arabic or English, free text) to a market-priors
# category. Unknown -> "" which resolves to neutral priors. Keyword substring match.
# Substring keywords -> market-priors category. Order matters (first match wins), so the
# more specific categories (sweet, savoury) come before the generic bread/bakery bucket.
_CATEGORY_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("pastry", "معجن", "كرواسون", "دانش", "croissant", "فطاير"), "pastry"),
    (("cake", "كيك", "جاتوه", "تورت", "gateau", "gateaux"), "cake"),
    (("sweet", "حلو", "حلوي", "حلوى", "شرقي", "dessert", "كنافة", "بسبوسة", "قطايف"), "sweet"),
    (("savoury", "savory", "مالح", "فطير", "سندوت", "sandwich", "feteer", "بيتزا", "pizza"), "savoury"),
    (("dry", "بسكوت", "بيتي فور", "biscuit", "cookie", "كوكيز"), "dry"),
    (("seasonal", "موسم", "كحك", "kahk"), "seasonal"),
    # Bread / bakery goods last -- it is the broadest bucket. "مخبوز" covers مخبوزات.
    (("bread", "عيش", "خبز", "مخبوز", "بيكري", "bakery", "بان", "توست", "toast"), "bread"),
]

_offer_service = OfferService()

# Bumped whenever the bridge's prediction logic changes; stored in each prediction so
# results stay auditable and comparable across versions (maps to RestoMind's
# prediction.modelVersionId). Cold-start rule-based today; becomes the trained model
# once real sales history flows through /data/ingest.
MODEL_VERSION = "restomind-bridge/rule_based-v0.1"


def map_category(name: str | None) -> str:
    """Best-effort map of a free-text category to a market-priors category."""
    if not name:
        return ""
    low = name.strip().lower()
    for keywords, mapped in _CATEGORY_KEYWORDS:
        if any(k in low for k in keywords):
            return mapped
    return ""


def _spoilage_severity(freshness_window_days: float | None) -> float:
    """Fraction of a leftover unit's value lost, from the product's freshness window.

    Mirrors the catalogue's shelf-life tiers, but driven by RestoMind's own
    `freshnessWindow` field instead of our hardcoded values.
    """
    d = freshness_window_days or 1
    if d <= 1:
        return 1.0
    if d <= 3:
        return 0.5
    if d <= 7:
        return 0.25
    return 0.15


@dataclass
class ProductInput:
    """A RestoMind product, as the bridge needs it."""

    product_id: str
    title: str
    category: str | None = None
    price: float = 0.0
    freshness_window: float | None = None   # days; RestoMind Product.freshnessWindow
    avg_daily_sales: float | None = None     # owner estimate; used until real data exists


def _forecast_one(
    p: "ProductInput", feats: dict, level: float | None = None,
    researched_overrides: dict[str, float] | None = None,
) -> tuple[float, list[dict]]:
    """Core rule-based point estimate for one product on one day.

    Shared by the daily production plan and the weekly prediction so both always agree.
    `level` overrides the daily baseline when a value learned from real sales exists
    (see the multi-tenant registry); otherwise the owner's estimate / default is used.

    `researched_overrides` -- explicitly-APPROVED multipliers from the Tavily-grounded
    market research agent (app/agents/market_research.py), keyed by event
    (PriorsStore.approved_multipliers). Deliberately a plain dict, not a DB lookup done
    in here: this function stays a pure calculation, and only ever sees a multiplier
    that already passed the pending_review -> approved gate -- an unreviewed research
    result must never reach a live forecast (see priors_store.py's module docstring).
    """
    priors = dict(category_priors(map_category(p.category)))
    if researched_overrides:
        priors.update(researched_overrides)
    mult, factors = rule_multiplier(priors, feats)
    if level is not None:
        base = level
    elif p.avg_daily_sales is not None:
        # 0.0 is a real answer ("this product sells nothing"), not a missing value.
        base = p.avg_daily_sales
    else:
        base = DEFAULT_DAILY_LEVEL
    return base * mult, factors


def production_plan(
    restaurant_id: str, products: list[ProductInput], target_date: dt.date,
    researched_overrides_by_category: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """Per-product production recommendation for a restaurant on a date (rule-based).

    `researched_overrides_by_category` -- approved market-research multipliers, keyed
    by the product's RESOLVED category (map_category(p.category)), each value itself
    an {event: multiplier} dict. See _forecast_one's docstring.
    """
    feats = CALENDAR.features(target_date)
    out: list[dict] = []

    for p in products:
        overrides = (researched_overrides_by_category or {}).get(map_category(p.category))
        qty, factors = _forecast_one(p, feats, researched_overrides=overrides)
        out.append({
            "productId": p.product_id,
            "title": p.title,
            "date": target_date.isoformat(),
            "recommendedQty": int(round(max(qty, 0))),
            "lowerBound": int(round(max(qty * INTERVAL_LOW, 0))),
            "upperBound": int(round(qty * INTERVAL_HIGH)),
            "confidence": "low",          # cold start: rule-based, no history yet
            "source": "rule_based",
            "factors": factors,
        })
    return out


def predict_week(
    restaurant_id: str, product: ProductInput, week_start: dt.date,
    promotion_active: bool = False, level: float | None = None,
    mode: str = "rule_based", confidence: str = "low",
    researched_overrides: dict[str, float] | None = None,
) -> dict:
    """Weekly prediction shaped for RestoMind's `predictions` collection.

    Mirrors their `prediction` document (`predictedOrders` for a `targetWeek`, plus a
    `featuresUsed` snapshot for auditability). Deliberately thin and swappable: today it
    sums the 7-day rule-based forecast; when an item has real history the same call will
    route through the trained model instead, without changing this contract.

    Note vs. their Phase-5 feature list: theirs is autoregressive only (lags/rolling +
    promo) with NO calendar signal. This bridge adds the Egyptian calendar from the week
    date -- that is exactly the Ramadan/Eid awareness their feature contract is missing.
    """
    daily: list[dict] = []
    for i in range(7):
        d = week_start + dt.timedelta(days=i)
        feats = CALENDAR.features(d)
        qty, factors = _forecast_one(product, feats, level=level, researched_overrides=researched_overrides)
        rounded = int(round(max(qty, 0)))
        daily.append({
            "date": d.isoformat(),
            # `predictedQuantity` is the canonical name -- it matches RestoMind's
            # DailyBreakdownItem schema, which is what consumes this array.
            "predictedQuantity": rounded,
            # DEPRECATED alias, kept one release so existing clients do not break.
            "qty": rounded,
            "factors": factors,
        })
    # Sum the rounded daily values so the weekly total always reconciles with the breakdown.
    total = sum(day["predictedQuantity"] for day in daily)

    # Feature snapshot -- what actually fed the prediction, for auditability.
    week_feats = CALENDAR.features(week_start)
    features_used = {
        "modelVersion": MODEL_VERSION,
        "mode": mode,
        "baseDailyLevel": round(
            level if level is not None
            else (product.avg_daily_sales if product.avg_daily_sales is not None else DEFAULT_DAILY_LEVEL),
            2,
        ),
        "levelSource": "learned_from_sales" if level is not None else "owner_estimate",
        "categoryResolved": map_category(product.category) or "neutral",
        "promotionActive": promotion_active,
        "calendar": {
            "isRamadan": bool(week_feats["is_ramadan"]),
            "isEidFitr": bool(week_feats["is_eid_fitr"]),
            "isKahkWindow": bool(week_feats["is_kahk_window"]),
            "daysToEidFitr": (
                int(week_feats["days_to_eid_fitr"]) if week_feats["days_to_eid_fitr"] < 999 else None
            ),
        },
    }

    # Distinct calendar drivers across the week, strongest first.
    seen: dict[str, dict] = {}
    for day in daily:
        for f in day["factors"]:
            if f["factor"] not in seen or abs(f["impact_pct"]) > abs(seen[f["factor"]]["impact_pct"]):
                seen[f["factor"]] = f
    factors = sorted(seen.values(), key=lambda f: abs(f["impact_pct"]), reverse=True)

    return {
        "restaurantId": restaurant_id,
        "productId": product.product_id,
        "modelVersionId": MODEL_VERSION,
        "targetWeek": week_start.isoformat(),
        "predictedOrders": int(total),
        "confidence": confidence,
        "featuresUsed": features_used,
        "factors": factors,
        "dailyBreakdown": daily,
    }


@dataclass
class StockInput(ProductInput):
    """A product plus its current unsold stock, for surplus detection."""

    current_stock: int = 0


def to_business_time(now: dt.datetime) -> dt.datetime:
    """Reduce any instant to Cairo wall-clock time, as a naive datetime.

    `close_hour` is a *Cairo* wall-clock hour (22 = 10pm local), and so are the
    hour index of the sell-through curve and the date the Egyptian calendar is
    keyed on. The RestoMind backend sends an offset-aware UTC timestamp
    (`new Date().toISOString()`), which Pydantic faithfully parses as UTC -- so
    reading `.hour` off it compared a UTC hour against a Cairo one. In summer
    (UTC+3) that is a three-hour error in the wrong direction: at Cairo 22:30,
    actual closing time, `.hour` read 19, the curve still expected 9% more
    sell-through, and `hours_left` claimed 2.5 hours remained. The scan
    systematically under-flagged surplus at exactly the moment it exists to run.
    `.date()` was wrong too, between Cairo 00:00 and 03:00, which mis-keys the
    holiday/Ramadan calendar features.

    A naive datetime is taken to already be Cairo wall-clock and passed through
    unchanged -- there is no offset to reason about, and that is the shape the
    bridge's own callers and tests use.
    """
    if now.tzinfo is None:
        return now
    return now.astimezone(BUSINESS_TIMEZONE).replace(tzinfo=None)


def surplus_offers(
    restaurant_id: str, stock: list[StockInput], now: dt.datetime, close_hour: int = 22,
) -> list[dict]:
    """Near-closing surplus per product, with a discount and Egyptian-Arabic copy.

    Risk = the share of current stock we do not expect to sell before closing, weighted
    by perishability (from `freshnessWindow`). Offer copy comes from the same generator
    the marketing endpoint uses (LLM + template fallback).
    """
    # Normalise before ANY wall-clock read: `.hour`, `.minute` and `.date()` below
    # are all compared against Cairo-local quantities.
    now = to_business_time(now)
    hours_left = max(0.0, close_hour - (now.hour + now.minute / 60))
    remaining_share = max(0.0, 1.0 - expected_sell_through(now.hour))
    results: list[dict] = []

    for s in stock:
        if s.current_stock <= 0:
            continue
        priors = category_priors(map_category(s.category))
        mult, _ = rule_multiplier(priors, CALENDAR.features(now.date()))
        # `or` is a truthy test, so an honest 0.0 ("this product sells nothing")
        # was silently replaced by DEFAULT_DAILY_LEVEL (40/day) -- exactly the
        # dead-slow stock that most needs discounting was given a healthy
        # expected sell-through, driving projected_surplus to 0 and skipping it
        # below. `_forecast_one` already distinguishes the two; match it.
        base_level = (
            s.avg_daily_sales if s.avg_daily_sales is not None else DEFAULT_DAILY_LEVEL
        )
        day_level = base_level * mult
        expected_remaining = day_level * remaining_share

        projected_surplus = max(0.0, s.current_stock - expected_remaining)
        raw_risk = projected_surplus / s.current_stock
        severity = _spoilage_severity(s.freshness_window)
        risk = raw_risk * severity
        if risk < 0.15 or projected_surplus < 1:
            continue

        # Discount tiers by raw risk, floored to stay above marginal cost is not possible
        # without cost -- so cap at a sane maximum instead.
        discount = 40 if raw_risk >= 0.8 else 30 if raw_risk >= 0.6 else 20 if raw_risk >= 0.4 else 10

        offer = _offer_service.build_freeform(
            title_ar=s.title, price=s.price, discount_pct=discount,
        ) if s.price > 0 else None

        results.append({
            "productId": s.product_id,
            "title": s.title,
            "currentStock": s.current_stock,
            "projectedSurplus": int(round(projected_surplus)),
            "riskScore": round(min(risk, 1.0), 3),
            "urgency": "high" if risk > 0.6 else "medium" if risk > 0.3 else "low",
            "hoursToClose": round(hours_left, 1),
            "suggestedDiscountPct": discount,
            "offerCopyAr": offer["copy_ar"] if offer else None,
            "newPrice": offer["new_price"] if offer else None,
        })

    return sorted(results, key=lambda r: r["riskScore"], reverse=True)
