"""RestoMind integration bridge.

Speaks the RestoMind backend's language: it takes that system's own product/restaurant
shapes (`productId`, `restaurantId`, `category`, `price`, `freshnessWindow`) and returns
model output ready for the Admin/Stores UI -- without touching the backend, which is
still being built.

The rule-based cold-start layer (market priors + calendar multipliers) has been
REMOVED (see HANDOFF.md §8). The bridge no longer invents a quantity from hand-written
priors. It forecasts from a **basis daily level** only:
  * a level learned from this restaurant's REAL sales via /integration/restomind/ingest
    (best - it came from actual history), or
  * the owner's `avgDailySales` estimate.

That level is a quiet-day mean with seasonality deliberately left out, so on its own it
answers every date with the same number -- which is what the production plan did for
every product outside the trained catalogue. A learned level now carries a learned
calendar with it (`tenant_calendar.TenantCalendar`, fitted from the same restaurant's
history) and the plan multiplies the two. An owner estimate has no history behind it and
therefore no calendar: it stays flat, because the alternative is guessing.

A product with neither has no forecast at all: the bridge answers with a "still
training" message so the caller can plan around the absence instead of trusting a guess.

Two outputs, matching the two screens:
  * production_plan  -> Admin: how much of each product to make on a date, and why.
  * surplus_offers   -> Stores: which products are at risk near closing, with a discount
                        and Egyptian-Arabic offer copy.

Trained-model routing: a product that carries a catalogue `sku` link (RestoMind
`Product.sku`, e.g. `PASTRY_CROISSANT`) is forecast by the trained CalendarDecomposed
model when the service is passed in -- so the admin production plan and the weekly
prediction finally move with the calendar (Ramadan/Eid) instead of repeating a flat
level. A product with no SKU, or a SKU the model cannot serve (unknown / below the
training threshold), falls back to the basis level below.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from app.core.egypt_calendar import CALENDAR
from app.core.items import spoilage_severity_for_shelf_life
from app.core.surplus import expected_sell_through
from app.models.service import ModelNotReadyError

if TYPE_CHECKING:
    from app.integration.tenant_calendar import TenantCalendar
    from app.models.service import ForecastService

# Every wall-clock quantity this bridge reasons about -- `close_hour`, the
# sell-through curve's hour index, the calendar's date -- is Cairo local time.
BUSINESS_TIMEZONE = ZoneInfo("Africa/Cairo")

# Bumped whenever the bridge's prediction logic changes; stored in each prediction so
# results stay auditable and comparable across versions (maps to RestoMind's
# prediction.modelVersionId). Used for the basis-level fallback path; the trained path
# stamps `calendar_decomposed/{source}` instead.
#
# v0.2: a learned level is multiplied by the restaurant's own learned calendar, so the
# same product no longer gets one number for every date of the year.
MODEL_VERSION = "restomind-bridge/basis-v0.2"

# Output margin when the bridge CAN forecast: a fixed band around the basis level.
# There is no trained interval on this path (that needs the trained model), so the
# bounds are deliberately modest rather than implying a calibrated uncertainty.
BASIS_QUANTUM_LOW, BASIS_QUANTUM_HIGH = 0.9, 1.1

# Placed on every response where a product cannot be forecast because it has no
# basis level (no learnt level and no owner estimate).
TRAINING_MESSAGE = (
    "Still training: no basis available for this product yet (no learned demand "
    "level from real sales and no owner estimate). Post history via "
    "/integration/restomind/ingest or supply avgDailySales to start forecasting."
)


def _spoilage_severity(freshness_window_days: float | None) -> float:
    """Fraction of a leftover unit's value lost, from the product's freshness window.

    Driven by RestoMind's own `freshnessWindow` field rather than the catalogue's
    hardcoded shelf life; the tier table itself is shared with the catalogue via
    `spoilage_severity_for_shelf_life` so the two definitions cannot drift apart.
    """
    return spoilage_severity_for_shelf_life(freshness_window_days)


# Default goodwill lost on a stockout, as a multiple of margin, when a restaurant has
# not told us its own figure. Mirrors `Item.stockout_goodwill_mult`'s default in
# app/core/items.py -- neutral middle ground until real per-product economics exist.
_DEFAULT_STOCKOUT_GOODWILL_MULT = 0.5


def _newsvendor_quantile(
    unit_cost: float | None, price: float, severity: float,
) -> float | None:
    """Profit-optimal service level q* = Cu / (Cu + Co), from REAL restaurant economics.

    `None` when `unit_cost` is unknown -- the caller falls back to the fixed
    INTERVAL_LOW/HIGH band rather than pretending to know a number it doesn't. This is
    the same formula as `Item.newsvendor_quantile` (app/core/items.py), just fed from
    a restaurant's own `unitCost`/`price`/`freshnessWindow` instead of the hardcoded
    11-item catalogue -- replacing that placeholder is the whole point of asking
    RestoMind for real unit economics.

    A cost of ZERO is treated as unknown, not as free. Arithmetically a zero cost means
    zero overage cost, so q* is exactly 1.0 and every such product is planned at its
    upper bound -- a permanent, invisible +10% on quantity. Backends that fill unset
    numeric columns with 0 (RestoMind's registry is full of `unitCost: 0` against real
    prices) were getting that silently. Nothing in a bakery costs nothing to make.
    """
    if unit_cost is None or unit_cost <= 0 or price <= 0:
        return None
    margin = price - unit_cost
    if margin <= 0:
        return 0.5
    cu = margin * (1.0 + _DEFAULT_STOCKOUT_GOODWILL_MULT)
    co = unit_cost * severity
    if cu + co <= 0:
        return 0.5
    return round(cu / (cu + co), 4)


@dataclass
class ProductInput:
    """A RestoMind product, as the bridge needs it."""

    product_id: str
    title: str
    category: str | None = None
    price: float = 0.0
    unit_cost: float | None = None          # EGP; RestoMind Product.unitCost, if known
    freshness_window: float | None = None   # days; RestoMind Product.freshnessWindow
    avg_daily_sales: float | None = None     # owner estimate; becomes the basis if no learned level
    sku: str | None = None                   # catalogue link; when present and trained, the
                                             # forecast uses the trained model instead of the basis
    # Retained for API compatibility with RestoMind payloads. Now that the bridge runs
    # basis levels and applies no calendar multiplier, windows are informational only.
    avg_daily_sales_window: tuple[dt.date, dt.date] | None = None


def _trained_day(model: "ForecastService", sku: str, target_date: dt.date):
    """One trained per-day forecast, or `None` when the SKU is unusable by the model.

    Unknown SKU (`KeyError`) and below-the-threshold products (`ModelNotReadyError`)
    both mean "this product has no trained model right now" -- the caller falls back to
    the basis level instead of failing the whole request.
    """
    try:
        return model.forecast(sku, target_date)
    except (KeyError, ModelNotReadyError):
        return None


def _trained_week(
    restaurant_id: str, product: ProductInput, week_start: dt.date,
    days: list, promotion_active: bool,
) -> dict:
    """Shape a trained 7-day run into RestoMind's prediction document.

    Mirrors `connect_restomind._predict_trained`: per-day quantities plus a `featuresUsed`
    snapshot naming the trained model, and the strongest distinct calendar drivers as the
    week's `factors` -- so the audit trail explains the number instead of showing [].
    """
    daily = [
        {"date": d.date.isoformat(),
         "predictedQuantity": int(d.quantity),
         "qty": int(d.quantity),
         "factors": d.factors}
        for d in days
    ]
    total = sum(day["predictedQuantity"] for day in daily)

    seen: dict[str, dict] = {}
    for d in days:
        for f in d.factors:
            if f["factor"] not in seen or abs(f["impact_pct"]) > abs(seen[f["factor"]]["impact_pct"]):
                seen[f["factor"]] = f
    factors = sorted(seen.values(), key=lambda f: abs(f["impact_pct"]), reverse=True)

    model_version = f"calendar_decomposed/{days[0].source}"
    return {
        "restaurantId": restaurant_id,
        "productId": product.product_id,
        "modelVersionId": model_version,
        "targetWeek": week_start.isoformat(),
        "predictedOrders": int(total),
        "confidence": days[0].confidence,
        "trainingMessage": None,
        "featuresUsed": {
            "sku": product.sku,
            "modelVersion": model_version,
            "mode": "calendar_decomposed",
            "levelSource": "trained_model",
            "promotionActive": promotion_active,
            "calendar": _calendar_snapshot(week_start),
        },
        "factors": factors,
        "dailyBreakdown": daily,
    }


def basis_level(p: ProductInput, level: float | None = None) -> float | None:
    """The daily level to plan from: learned (best) else owner estimate.

    Never falls back to a made-up default: with no learned level and no owner
    estimate there is no basis for a forecast and `None` is returned (the caller
    reports "still training").
    """
    if level is not None:
        return float(level)
    if p.avg_daily_sales is not None:
        # 0.0 is a real answer ("this product sells nothing"), not a missing value.
        return float(p.avg_daily_sales)
    return None


def _calendar_for(
    calendar: "TenantCalendar | None", product_id: str, day: dt.date,
) -> tuple[float, list[dict]]:
    """`(multiplier, factors)` for one product on one day, or a neutral `(1.0, [])`.

    Shared by the production plan, the weekly prediction and the surplus scan so the
    three screens cannot disagree about what today does to demand. A product the
    restaurant has not yet earned a calendar for gets exactly the old flat behaviour.
    """
    if calendar is None or product_id not in calendar:
        return 1.0, []
    return calendar.multiplier(product_id, day), calendar.explain(product_id, day)


def production_plan(
    restaurant_id: str, products: list[ProductInput], target_date: dt.date,
    levels: dict[str, tuple[float | None, str, str]] | None = None,
    model: "ForecastService | None" = None,
    calendar: "TenantCalendar | None" = None,
) -> list[dict]:
    """Per-product production recommendation for a restaurant on a date.

    `levels` maps productId -> (learned_level, mode, confidence) from the registry.
    Where a product has a level learned from that restaurant's real sales, it is used
    in place of the owner's `avgDailySales` estimate. A product with no learned level
    and no owner estimate has NO recommendation: the response carries a "still
    training" message instead of a quantity made up from priors.

    `model`, when given, upgrades products that carry a trained catalogue `sku` link:
    they are forecast by the trained CalendarDecomposed model for `target_date`
    (calendar-aware, with real confidence and factors). Everything else keeps the
    basis-level behaviour below. Passing no model (or a product with no SKU) is
    unchanged behaviour, so tests and callers that only use the stateless bridge are
    unaffected.

    `calendar`, when given, is the restaurant's OWN learned calendar (see
    `tenant_calendar`). It moves the basis level onto `target_date` for products that
    have enough history to have earned one, which is what stops this endpoint answering
    every date of the year with a single number. Products on the owner's estimate are
    not in it and stay flat.
    """
    levels = levels or {}
    out: list[dict] = []

    for p in products:
        if p.sku and model is not None:
            trained = _trained_day(model, p.sku, target_date)
            if trained is not None:
                out.append({
                    "productId": p.product_id,
                    "title": p.title,
                    "date": target_date.isoformat(),
                    "recommendedQty": trained.quantity,
                    "lowerBound": trained.lower,
                    "upperBound": trained.upper,
                    "confidence": trained.confidence,
                    "source": trained.source,
                    "levelSource": "trained_model",
                    "baseDailyLevel": round(float(trained.quantity), 2),
                    "factors": trained.factors,
                    "trainingMessage": None,
                })
                continue
        level, mode, confidence = levels.get(p.product_id, (None, "training", "low"))
        base = basis_level(p, level)
        if base is None:
            out.append({
                "productId": p.product_id,
                "title": p.title,
                "date": target_date.isoformat(),
                "recommendedQty": 0,
                "lowerBound": 0,
                "upperBound": 0,
                "confidence": "low",
                "source": "training",
                "levelSource": "none",
                "baseDailyLevel": 0.0,
                "factors": [],
                "trainingMessage": TRAINING_MESSAGE,
            })
            continue

        # The level is an ORDINARY-day figure by construction. This is what puts the
        # target date back into it -- without it `target_date` never entered the
        # arithmetic at all and the plan was the same on a Tuesday and in Ramadan.
        multiplier, factors = _calendar_for(calendar, p.product_id, target_date)
        planned = base * multiplier

        lower = max(planned * BASIS_QUANTUM_LOW, 0)
        upper = max(planned * BASIS_QUANTUM_HIGH, 0)
        # With real unit_cost + price, pick the profit-optimal point within the
        # uncertainty band instead of always the raw point estimate: a thin-margin,
        # short-shelf-life product (q* low) should be produced nearer the lower bound
        # to avoid waste, a high-margin/long-shelf-life one (q* high) nearer the upper
        # bound to avoid stockouts. Unknown economics keeps the old behaviour exactly
        # (`q is None` -> the point estimate, same as before this existed).
        severity = _spoilage_severity(p.freshness_window)
        q = _newsvendor_quantile(p.unit_cost, p.price, severity)
        recommended = planned if q is None else lower + q * (upper - lower)
        out.append({
            "productId": p.product_id,
            "title": p.title,
            "date": target_date.isoformat(),
            "recommendedQty": int(round(max(recommended, 0))),
            "lowerBound": int(round(lower)),
            "upperBound": int(round(upper)),
            "confidence": confidence,
            "source": mode,
            "levelSource": "learned_from_sales" if level is not None else "owner_estimate",
            # The ORDINARY-day level, before the calendar. Reported separately from
            # `calendarMultiplier` so a reader can see both halves of the quantity.
            "baseDailyLevel": round(float(base), 2),
            "calendarMultiplier": round(multiplier, 3),
            "factors": factors,
            "trainingMessage": None,
        })
    return out


def predict_week(
    restaurant_id: str, product: ProductInput, week_start: dt.date,
    promotion_active: bool = False, level: float | None = None,
    mode: str = "training", confidence: str = "low",
    model: "ForecastService | None" = None,
    calendar: "TenantCalendar | None" = None,
) -> dict:
    """Weekly prediction shaped for RestoMind's `predictions` collection.

    Mirrors their `prediction` document (`predictedOrders` for a `targetWeek`, plus a
    `featuresUsed` snapshot for auditability). Forecast is the basis level shaped across
    the week by the restaurant's own learned calendar when it has one (`calendar`);
    without one the level repeats, as it did before -- an owner's estimate has no
    history behind it to learn a weekday profile from.

    `model`, when given, upgrades a product carrying a trained catalogue `sku` link to
    the trained CalendarDecomposed model: seven calendar-aware per-day forecasts whose
    sum is `predictedOrders`, branded `calendar_decomposed/*` (this is exactly what
    `connect_restomind` already did for SKU products, now inside the bridge so one call
    serves both paths). Products without a usable trained SKU keep the basis path below.

    A product with no basis returns `predictedOrders: 0` and a conspicuous
    `trainingMessage` -- still training, do not order based on this.
    """
    if product.sku and model is not None:
        try:
            days = model.forecast_week(product.sku, week_start)
        except (KeyError, ModelNotReadyError):
            days = None
        if days is not None:
            return _trained_week(
                restaurant_id, product, week_start, days, promotion_active
            )
    base = basis_level(product, level)
    if base is None:
        features_used = {
            "basisProvided": None,
            "levelSource": "none",
            "promotionActive": promotion_active,
            "calendar": _calendar_snapshot(week_start),
        }
        return {
            "restaurantId": restaurant_id,
            "productId": product.product_id,
            "modelVersionId": MODEL_VERSION,
            "targetWeek": week_start.isoformat(),
            "predictedOrders": 0,
            "confidence": "low",
            "trainingMessage": TRAINING_MESSAGE,
            "featuresUsed": features_used,
            "factors": [],
            "dailyBreakdown": [
                {"date": (week_start + dt.timedelta(days=i)).isoformat(),
                 "predictedQuantity": 0, "qty": 0, "factors": []}
                for i in range(7)
            ],
        }

    daily: list[dict] = []
    strongest: dict[str, dict] = {}
    for i in range(7):
        d = week_start + dt.timedelta(days=i)
        multiplier, factors = _calendar_for(calendar, product.product_id, d)
        rounded = int(round(max(base * multiplier, 0)))
        daily.append({
            "date": d.isoformat(),
            # `predictedQuantity` is the canonical name -- it matches RestoMind's
            # DailyBreakdownItem schema, which is what consumes this array.
            "predictedQuantity": rounded,
            # DEPRECATED alias, kept one release so existing clients do not break.
            "qty": rounded,
            "factors": factors,
        })
        # The week's headline factors are each driver at its strongest day, matching
        # how `_trained_week` summarises the trained path.
        for f in factors:
            if f["factor"] not in strongest or abs(f["impact_pct"]) > abs(strongest[f["factor"]]["impact_pct"]):
                strongest[f["factor"]] = f
    # Sum the rounded daily values so the weekly total always reconciles with the breakdown.
    total = sum(day["predictedQuantity"] for day in daily)

    # Feature snapshot -- what actually fed the prediction, for auditability.
    features_used = {
        "modelVersion": MODEL_VERSION,
        "mode": mode,
        "baseDailyLevel": round(float(base), 2),
        "levelSource": "learned_from_sales" if level is not None else "owner_estimate",
        # Whether the week has a shape at all, or is one number seven times.
        "calendarSource": (
            "learned_from_sales"
            if calendar is not None and product.product_id in calendar
            else "none"
        ),
        "promotionActive": promotion_active,
        "calendar": _calendar_snapshot(week_start),
    }

    return {
        "restaurantId": restaurant_id,
        "productId": product.product_id,
        "modelVersionId": MODEL_VERSION,
        "targetWeek": week_start.isoformat(),
        "predictedOrders": int(total),
        "confidence": confidence,
        "trainingMessage": None,
        "featuresUsed": features_used,
        "factors": sorted(strongest.values(), key=lambda f: abs(f["impact_pct"]), reverse=True),
        "dailyBreakdown": daily,
    }


def _calendar_snapshot(day: dt.date) -> dict:
    """Calendar state for the week's start date, mirroring it into the audit trail."""
    feats = CALENDAR.features(day)
    return {
        "isRamadan": bool(feats["is_ramadan"]),
        "isEidFitr": bool(feats["is_eid_fitr"]),
        "isKahkWindow": bool(feats["is_kahk_window"]),
        "daysToEidFitr": (
            int(feats["days_to_eid_fitr"]) if feats["days_to_eid_fitr"] < 999 else None
        ),
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
    levels: dict[str, tuple[float | None, str, str]] | None = None,
    calendar: "TenantCalendar | None" = None,
) -> list[dict]:
    """Near-closing surplus per product, with a suggested discount.

    Risk = the share of current stock we do not expect to sell before closing, weighted
    by perishability (from `freshnessWindow`).

    `levels` maps productId -> (learned_level, mode, confidence) from the registry. A
    learned level replaces the owner's estimate when computing expected sell-through,
    which is what decides whether stock is at risk at all -- an owner estimate or
    learned level is the basis for expected sales. A product with no basis is skipped
    (no foundation to judge risk).

    `calendar` shapes that basis onto today, same as the production plan: a Friday's
    stock is judged against a Friday's demand. Flagging surplus against a flat weekday
    average is how you end up discounting on the busiest day of the week.
    """
    levels = levels or {}
    # Normalise before ANY wall-clock read: `.hour`, `.minute` and `.date()` below
    # are all compared against Cairo-local quantities.
    now = to_business_time(now)
    hours_left = max(0.0, close_hour - (now.hour + now.minute / 60))
    remaining_share = max(0.0, 1.0 - expected_sell_through(now.hour))
    results: list[dict] = []

    for s in stock:
        if s.current_stock <= 0:
            continue
        learned_level, _, _ = levels.get(s.product_id, (None, "training", "low"))
        base = basis_level(s, level=learned_level)
        if base is None:
            # No basis -> no expected sell-through -> risk cannot be judged. Skip.
            continue
        # `or` is a truthy test, so an honest 0.0 ("this product sells nothing")
        # must not be replaced by anything. `basis_level` keeps 0.0 distinct from a
        # missing value, and `base` is used exactly as given.
        multiplier, _ = _calendar_for(calendar, s.product_id, now.date())
        expected_remaining = base * multiplier * remaining_share

        projected_surplus = max(0.0, s.current_stock - expected_remaining)
        raw_risk = projected_surplus / s.current_stock
        severity = _spoilage_severity(s.freshness_window)
        risk = raw_risk * severity
        if risk < 0.15 or projected_surplus < 1:
            continue

        # Discount tiers by raw risk, capped at a sane maximum.
        discount = 40 if raw_risk >= 0.8 else 30 if raw_risk >= 0.6 else 20 if raw_risk >= 0.4 else 10
        # Floored to stay above marginal cost, now that real unit_cost lets us: selling
        # below cost to avoid waste just trades one loss for a bigger one. Unknown cost
        # keeps the old behaviour (cap-only, no floor).
        if s.unit_cost is not None and s.price > 0:
            max_discount_pct = max(0, int((1 - s.unit_cost / s.price) * 100))
            discount = min(discount, max_discount_pct)

        results.append({
            "productId": s.product_id,
            "title": s.title,
            "currentStock": s.current_stock,
            "projectedSurplus": int(round(projected_surplus)),
            "riskScore": round(min(risk, 1.0), 3),
            "urgency": "high" if risk > 0.6 else "medium" if risk > 0.3 else "low",
            "hoursToClose": round(hours_left, 1),
            "suggestedDiscountPct": discount,
        })

    return sorted(results, key=lambda r: r["riskScore"], reverse=True)
