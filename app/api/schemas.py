"""Pydantic request/response schemas.

Written for integration into the existing e-commerce system: everything is plain
structured JSON with explicit types, no files, no bespoke encodings. Field
descriptions carry through to Swagger so the backend team can integrate from /docs.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator

from app.core.items import BY_SKU


class SKUMixin(BaseModel):
    """Shared SKU validation so an unknown item fails at the edge with a clear message."""

    @field_validator("sku", check_fields=False)
    @classmethod
    def _known_sku(cls, v: str) -> str:
        if v not in BY_SKU:
            raise ValueError(f"unknown SKU '{v}'. Known: {', '.join(sorted(BY_SKU))}")
        return v


# -- forecasting ---------------------------------------------------------------------


class Factor(BaseModel):
    factor: str = Field(..., description="Calendar driver, e.g. 'Ramadan'")
    impact_pct: float = Field(..., description="Percent change attributed to this driver")
    direction: str = Field(..., description="'increase' or 'decrease'")


class DailyForecastRequest(SKUMixin):
    sku: str = Field(..., description="Item SKU", examples=["PASTRY_CROISSANT"])
    date: dt.date = Field(..., description="Date to forecast production for")


class ForecastResponse(BaseModel):
    sku: str
    date: dt.date
    recommended_quantity: int = Field(..., description="Units to produce")
    lower_bound: int = Field(..., description="10th percentile of predicted demand")
    upper_bound: int = Field(..., description="90th percentile of predicted demand")
    confidence: str = Field(..., description="'high' | 'medium' | 'low'")
    source: str = Field(..., description="'batch'")
    factors: list[Factor] = Field(
        default_factory=list, description="Why the number differs from a normal day"
    )


class WeeklyForecastRequest(SKUMixin):
    sku: str = Field(..., examples=["CAKE_GATEAU"])
    start_date: dt.date = Field(..., description="First day of the 7-day horizon")


class WeeklyForecastResponse(BaseModel):
    sku: str
    start_date: dt.date
    total_quantity: int
    days: list[ForecastResponse]


# -- batch forecasting (all items at once) -------------------------------------------


class DailyBatchRequest(BaseModel):
    date: dt.date = Field(..., description="Date to forecast production for")
    skus: list[str] | None = Field(
        None,
        description="Items to forecast. Omit or null to forecast every known item.",
        examples=[["PASTRY_CROISSANT", "SWEET_KONAFA"]],
    )

    @field_validator("skus")
    @classmethod
    def _known_skus(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = set(v) - set(BY_SKU)
        if unknown:
            raise ValueError(f"unknown SKUs: {', '.join(sorted(unknown))}")
        return v


class DailyBatchResponse(BaseModel):
    date: dt.date
    item_count: int = Field(..., description="Number of items forecast")
    total_quantity: int = Field(..., description="Sum of recommended quantities")
    items: list[ForecastResponse] = Field(..., description="One forecast per item")


class WeeklyBatchRequest(BaseModel):
    start_date: dt.date = Field(..., description="First day of the 7-day horizon")
    skus: list[str] | None = Field(
        None, description="Items to forecast. Omit or null for every known item."
    )

    @field_validator("skus")
    @classmethod
    def _known_skus(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = set(v) - set(BY_SKU)
        if unknown:
            raise ValueError(f"unknown SKUs: {', '.join(sorted(unknown))}")
        return v


class WeeklyBatchResponse(BaseModel):
    start_date: dt.date
    item_count: int
    items: list[WeeklyForecastResponse] = Field(
        ..., description="A 7-day plan per item"
    )


# -- data ingestion & model status ---------------------------------------------------


class SalesRecord(BaseModel):
    """One end-of-day actual for one item at one branch."""

    date: dt.date = Field(..., description="Business day")
    sku: str = Field(..., description="Item SKU")
    sales_qty: int = Field(..., ge=0, description="Units actually sold")
    production_qty: int = Field(..., ge=0, description="Units produced that day")
    closing_stock: int = Field(0, ge=0, description="Unsold units at close (leftover)")

    @field_validator("sku")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in BY_SKU:
            raise ValueError(f"unknown SKU '{v}'")
        return v


class IngestRequest(BaseModel):
    records: list[SalesRecord] = Field(
        ..., min_length=1,
        description="End-of-day actuals. The backend posts these each night so history "
                    "accumulates and items promote from untrained to the trained model.",
    )


class IngestResponse(BaseModel):
    rows_ingested: int
    model_retrained: bool = Field(..., description="Did an item cross the training threshold?")
    newly_switched_to_ml: list[str] = Field(
        default_factory=list, description="Items that moved to the trained model this call"
    )
    total_days_by_item: dict[str, int]


class ItemModeStatus(BaseModel):
    sku: str
    mode: str = Field(..., description="'untrained' or 'trained_model'")
    observed_days: int
    days_until_switch: int = Field(..., description="Days of data still needed; 0 if trained")
    progress: float = Field(..., description="observed_days / threshold, capped at 1.0")


class ModelStatusResponse(BaseModel):
    train_threshold_days: int
    model_trained_at: str | None
    items_untrained: int
    items_trained: int
    items: list[ItemModeStatus]


# -- seasonality ---------------------------------------------------------------------


class SeasonalityRequest(SKUMixin):
    sku: str = Field(..., examples=["SWEET_KONAFA"])
    date: dt.date


class CalendarContext(BaseModel):
    is_ramadan: bool
    ramadan_day: int | None = None
    holiday: str | None = None
    is_weekend: bool
    is_school_term: bool
    days_to_eid_fitr: int | None = None


class SeasonalityResponse(BaseModel):
    sku: str
    date: dt.date
    baseline_quantity: int = Field(..., description="Prediction with all events neutralised")
    adjusted_quantity: int = Field(..., description="Prediction for the real date")
    multiplier: float = Field(..., description="adjusted / baseline")
    factors: list[Factor]
    calendar: CalendarContext


# -- waste prevention ----------------------------------------------------------------


class WasteAlertRequest(SKUMixin):
    sku: str = Field(..., examples=["PASTRY_CROISSANT"])
    date: dt.date
    planned_quantity: int = Field(..., ge=0, description="Quantity the manager entered")


class WasteAlertResponse(BaseModel):
    sku: str
    date: dt.date
    planned_qty: int
    forecast_qty: int
    forecast_upper: int
    excess_qty: int
    severity: str = Field(..., description="'none' | 'low' | 'medium' | 'high'")
    message: str
    projected_waste_cost_egp: float
    factors: list[Factor]


# -- surplus -------------------------------------------------------------------------


class SurplusRequest(BaseModel):
    stock: dict[str, int] = Field(
        ..., description="Current unsold stock by SKU",
        examples=[{"CAKE_GATEAU": 40, "PASTRY_CROISSANT": 25}],
    )
    timestamp: dt.datetime | None = Field(
        None, description="Time of check; defaults to now. Meant to run near closing."
    )
    close_hour: int = Field(22, ge=0, le=23, description="Branch closing hour (24h)")

    @field_validator("stock")
    @classmethod
    def _known_skus(cls, v: dict[str, int]) -> dict[str, int]:
        unknown = set(v) - set(BY_SKU)
        if unknown:
            raise ValueError(f"unknown SKUs: {', '.join(sorted(unknown))}")
        return v


class SurplusItemResponse(BaseModel):
    sku: str
    item_name_ar: str
    current_stock: int
    expected_remaining_sales: float
    projected_surplus: int
    risk_score: float = Field(..., description="0-1, perishability-weighted")
    urgency: str
    suggested_discount_pct: int
    value_at_risk_egp: float
    hours_to_close: float


class SurplusResponse(BaseModel):
    checked_at: dt.datetime
    items_at_risk: list[SurplusItemResponse]
    total_value_at_risk_egp: float


# -- marketing -----------------------------------------------------------------------


class OfferRequest(SKUMixin):
    sku: str = Field(..., examples=["CAKE_GATEAU"])
    discount_pct: int = Field(..., ge=5, le=70, description="Discount percentage")
    close_time: str = Field("10 بالليل", description="Human-readable offer deadline")


class OfferResponse(BaseModel):
    sku: str
    item_name_ar: str
    discount_pct: int
    old_price: float
    new_price: float
    copy_ar: str = Field(..., description="Promotional copy in Egyptian Arabic")
    valid_until: str
    generator: str = Field(..., description="'llm' or 'template' -- which produced the copy")
    hashtags: list[str]


class PublishRequest(BaseModel):
    sku: str
    copy_ar: str = Field(..., description="Copy to publish")
    platforms: list[str] = Field(
        default_factory=lambda: ["facebook"],
        description="Target platforms: 'facebook' and/or 'instagram'",
    )
    dry_run: bool = Field(
        True,
        description=(
            "When true (default) returns a preview without contacting Meta. "
            "Live publishing posts to a real public page and requires explicit opt-in."
        ),
    )

    @field_validator("platforms")
    @classmethod
    def _known_platforms(cls, v: list[str]) -> list[str]:
        allowed = {"facebook", "instagram"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"unsupported platforms: {', '.join(sorted(bad))}")
        return v


class PublishResponse(BaseModel):
    status: str = Field(..., description="'preview' | 'published' | 'failed'")
    dry_run: bool
    platforms: list[str]
    preview: dict = Field(default_factory=dict, description="Rendered post per platform")
    post_ids: dict = Field(default_factory=dict, description="Live post IDs, when published")
    message: str


# -- RestoMind integration bridge ----------------------------------------------------


class DateWindow(BaseModel):
    """An inclusive date range a measured average was taken over."""

    from_: dt.date = Field(..., alias="from")
    to: dt.date

    model_config = {"populate_by_name": True}


class RMProduct(BaseModel):
    """A RestoMind product, as the bridge needs it (mirrors their Product model)."""

    productId: str = Field(..., description="RestoMind Product _id")
    title: str
    category: str | None = Field(None, description="Category name (Arabic/English free text)")
    price: float = Field(0.0, ge=0)
    freshnessWindow: float | None = Field(
        None, description="Shelf life in days (RestoMind Product.freshnessWindow)"
    )
    avgDailySales: float | None = Field(
        None, description="Typical daily sales. Must be an ORDINARY-day figure unless "
                          "avgDailySalesWindow says otherwise."
    )
    avgDailySalesWindow: DateWindow | None = Field(
        None,
        description=(
            "Set ONLY when avgDailySales is a mean measured over real days, giving the "
            "window it was measured over. Informational: since the rule-based calendar "
            "multipliers were removed, no de-seasonalising is applied."
        ),
    )
    sku: str | None = Field(
        None, description="Catalogue SKU link (e.g. PASTRY_CROISSANT). When present and "
                          "the item is trained, the production plan and predictions use "
                          "the trained CalendarDecomposed model (calendar-aware) instead "
                          "of the basis level."
    )


class RMProductionPlanRequest(BaseModel):
    restaurantId: str
    date: dt.date
    products: list[RMProduct] = Field(..., min_length=1)


class RMPlanItem(BaseModel):
    productId: str
    title: str
    date: str
    recommendedQty: int
    lowerBound: int
    upperBound: int
    confidence: str
    source: str
    # Whether this quantity came from the restaurant's own sales or the owner's
    # estimate. Without it a plan cannot be told apart from a guess.
    levelSource: str = "owner_estimate"
    baseDailyLevel: float = 0.0
    factors: list[Factor]
    # Present when the product has NO basis (no learned level, no owner estimate): the
    # bridge says so explicitly instead of inventing a quantity from priors.
    trainingMessage: str | None = None


class RMProductionPlanResponse(BaseModel):
    restaurantId: str
    date: dt.date
    totalRecommendedQty: int
    items: list[RMPlanItem]


class RMStockProduct(RMProduct):
    currentStock: int = Field(..., ge=0, description="Unsold units on hand right now")


class RMSurplusRequest(BaseModel):
    restaurantId: str
    stock: list[RMStockProduct] = Field(..., min_length=1)
    timestamp: dt.datetime | None = Field(None, description="Defaults to now; meant near closing")
    closeHour: int = Field(22, ge=0, le=23)


class RMSurplusItem(BaseModel):
    productId: str
    title: str
    currentStock: int
    projectedSurplus: int
    riskScore: float
    urgency: str
    hoursToClose: float
    suggestedDiscountPct: int
    offerCopyAr: str | None = None
    newPrice: float | None = None


class RMSurplusResponse(BaseModel):
    restaurantId: str
    checkedAt: dt.datetime
    itemsAtRisk: list[RMSurplusItem]


class RMPredictRequest(BaseModel):
    """Weekly prediction request, shaped to feed RestoMind's `predictions` collection."""

    restaurantId: str
    productId: str
    title: str
    targetWeek: dt.date = Field(..., description="First day of the target week (YYYY-MM-DD)")
    category: str | None = Field(None, description="Category name (Arabic/English free text)")
    avgDailySales: float | None = Field(
        None, description="Typical daily sales. Must be an ORDINARY-day figure unless "
                          "avgDailySalesWindow says otherwise."
    )
    avgDailySalesWindow: DateWindow | None = Field(
        None,
        description=(
            "Set ONLY when avgDailySales is a mean measured over real days. "
            "Informational: the rule-based calendar multipliers are gone, so no "
            "de-seasonalising is applied."
        ),
    )
    sku: str | None = Field(
        None, description="Catalogue SKU link; when present and trained, the prediction "
                          "comes from the trained CalendarDecomposed model instead of "
                          "the basis level."
    )
    promotionActive: bool = Field(False, description="Is a discount offer live this week?")


class RMPredictResponse(BaseModel):
    """Mirrors RestoMind's prediction document; drop-in for the `predictions` collection."""

    restaurantId: str
    productId: str
    modelVersionId: str
    targetWeek: str
    predictedOrders: int = Field(..., description="Predicted total units for the week")
    confidence: str
    featuresUsed: dict = Field(..., description="Snapshot of inputs, for auditability")
    factors: list[Factor] = Field(..., description="Drivers behind the number (empty post-rule)")
    dailyBreakdown: list[dict] = Field(..., description="Per-day predicted units")
    # Present when the product has no basis to forecast from: still training.
    trainingMessage: str | None = None


class RMSalesRow(BaseModel):
    date: dt.date
    productId: str
    salesQty: int = Field(..., ge=0)


class RMIngestRequest(BaseModel):
    """Per-restaurant sales history, so the bridge learns real demand levels.

    Feed this from RestoMind's `sales_transactions`. Optionally include `products` to
    register menu metadata (title/category/price) alongside the sales.
    """

    restaurantId: str
    records: list[RMSalesRow] = Field(..., min_length=1)
    products: list[RMProduct] | None = None


class RMIngestResponse(BaseModel):
    restaurantId: str
    rowsIngested: int
    productsTracked: int
    daysByProduct: dict[str, int]
    learnedLevels: dict[str, float]


class RMRegistryStatusResponse(BaseModel):
    restaurantId: str
    productsTracked: int
    usingLearnedLevel: int
    # This service owns the threshold, so it reports it rather than leaving the
    # caller to restate it. RestoMind renders a per-product progress bar from
    # observedDays/minDaysForLearned; when it hardcoded the number instead, a
    # tuning change here silently made that bar wrong.
    minDaysForLearned: int
    quietWindowDays: int
    confidentDays: int
    items: list[dict]


# -- errors --------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    detail: str
    hint: str | None = None
