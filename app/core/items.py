"""Bakery item catalogue: economics, shelf life, and demand behaviour.

Single source of truth, shared by the synthetic generator, the forecaster's
newsvendor quantile, and the surplus/marketing endpoints.

The prices and costs here are *plausible Egyptian bakery figures, not measured ones*.
They set the newsvendor service level q*, so when real unit economics arrive they
should replace these -- the forecast is only profit-optimal to the extent these are right.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def spoilage_severity_for_shelf_life(shelf_life_days: float | None) -> float:
    """Fraction of a leftover unit's cost that is actually lost, from shelf life alone.

    Single source of truth for this tier table -- `Item.spoilage_severity` (below, for
    the built-in catalogue) and the RestoMind bridge's `_spoilage_severity`
    (`app/integration/restomind.py`, for a real restaurant's `freshnessWindow`) both
    call this instead of keeping their own copy of the same four numbers.

    A same-day item (baladi bread) left on the shelf is a total write-off. Petit four
    keeps two weeks and kahk a month, so an unsold unit is mostly just stock that sells
    tomorrow. Charging both at full COGS would badly over-penalise the long-life items
    and push their production below what is profitable.
    """
    d = shelf_life_days if shelf_life_days is not None else 1
    if d <= 1:
        return 1.0
    if d <= 3:
        return 0.5
    if d <= 7:
        return 0.25
    return 0.15


@dataclass(frozen=True)
class Item:
    """One bakery SKU.

    The `*_sensitivity` fields are the *true* effect sizes used to generate synthetic
    demand. The forecaster never sees them -- they exist so the recovery test can check
    that the model rediscovers them.
    """

    sku: str
    name_ar: str
    name_en: str
    category: str
    unit_price: float          # EGP charged to customer
    unit_cost: float           # EGP cost to produce
    shelf_life_days: int       # 1 = must sell same day
    base_daily_demand: float   # units/day at baseline

    # --- true effect sizes (synthetic ground truth) ---
    weekend_mult: float = 1.0          # Fri/Sat multiplier
    ramadan_mult: float = 1.0          # overall Ramadan level shift
    ramadan_late_boost: float = 1.0    # extra lift in the last 10 days
    eid_mult: float = 1.0              # during Eid al-Fitr/Adha
    kahk_peak_mult: float = 1.0        # peak of the pre-Eid kahk ramp
    school_mult: float = 1.0           # during school term
    payday_mult: float = 1.0           # around the salary window
    sham_mult: float = 1.0             # Sham El-Nessim
    holiday_mult: float = 1.0          # generic public holiday
    seasonal_only: bool = False        # near-zero demand outside its event window
    noise_sigma: float = 0.18          # lognormal multiplicative noise

    # Goodwill lost when a customer finds the shelf empty, as a multiple of unit margin.
    # A daily-staple shopper who cannot get baladi bread may switch bakery for good, so
    # the true cost of that stockout far exceeds one missed sale. Indulgence items are
    # more forgiving -- nobody changes bakery over a sold-out slice of gateau.
    stockout_goodwill_mult: float = 0.5

    @property
    def margin(self) -> float:
        """EGP earned per unit sold."""
        return self.unit_price - self.unit_cost

    @property
    def spoilage_severity(self) -> float:
        """Fraction of a leftover unit's cost that is actually lost. See module-level
        `spoilage_severity_for_shelf_life` -- the tier table lives there now."""
        return spoilage_severity_for_shelf_life(self.shelf_life_days)

    @property
    def shortage_cost(self) -> float:
        """Cu: full cost of failing to have one more unit available.

        Lost margin plus lost goodwill. Pricing goodwill at zero is a trap: the
        optimiser then discovers it can eliminate almost all waste by simply letting
        shelves run empty, which is cheap on a spreadsheet and ruinous in a high street.
        """
        return self.margin * (1.0 + self.stockout_goodwill_mult)

    @property
    def newsvendor_quantile(self) -> float:
        """Profit-optimal service level q* = Cu / (Cu + Co).

        Cu = margin + goodwill lost on a unit we failed to make.
        Co = cost actually destroyed when a unit goes unsold.

        q* is deliberately *per item*, not a global setting. It stays low for
        low-markup, same-day staples where waste destroys real value, and rises for
        high-markup or long-shelf-life items where running out is the costlier mistake.
        Forecasting the mean would get both cases wrong in opposite directions.
        """
        cu = self.shortage_cost
        co = self.unit_cost * self.spoilage_severity
        if cu + co <= 0:
            return 0.5
        return round(cu / (cu + co), 4)


# Ramadan reshapes the day completely: nothing is eaten in daylight, then a large
# iftar with dessert. Breakfast items collapse; desserts roughly double.
_CATALOGUE: list[Item] = [
    Item(
        sku="BREAD_BALADI", name_ar="عيش بلدي", name_en="Baladi Bread",
        category="bread", unit_price=1.5, unit_cost=0.9, shelf_life_days=1,
        base_daily_demand=1200, weekend_mult=1.10, ramadan_mult=1.15,
        payday_mult=1.03, holiday_mult=1.05, noise_sigma=0.12, stockout_goodwill_mult=1.60,
    ),
    Item(
        sku="BREAD_FINO", name_ar="عيش فينو", name_en="Fino Bread",
        category="bread", unit_price=2.5, unit_cost=1.4, shelf_life_days=1,
        base_daily_demand=420, weekend_mult=1.05, ramadan_mult=0.85,
        school_mult=1.18, payday_mult=1.04, noise_sigma=0.15, stockout_goodwill_mult=1.20,
    ),
    Item(
        sku="PASTRY_CROISSANT", name_ar="كرواسون", name_en="Croissant",
        category="pastry", unit_price=18.0, unit_cost=9.5, shelf_life_days=2,
        base_daily_demand=180, weekend_mult=1.35, ramadan_mult=0.45,
        school_mult=1.22, payday_mult=1.12, holiday_mult=1.15, noise_sigma=0.20, stockout_goodwill_mult=0.60,
    ),
    Item(
        sku="PASTRY_DONUT", name_ar="دوناتس", name_en="Donuts",
        category="pastry", unit_price=15.0, unit_cost=7.0, shelf_life_days=1,
        base_daily_demand=140, weekend_mult=1.45, ramadan_mult=0.50,
        school_mult=1.25, payday_mult=1.15, noise_sigma=0.24, stockout_goodwill_mult=0.50,
    ),
    Item(
        sku="CAKE_GATEAU", name_ar="جاتوه", name_en="Gateau Slice",
        category="cake", unit_price=35.0, unit_cost=17.0, shelf_life_days=2,
        base_daily_demand=95, weekend_mult=1.55, ramadan_mult=1.75,
        ramadan_late_boost=1.20, eid_mult=2.10, payday_mult=1.25,
        holiday_mult=1.40, noise_sigma=0.26, stockout_goodwill_mult=0.35,
    ),
    Item(
        sku="SWEET_BASBOUSA", name_ar="بسبوسة", name_en="Basbousa",
        category="sweet", unit_price=12.0, unit_cost=5.5, shelf_life_days=3,
        base_daily_demand=110, weekend_mult=1.30, ramadan_mult=2.05,
        ramadan_late_boost=1.25, eid_mult=1.60, payday_mult=1.10,
        holiday_mult=1.25, noise_sigma=0.22, stockout_goodwill_mult=0.35,
    ),
    Item(
        sku="SWEET_KONAFA", name_ar="كنافة", name_en="Konafa",
        category="sweet", unit_price=45.0, unit_cost=22.0, shelf_life_days=2,
        base_daily_demand=40, weekend_mult=1.20, ramadan_mult=4.50,
        ramadan_late_boost=1.30, eid_mult=1.80, payday_mult=1.10,
        noise_sigma=0.30, stockout_goodwill_mult=0.40,
    ),
    Item(
        sku="SEASONAL_KAHK", name_ar="كحك العيد", name_en="Eid Kahk",
        category="seasonal", unit_price=90.0, unit_cost=42.0, shelf_life_days=30,
        base_daily_demand=8, kahk_peak_mult=45.0, eid_mult=6.0,
        seasonal_only=True, noise_sigma=0.35, stockout_goodwill_mult=0.90,
    ),
    Item(
        sku="SAVOURY_FETEER", name_ar="فطير مشلتت", name_en="Feteer Meshaltet",
        category="savoury", unit_price=60.0, unit_cost=28.0, shelf_life_days=1,
        base_daily_demand=55, weekend_mult=1.60, ramadan_mult=1.30,
        sham_mult=3.20, payday_mult=1.15, holiday_mult=1.35, noise_sigma=0.28, stockout_goodwill_mult=0.60,
    ),
    Item(
        sku="SAVOURY_SANDWICH", name_ar="سندوتشات", name_en="Sandwiches",
        category="savoury", unit_price=25.0, unit_cost=13.0, shelf_life_days=1,
        base_daily_demand=160, weekend_mult=0.70, ramadan_mult=0.30,
        school_mult=1.55, payday_mult=1.08, noise_sigma=0.22, stockout_goodwill_mult=0.80,
    ),
    Item(
        sku="DRY_PETITFOUR", name_ar="بيتي فور", name_en="Petit Four",
        category="dry", unit_price=70.0, unit_cost=32.0, shelf_life_days=14,
        base_daily_demand=25, weekend_mult=1.25, ramadan_mult=1.40,
        eid_mult=3.20, payday_mult=1.20, holiday_mult=1.50, noise_sigma=0.30, stockout_goodwill_mult=0.30,
    ),
]

# These 11 items are the SIMULATION fixtures, not a runtime catalogue. generate() needs
# known items with known effect sizes to synthesise the demo dataset, and the demo
# scripts (dashboard, run_simulation, backtest) are written against them. The running
# service does NOT know them by default: it learns its items from uploaded data (COLD
# START / dynamic catalogue, see `Catalogue` below). The /model/status and forecast
# routes never consult this mapping.
CATALOGUE: list[Item] = _CATALOGUE

# Back-compat alias for demo scripts and the test suite, so they keep working against
# the synthetic fixtures. Runtime code uses Catalogue instead.
BY_SKU: dict[str, Item] = {i.sku: i for i in CATALOGUE}


def get_item(sku: str) -> Item:
    """Look up a synthetic fixture item, raising a clear error for an unknown SKU."""
    try:
        return BY_SKU[sku]
    except KeyError:
        raise KeyError(
            f"Unknown SKU {sku!r}. Known SKUs: {', '.join(sorted(BY_SKU))}"
        ) from None


class Catalogue:
    """A dynamic, per-service item registry.

    The runtime equivalent of the fixed `BY_SKU`: items are registered from whatever a
    real bakery/user actually uploads (via /catalogue/upsert or the economics attached
    to ingested sales). A brand-new bakery with nothing uploaded has an EMPTY catalogue
    -- no hardcoded 11, no forecast for anything until its own items arrive.

    Items are registered by their economics because those set the newsvendor quantile
    that decides how conservative the forecast is. Where uploads omit a field the
    sensible bakery default is applied (same-day shelf life, zero price/cost only when
    genuinely unknown).
    """

    def __init__(self) -> None:
        self._items: dict[str, Item] = {}

    def register(
        self,
        sku: str,
        *,
        unit_price: float | None = None,
        unit_cost: float | None = None,
        shelf_life_days: int | None = None,
        name_ar: str | None = None,
        name_en: str | None = None,
        category: str | None = None,
    ) -> Item:
        """Register or refresh one item, returning it.

        Precedence for each field: an explicitly provided value > the previously
        registered value > the synthetic fixture (if any) > the bare default. This
        keeps economics intact when a caller refreshes an item without restating them
        (e.g. an ingest that only sends sales).
        """
        prev = self._items.get(sku)
        base = _CATALOGUE_BY_SKU.get(sku)

        # Empty/zero explicit values mean "not stated" -- preserve the previous value
        # (or the fixture) rather than clobbering a good registration with a default.
        unit_price = unit_price if unit_price else None
        unit_cost = unit_cost if unit_cost else None
        shelf_life_days = shelf_life_days if shelf_life_days else None
        name_ar = name_ar or None
        name_en = name_en or None
        category = category or None

        def pick(provided, prev_val, base_val, default):
            if provided is not None:
                return provided
            if prev_val is not None:
                return prev_val
            return base_val if base_val is not None else default

        item = Item(
            sku=sku,
            name_ar=pick(name_ar, prev.name_ar if prev else None,
                         base.name_ar if base else None, sku),
            name_en=pick(name_en, prev.name_en if prev else None,
                         base.name_en if base else None, sku),
            category=pick(category if category else None,
                          prev.category if prev else None,
                          base.category if base else None, "general"),
            unit_price=float(pick(unit_price, prev.unit_price if prev else None,
                                  base.unit_price if base else None, 0.0)),
            unit_cost=float(pick(unit_cost, prev.unit_cost if prev else None,
                                 base.unit_cost if base else None, 0.0)),
            shelf_life_days=int(pick(shelf_life_days, prev.shelf_life_days if prev else None,
                                     base.shelf_life_days if base else None, 1)),
            base_daily_demand=(prev.base_daily_demand if prev else
                               (base.base_daily_demand if base else 0.0)),
        )
        self._items[sku] = item
        return item

    def get(self, sku: str, default=None) -> Item | None:
        return self._items.get(sku, default)

    def skus(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[Item]:
        return [self._items[s] for s in self.skus()]

    def as_dict(self) -> dict[str, Item]:
        return dict(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __contains__(self, sku: object) -> bool:
        return sku in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def register_from_data(self, rows: pd.DataFrame | None) -> None:
        """Register every SKU present in a data frame, taking economics from its columns.

        The synthetic generator and ingest payloads carry `unit_price` / `unit_cost`
        (and optionally `shelf_life_days`) per row, so a data-driven item registry needs
        no separate metadata call -- an SKU becomes known the first time data about it
        arrives.
        """
        if rows is None or rows.empty:
            return
        for sku, grp in rows.groupby("sku", sort=False):
            first = grp.iloc[0]
            price = float(first.get("unit_price", 0.0) or 0.0) or None
            cost = float(first.get("unit_cost", 0.0) or 0.0) or None
            shelf = int(first.get("shelf_life_days", 1) or 1) or None
            self.register(
                sku,
                unit_price=price,
                unit_cost=cost,
                shelf_life_days=shelf,
                name_ar=str(first.get("item_name_ar", "") or "") or None,
                name_en=str(first.get("item_name_en", "") or "") or None,
                category=str(first.get("category", "") or "") or None,
            )


def new_item_from_row(row) -> Item:
    """Build an Item from an uploaded/simulated data row.

    Economics (price/cost/shelf life) are read from the row's columns when present,
    defaulting to a same-day staple otherwise. This is what keeps the forecast
    profit-optimal per item without a fixed catalogue.
    """
    price = float(row.get("unit_price", 0.0) or 0.0)
    cost = float(row.get("unit_cost", 0.0) or 0.0)
    shelf = int(row.get("shelf_life_days", 1) or 1)
    return Item(
        sku=str(row["sku"]),
        name_ar=str(row.get("item_name_ar", "") or ""),
        name_en=str(row.get("item_name_en", "") or ""),
        category=str(row.get("category", "") or "general"),
        unit_price=price,
        unit_cost=cost,
        shelf_life_days=shelf,
        base_daily_demand=float(row.get("base_daily_demand", 0.0) or 0.0),
    )


def newsvendor_q_from_row(row) -> float:
    """Profit-optimal quantile from a data row's own economics (defaults: 0.5).

    Used wherever a model must know how conservatively to forecast a SKU WITHOUT a
    fixed catalogue lookup -- the newsvendor q* is derived from the uploaded price,
    cost and shelf life, so a brand new SKU still gets a sane service level.
    """
    item = new_item_from_row(row)
    base = _CATALOGUE_BY_SKU.get(item.sku)
    if item.unit_price <= 0 or item.unit_cost <= 0:
        return (base.newsvendor_quantile if base else 0.5)
    return item.newsvendor_quantile


_CATALOGUE_BY_SKU: dict[str, Item] = {i.sku: i for i in CATALOGUE}
