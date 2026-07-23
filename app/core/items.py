"""Bakery item catalogue: economics, shelf life, and demand behaviour.

Single source of truth, shared by the synthetic generator, the forecaster's
newsvendor quantile, and the surplus/marketing endpoints.

The prices and costs here are *plausible Egyptian bakery figures, not measured ones*.
They set the newsvendor service level q*, so when real unit economics arrive they
should replace these -- the forecast is only profit-optimal to the extent these are right.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        """Fraction of a leftover unit's cost that is actually lost.

        A same-day item (baladi bread) left on the shelf is a total write-off. Petit
        four keeps two weeks and kahk a month, so an unsold unit is mostly just stock
        that sells tomorrow. Charging both at full COGS would badly over-penalise the
        long-life items and push their production below what is profitable.
        """
        if self.shelf_life_days <= 1:
            return 1.0
        if self.shelf_life_days <= 3:
            return 0.5
        if self.shelf_life_days <= 7:
            return 0.25
        return 0.15

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
CATALOGUE: list[Item] = [
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

BY_SKU: dict[str, Item] = {i.sku: i for i in CATALOGUE}


def get_item(sku: str) -> Item:
    """Look up an item, raising a clear error for an unknown SKU."""
    try:
        return BY_SKU[sku]
    except KeyError:
        raise KeyError(
            f"Unknown SKU {sku!r}. Known SKUs: {', '.join(sorted(BY_SKU))}"
        ) from None
