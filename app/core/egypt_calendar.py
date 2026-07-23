"""Egyptian calendar features for bakery demand forecasting.

This module is the differentiating piece of the forecaster. Egyptian bakery demand
is driven by three overlapping calendars:

  * Gregorian  -- national holidays, school terms, salary cycles
  * Hijri      -- Ramadan and the two Eids
  * Coptic     -- Coptic Christmas, Coptic Easter, Sham El-Nessim

The Hijri events move roughly 11 days earlier every Gregorian year. A model given
only two years of history cannot possibly infer that drift, so these effects are
*computed and injected as features* rather than left to be learned.

A note on moon sighting
-----------------------
Ramadan and the Eids are officially declared in Egypt by moon sighting (Dar al-Ifta),
which can land one day either side of the arithmetic Umm al-Qura calendar used here.
For forecasting that is usually tolerable, but the error lands exactly on the highest
demand days of the year. `RAMADAN_OVERRIDES` therefore lets confirmed real start dates
be pinned; anything not pinned falls back to the computed date.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import pandas as pd
from hijridate import Gregorian, Hijri

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Egypt's weekend is Friday + Saturday, not Saturday + Sunday. Getting this wrong
# inverts the entire weekly demand profile.
FRIDAY, SATURDAY = 4, 5

# Fixed-date Egyptian public holidays (month, day) -> name.
FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 7): "Coptic Christmas",
    (1, 25): "Revolution Day / Police Day",
    (4, 25): "Sinai Liberation Day",
    (5, 1): "Labour Day",
    (6, 30): "June 30 Revolution",
    (7, 23): "July 23 Revolution",
    (10, 6): "Armed Forces Day",
}

# Confirmed Ramadan 1st-day start dates as actually declared in Egypt.
# Sourced from Dar al-Ifta announcements; extend as new years are declared.
RAMADAN_OVERRIDES: dict[int, dt.date] = {
    2022: dt.date(2022, 4, 2),
    2023: dt.date(2023, 3, 23),
    2024: dt.date(2024, 3, 11),
    2025: dt.date(2025, 3, 1),
    2026: dt.date(2026, 2, 18),
}

# Approximate Egyptian school calendar: (start_md, end_md) per term.
# First term runs late Sept -> mid Jan; second term mid Feb -> early June.
SCHOOL_TERMS = [((9, 20), (1, 15)), ((2, 10), (6, 5))]

# Days before Eid al-Fitr during which kahk (Eid biscuit) demand ramps up.
KAHK_WINDOW_DAYS = 10

# Egyptian salaries land at end of month; spending lifts around that boundary.
PAYDAY_TAIL_DAYS = 3   # last N days of the month
PAYDAY_HEAD_DAYS = 5   # first N days of the month


# --------------------------------------------------------------------------------------
# Moveable feast computation
# --------------------------------------------------------------------------------------


def orthodox_easter(year: int) -> dt.date:
    """Coptic/Orthodox Easter for a Gregorian year (Meeus Julian algorithm).

    Computed on the Julian calendar, then shifted to Gregorian. The +13 day offset
    holds for 1900-2099, which comfortably covers any forecasting horizon here.
    """
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = ((d + e + 114) % 31) + 1
    julian = dt.date(year, month, day)
    return julian + dt.timedelta(days=13)


def sham_el_nessim(year: int) -> dt.date:
    """Sham El-Nessim: the Monday directly after Coptic Easter.

    A national spring festival celebrated across religious lines, and a large demand
    event for feteer and fesikh-accompanying breads.
    """
    return orthodox_easter(year) + dt.timedelta(days=1)


def _hijri_years_touching(year: int) -> range:
    """Hijri years overlapping a given Gregorian year."""
    start = Gregorian(year, 1, 1).to_hijri().year
    end = Gregorian(year, 12, 31).to_hijri().year
    return range(start, end + 1)


def _hijri_to_greg(h_year: int, h_month: int, h_day: int) -> dt.date | None:
    try:
        return dt.date(*Hijri(h_year, h_month, h_day).to_gregorian().datetuple())
    except (ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class IslamicEvents:
    """Islamic event dates falling in (or spilling into) one Gregorian year."""

    ramadan_starts: list[dt.date] = field(default_factory=list)
    ramadan_ends: list[dt.date] = field(default_factory=list)
    eid_fitr_starts: list[dt.date] = field(default_factory=list)
    eid_adha_starts: list[dt.date] = field(default_factory=list)
    islamic_new_year: list[dt.date] = field(default_factory=list)
    prophet_birthday: list[dt.date] = field(default_factory=list)


@lru_cache(maxsize=256)
def islamic_events(year: int) -> IslamicEvents:
    """Compute Islamic events for a Gregorian year, applying declared overrides."""
    ev = IslamicEvents()
    for hy in _hijri_years_touching(year):
        start = _hijri_to_greg(hy, 9, 1)
        if start is None:
            continue

        # Prefer the officially declared start when we have it.
        override = RAMADAN_OVERRIDES.get(start.year)
        if override is not None:
            shift = (override - start).days
            # Only trust an override that is a small sighting correction, not a
            # mismatch caused by two Ramadans landing in one Gregorian year.
            if abs(shift) <= 3:
                start = override

        eid_fitr = _hijri_to_greg(hy, 10, 1)
        if eid_fitr is None:
            continue
        # Keep Eid consistent with any sighting correction applied to Ramadan.
        ramadan_len = (eid_fitr - _hijri_to_greg(hy, 9, 1)).days
        eid_fitr = start + dt.timedelta(days=ramadan_len)

        ev.ramadan_starts.append(start)
        ev.ramadan_ends.append(eid_fitr - dt.timedelta(days=1))
        ev.eid_fitr_starts.append(eid_fitr)

        for lst, (m, d) in (
            (ev.eid_adha_starts, (12, 10)),
            (ev.islamic_new_year, (1, 1)),
            (ev.prophet_birthday, (3, 12)),
        ):
            g = _hijri_to_greg(hy, m, d)
            if g is not None:
                lst.append(g)
    return ev


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------


class EgyptCalendar:
    """Emits per-date calendar features for the forecasting pipeline.

    Consumed three ways: as engineered columns by LightGBM and River, and as a
    Prophet-style holidays frame via :meth:`prophet_holidays`.
    """

    def __init__(self, kahk_window_days: int = KAHK_WINDOW_DAYS) -> None:
        self.kahk_window_days = kahk_window_days

    # -- individual signals -------------------------------------------------------

    def _ramadan_position(self, d: dt.date) -> tuple[int, int]:
        """Return (day_index, total_days); day_index is 1-based, 0 when not Ramadan."""
        for year in (d.year, d.year - 1):
            ev = islamic_events(year)
            for start, end in zip(ev.ramadan_starts, ev.ramadan_ends):
                if start <= d <= end:
                    return (d - start).days + 1, (end - start).days + 1
        return 0, 0

    def _next_eid_fitr(self, d: dt.date) -> dt.date | None:
        candidates = [
            e
            for year in (d.year, d.year + 1)
            for e in islamic_events(year).eid_fitr_starts
            if e >= d
        ]
        return min(candidates) if candidates else None

    def _eid_day_index(self, d: dt.date, starts: Iterable[dt.date], length: int) -> int:
        """1-based day within an Eid holiday, 0 if outside it."""
        for s in starts:
            if s <= d < s + dt.timedelta(days=length):
                return (d - s).days + 1
        return 0

    def _is_school_term(self, d: dt.date) -> bool:
        for (sm, sd), (em, ed) in SCHOOL_TERMS:
            start, end = (sm, sd), (em, ed)
            cur = (d.month, d.day)
            if start <= end:
                if start <= cur <= end:
                    return True
            else:  # term wraps the new year (e.g. Sep 20 -> Jan 15)
                if cur >= start or cur <= end:
                    return True
        return False

    def _is_payday_window(self, d: dt.date) -> bool:
        last_day = (d.replace(day=1) + pd.offsets.MonthEnd(1)).day
        return d.day <= PAYDAY_HEAD_DAYS or d.day > last_day - PAYDAY_TAIL_DAYS

    def _holiday_name(self, d: dt.date) -> str | None:
        if (d.month, d.day) in FIXED_HOLIDAYS:
            return FIXED_HOLIDAYS[(d.month, d.day)]
        if d == sham_el_nessim(d.year):
            return "Sham El-Nessim"
        if d == orthodox_easter(d.year):
            return "Coptic Easter"
        ev = islamic_events(d.year)
        if self._eid_day_index(d, ev.eid_fitr_starts, 3):
            return "Eid al-Fitr"
        if self._eid_day_index(d, ev.eid_adha_starts, 4):
            return "Eid al-Adha"
        if d in ev.islamic_new_year:
            return "Islamic New Year"
        if d in ev.prophet_birthday:
            return "Prophet's Birthday"
        return None

    # -- public API ---------------------------------------------------------------

    def features(self, d: dt.date | str | pd.Timestamp) -> dict:
        """All calendar features for a single date."""
        if isinstance(d, str):
            d = pd.Timestamp(d).date()
        elif isinstance(d, pd.Timestamp):
            d = d.date()
        elif isinstance(d, dt.datetime):
            d = d.date()

        ev = islamic_events(d.year)
        ram_idx, ram_len = self._ramadan_position(d)
        next_fitr = self._next_eid_fitr(d)
        days_to_fitr = (next_fitr - d).days if next_fitr else 999

        eid_fitr_idx = self._eid_day_index(d, ev.eid_fitr_starts, 3)
        eid_adha_idx = self._eid_day_index(d, ev.eid_adha_starts, 4)
        holiday = self._holiday_name(d)

        return {
            "date": d,
            "day_of_week": d.weekday(),
            "is_weekend": int(d.weekday() in (FRIDAY, SATURDAY)),
            "day_of_month": d.day,
            "month": d.month,
            "week_of_year": d.isocalendar().week,
            # Ramadan modelled as a position curve, not a flag: demand within the
            # month is far from uniform (last 10 days differ sharply from the first).
            "is_ramadan": int(ram_idx > 0),
            "ramadan_day_index": ram_idx,
            "ramadan_progress": round(ram_idx / ram_len, 4) if ram_len else 0.0,
            "is_last_ten_of_ramadan": int(ram_idx > 0 and ram_idx > ram_len - 10),
            # Kahk ramps steeply in the days before Eid al-Fitr.
            "days_to_eid_fitr": min(days_to_fitr, 999),
            "is_kahk_window": int(0 < days_to_fitr <= self.kahk_window_days),
            "kahk_ramp": round(
                max(0.0, (self.kahk_window_days - days_to_fitr) / self.kahk_window_days), 4
            )
            if 0 < days_to_fitr <= self.kahk_window_days
            else 0.0,
            "is_eid_fitr": int(eid_fitr_idx > 0),
            "eid_fitr_day_index": eid_fitr_idx,
            "is_eid_adha": int(eid_adha_idx > 0),
            "eid_adha_day_index": eid_adha_idx,
            "is_sham_el_nessim": int(d == sham_el_nessim(d.year)),
            "is_coptic_christmas": int((d.month, d.day) == (1, 7)),
            "is_public_holiday": int(holiday is not None),
            "holiday_name": holiday,
            "is_school_term": int(self._is_school_term(d)),
            "is_payday_window": int(self._is_payday_window(d)),
        }

    def feature_frame(self, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        """Calendar features for an inclusive date range, as a DataFrame."""
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame([self.features(d) for d in dates])

    def prophet_holidays(self, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        """Holidays frame for Prophet, with windows around the big multi-day events.

        Prophet models each named holiday with its own effect, so Ramadan is emitted
        as a single window rather than per-day; the per-day shape is what the
        LightGBM path uses `ramadan_day_index` for.
        """
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date()
        rows: list[dict] = []

        for year in range(start_d.year - 1, end_d.year + 2):
            ev = islamic_events(year)
            for s, e in zip(ev.ramadan_starts, ev.ramadan_ends):
                rows.append(
                    {"holiday": "ramadan", "ds": pd.Timestamp(s),
                     "lower_window": 0, "upper_window": (e - s).days}
                )
            for s in ev.eid_fitr_starts:
                # Negative lower_window pulls in the pre-Eid kahk ramp.
                rows.append(
                    {"holiday": "eid_al_fitr", "ds": pd.Timestamp(s),
                     "lower_window": -self.kahk_window_days, "upper_window": 3}
                )
            for s in ev.eid_adha_starts:
                rows.append(
                    {"holiday": "eid_al_adha", "ds": pd.Timestamp(s),
                     "lower_window": -2, "upper_window": 4}
                )
            rows.append(
                {"holiday": "sham_el_nessim", "ds": pd.Timestamp(sham_el_nessim(year)),
                 "lower_window": -1, "upper_window": 1}
            )
            for (m, d), name in FIXED_HOLIDAYS.items():
                rows.append(
                    {"holiday": name.lower().replace(" ", "_").replace("/", ""),
                     "ds": pd.Timestamp(dt.date(year, m, d)),
                     "lower_window": 0, "upper_window": 0}
                )

        df = pd.DataFrame(rows).drop_duplicates(subset=["holiday", "ds"])
        return df.sort_values("ds").reset_index(drop=True)


# Module-level default instance for convenience.
CALENDAR = EgyptCalendar()
