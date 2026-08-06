"""France's own real calendar -- the counterpart to `m5_calendar.py`/`favorita_calendar.py`,
using the `holidays` library (PyPI, MIT-licensed, actively maintained) for accurate
`jours feries` rather than hand-rolling Easter-relative date math.

This is what should have been used the first time the French bakery dataset was tested
this session -- that run went through the rule-based bridge with EGYPT's calendar
(Ramadan/Eid/Kahk applied to French demand) and lost badly to naive (WAPE 0.605). Same
real data, same real dates, this time France's own public holidays.
"""

from __future__ import annotations

import datetime as dt

import holidays as holidays_lib
import pandas as pd


class FrenchCalendar:
    """Real French national holidays, loaded once per needed year range."""

    def __init__(self) -> None:
        self._cache: dict[int, holidays_lib.HolidayBase] = {}

    def _for_year(self, year: int) -> holidays_lib.HolidayBase:
        if year not in self._cache:
            self._cache[year] = holidays_lib.France(years=[year - 1, year, year + 1])
        return self._cache[year]

    def features(self, d: dt.date | str | pd.Timestamp) -> dict:
        if isinstance(d, str):
            d = pd.Timestamp(d).date()
        elif isinstance(d, (pd.Timestamp, dt.datetime)):
            d = d.date()

        fr = self._for_year(d.year)
        name = fr.get(d)

        return {
            "date": d,
            "day_of_week": d.weekday(),
            "is_weekend": int(d.weekday() >= 5),   # France: Saturday/Sunday
            "is_public_holiday": int(name is not None),
            "holiday_name": name,
            # August is France's dominant national holiday month (les grandes
            # vacances) -- much of the country, and many small businesses, close or
            # slow down for it. Worth its own flag, distinct from a single-day holiday.
            "is_august": int(d.month == 8),
        }

    def feature_frame(self, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame([self.features(d) for d in dates])


CALENDAR = FrenchCalendar()
