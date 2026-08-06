"""Favorita's own real calendar -- the Ecuador counterpart to `m5_calendar.py`.

`holidays_events.csv` records real Ecuadorian holidays/events at three locale levels
(National / Regional / Local). Because the test subset is store 44 in Quito, Pichincha,
only National events plus Local(Quito) and Regional(Pichincha) rows actually apply --
a Local holiday in Cuenca has no bearing on a Quito store's demand, so it is excluded
rather than pooled in as noise.

`transferred=True` rows mark a holiday whose actual day off moved elsewhere; the
following `Transfer`-type row marks the date it moved TO. Standard treatment for this
well-known dataset: the transferred date is not actually observed as a holiday, so it
is excluded; the Transfer row is included in its place.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

HOLIDAYS_CSV = Path(__file__).resolve().parents[2] / "data" / "favorita" / "holidays_events.csv"

STORE_CITY = "Quito"
STORE_STATE = "Pichincha"

EVENT_TYPES = ["Holiday", "Event", "Additional", "Bridge"]


class FavoritaCalendar:
    """Real Ecuador/Favorita calendar for one store's locale, loaded once."""

    def __init__(
        self, csv_path: Path = HOLIDAYS_CSV,
        city: str = STORE_CITY, state: str = STORE_STATE,
    ) -> None:
        raw = pd.read_csv(csv_path)
        raw["date"] = pd.to_datetime(raw["date"]).dt.date

        applies = (
            (raw["locale"] == "National")
            | ((raw["locale"] == "Regional") & (raw["locale_name"] == state))
            | ((raw["locale"] == "Local") & (raw["locale_name"] == city))
        )
        # Exclude the original date of a transferred holiday -- it was not actually
        # observed as a day off there; the matching Transfer row covers the real date.
        not_moved_away = ~((raw["type"] == "Holiday") & (raw["transferred"] == True))  # noqa: E712
        relevant = raw[applies & not_moved_away]

        self._by_date: dict[dt.date, list[pd.Series]] = {}
        for _, row in relevant.iterrows():
            self._by_date.setdefault(row["date"], []).append(row)

    def features(self, d: dt.date | str | pd.Timestamp) -> dict:
        if isinstance(d, str):
            d = pd.Timestamp(d).date()
        elif isinstance(d, (pd.Timestamp, dt.datetime)):
            d = d.date()

        rows = self._by_date.get(d, [])
        types_today = {r["type"] for r in rows}

        return {
            "date": d,
            "day_of_week": d.weekday(),
            "is_weekend": int(d.weekday() >= 5),
            "is_holiday": int("Holiday" in types_today or "Transfer" in types_today),
            "is_event": int("Event" in types_today),
            "is_additional": int("Additional" in types_today),
            "is_bridge": int("Bridge" in types_today),
            "is_any_event": int(len(rows) > 0),
        }

    def feature_frame(self, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame([self.features(d) for d in dates])


CALENDAR = FavoritaCalendar()
