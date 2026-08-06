"""M5's own real calendar -- the counterpart to `egypt_calendar.py`, but every event
here is recorded, not estimated.

WHY THIS EXISTS
---------------
`egypt_calendar.py` computes Ramadan/Eid/Kahk from astronomical Hijri conversion plus
hand-set priors for effect size. That's appropriate for a cold-start bakery with no
history -- but it means every backtest run against it is graded on assumptions the model
itself was built from.

M5 (Walmart) ships its own `calendar.csv`: real dates, real US retail events
(Thanksgiving, Christmas, SuperBowl, ...), real SNAP benefit-disbursement windows per
state. Nothing here is a prior -- it's what actually happened. Mirrors
`EgyptCalendar`'s `features(d)` / `feature_frame(start, end)` interface shape so the
rest of the pipeline (features, decomposition) stays structurally familiar, but every
column is sourced from the file, not computed from a rule.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

CALENDAR_CSV = Path(__file__).resolve().parents[2] / "data" / "m5" / "calendar.csv"

# M5's four recorded event categories (event_type_1/2 union).
EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]


class M5Calendar:
    """Real Walmart/M5 calendar, loaded once from `calendar.csv`."""

    def __init__(self, csv_path: Path = CALENDAR_CSV) -> None:
        raw = pd.read_csv(csv_path)
        raw["date"] = pd.to_datetime(raw["date"]).dt.date
        self._by_date: dict[dt.date, pd.Series] = {
            row["date"]: row for _, row in raw.iterrows()
        }
        self._min_date = raw["date"].min()
        self._max_date = raw["date"].max()

    def features(self, d: dt.date | str | pd.Timestamp) -> dict:
        """All calendar features for a single date, straight from M5's own file."""
        if isinstance(d, str):
            d = pd.Timestamp(d).date()
        elif isinstance(d, (pd.Timestamp, dt.datetime)):
            d = d.date()

        row = self._by_date.get(d)
        if row is None:
            # Outside M5's recorded range -- no event data available, treat as quiet.
            return {
                "date": d, "day_of_week": d.weekday(),
                "is_weekend": int(d.weekday() >= 5),  # US: Saturday/Sunday
                "is_sporting": 0, "is_cultural": 0, "is_national": 0, "is_religious": 0,
                "is_any_event": 0, "event_name": None,
                "snap_ca": 0, "snap_tx": 0, "snap_wi": 0,
            }

        types_today = {row.get("event_type_1"), row.get("event_type_2")}
        names_today = [n for n in (row.get("event_name_1"), row.get("event_name_2"))
                       if isinstance(n, str)]

        return {
            "date": d,
            "day_of_week": d.weekday(),
            "is_weekend": int(d.weekday() >= 5),
            "is_sporting": int("Sporting" in types_today),
            "is_cultural": int("Cultural" in types_today),
            "is_national": int("National" in types_today),
            "is_religious": int("Religious" in types_today),
            "is_any_event": int(len(names_today) > 0),
            "event_name": names_today[0] if names_today else None,
            "snap_ca": int(row.get("snap_CA", 0)),
            "snap_tx": int(row.get("snap_TX", 0)),
            "snap_wi": int(row.get("snap_WI", 0)),
        }

    def feature_frame(self, start: str | dt.date, end: str | dt.date) -> pd.DataFrame:
        """Calendar features for an inclusive date range, as a DataFrame."""
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame([self.features(d) for d in dates])


CALENDAR = M5Calendar()
