"""Validate the Egyptian calendar against externally known dates.

These assertions use dates that can be checked against public record. If the
underlying Hijri/Coptic libraries change behaviour, this suite fails loudly rather
than silently shifting the biggest demand events of the year.
"""

import datetime as dt

import pytest

from app.core.egypt_calendar import (
    CALENDAR,
    islamic_events,
    orthodox_easter,
    sham_el_nessim,
)


# -- Coptic calendar ----------------------------------------------------------------


@pytest.mark.parametrize(
    "year,expected",
    [
        (2023, dt.date(2023, 4, 16)),
        (2024, dt.date(2024, 5, 5)),
        (2025, dt.date(2025, 4, 20)),
        (2026, dt.date(2026, 4, 12)),
    ],
)
def test_orthodox_easter_matches_known_dates(year, expected):
    assert orthodox_easter(year) == expected


@pytest.mark.parametrize(
    "year,expected",
    [
        (2023, dt.date(2023, 4, 17)),
        (2024, dt.date(2024, 5, 6)),
        (2025, dt.date(2025, 4, 21)),
        (2026, dt.date(2026, 4, 13)),
    ],
)
def test_sham_el_nessim_is_monday_after_easter(year, expected):
    result = sham_el_nessim(year)
    assert result == expected
    assert result.weekday() == 0, "Sham El-Nessim always falls on a Monday"


# -- Hijri calendar -----------------------------------------------------------------


@pytest.mark.parametrize(
    "year,expected_start",
    [
        (2023, dt.date(2023, 3, 23)),
        (2024, dt.date(2024, 3, 11)),
        (2025, dt.date(2025, 3, 1)),
        (2026, dt.date(2026, 2, 18)),
    ],
)
def test_ramadan_start_matches_egypt_declared_dates(year, expected_start):
    starts = islamic_events(year).ramadan_starts
    assert expected_start in starts


def test_ramadan_drifts_earlier_each_year():
    """The ~11-day annual drift is the reason these must be computed, not learned."""
    starts = []
    for year in range(2023, 2028):
        ev = islamic_events(year)
        starts.extend(s for s in ev.ramadan_starts if s.year == year)
    starts.sort()
    for earlier, later in zip(starts, starts[1:]):
        drift = (later.replace(year=earlier.year) - earlier).days
        assert -13 <= drift <= -9, f"expected ~11 day earlier drift, got {drift}"


def test_ramadan_length_is_29_or_30_days():
    for year in range(2023, 2028):
        ev = islamic_events(year)
        for start, end in zip(ev.ramadan_starts, ev.ramadan_ends):
            assert 29 <= (end - start).days + 1 <= 30


def test_eid_al_fitr_follows_last_day_of_ramadan():
    for year in range(2023, 2028):
        ev = islamic_events(year)
        for end, fitr in zip(ev.ramadan_ends, ev.eid_fitr_starts):
            assert fitr == end + dt.timedelta(days=1)


# -- Feature extraction -------------------------------------------------------------


def test_weekend_is_friday_and_saturday():
    """Egypt's weekend is Fri/Sat. Using Sat/Sun inverts the weekly demand profile."""
    friday = CALENDAR.features(dt.date(2024, 3, 1))
    saturday = CALENDAR.features(dt.date(2024, 3, 2))
    sunday = CALENDAR.features(dt.date(2024, 3, 3))
    assert friday["is_weekend"] == 1
    assert saturday["is_weekend"] == 1
    assert sunday["is_weekend"] == 0


def test_ramadan_day_index_runs_start_to_end():
    start = dt.date(2024, 3, 11)
    assert CALENDAR.features(start)["ramadan_day_index"] == 1
    assert CALENDAR.features(start + dt.timedelta(days=9))["ramadan_day_index"] == 10
    assert CALENDAR.features(start - dt.timedelta(days=1))["is_ramadan"] == 0
    mid = CALENDAR.features(start + dt.timedelta(days=14))
    assert mid["is_ramadan"] == 1
    assert 0 < mid["ramadan_progress"] < 1


def test_last_ten_of_ramadan_flag():
    # Ramadan 2024: 11 Mar - 9 Apr. The final ten days start ~31 Mar.
    assert CALENDAR.features(dt.date(2024, 4, 5))["is_last_ten_of_ramadan"] == 1
    assert CALENDAR.features(dt.date(2024, 3, 15))["is_last_ten_of_ramadan"] == 0


def test_kahk_window_ramps_up_before_eid():
    """Kahk demand should ramp, not switch on -- the ramp is what the model fits."""
    eid = dt.date(2024, 4, 10)
    far = CALENDAR.features(eid - dt.timedelta(days=20))
    early = CALENDAR.features(eid - dt.timedelta(days=8))
    late = CALENDAR.features(eid - dt.timedelta(days=2))
    assert far["is_kahk_window"] == 0
    assert early["is_kahk_window"] == 1
    assert late["is_kahk_window"] == 1
    assert late["kahk_ramp"] > early["kahk_ramp"], "ramp must increase towards Eid"


def test_eid_al_fitr_flagged_as_holiday():
    f = CALENDAR.features(dt.date(2024, 4, 10))
    assert f["is_eid_fitr"] == 1
    assert f["is_public_holiday"] == 1
    assert f["holiday_name"] == "Eid al-Fitr"


def test_fixed_national_holidays():
    assert CALENDAR.features(dt.date(2024, 7, 23))["holiday_name"] == "July 23 Revolution"
    assert CALENDAR.features(dt.date(2024, 1, 25))["is_public_holiday"] == 1
    assert CALENDAR.features(dt.date(2024, 1, 7))["is_coptic_christmas"] == 1


def test_school_term_covers_term_time_and_excludes_summer():
    assert CALENDAR.features(dt.date(2024, 11, 12))["is_school_term"] == 1
    assert CALENDAR.features(dt.date(2024, 3, 5))["is_school_term"] == 1
    assert CALENDAR.features(dt.date(2024, 7, 15))["is_school_term"] == 0


def test_payday_window_covers_month_boundary():
    assert CALENDAR.features(dt.date(2024, 3, 30))["is_payday_window"] == 1
    assert CALENDAR.features(dt.date(2024, 4, 2))["is_payday_window"] == 1
    assert CALENDAR.features(dt.date(2024, 3, 15))["is_payday_window"] == 0


# -- Frames -------------------------------------------------------------------------


def test_feature_frame_shape_and_coverage():
    df = CALENDAR.feature_frame("2024-01-01", "2024-12-31")
    assert len(df) == 366  # 2024 is a leap year
    assert df["is_ramadan"].sum() in (29, 30)
    assert df["is_weekend"].sum() == 104  # 52 Fridays + 52 Saturdays
    assert df["holiday_name"].notna().sum() > 7


def test_prophet_holidays_frame_has_required_columns():
    df = CALENDAR.prophet_holidays("2024-01-01", "2025-12-31")
    assert {"holiday", "ds", "lower_window", "upper_window"} <= set(df.columns)
    assert "ramadan" in set(df["holiday"])
    assert "eid_al_fitr" in set(df["holiday"])
    ramadan = df[df["holiday"] == "ramadan"].iloc[0]
    assert ramadan["upper_window"] >= 28, "Ramadan window should span the month"
