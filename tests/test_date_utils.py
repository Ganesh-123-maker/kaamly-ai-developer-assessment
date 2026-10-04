"""Tests for app/date_utils.py relative date resolution."""

from __future__ import annotations

import pytest

from app.date_utils import resolve_relative_date

REF_SUNDAY = "2026-10-04"   # Sunday
REF_SATURDAY = "2026-10-10" # Saturday


def test_today_and_tomorrow():
    assert resolve_relative_date("today", REF_SUNDAY) == "2026-10-04"
    assert resolve_relative_date("tomorrow", REF_SUNDAY) == "2026-10-05"


def test_iso_dates():
    assert resolve_relative_date("2026-10-15", REF_SUNDAY) == "2026-10-15"
    with pytest.raises(ValueError, match="in the past"):
        resolve_relative_date("2026-10-01", REF_SUNDAY)


def test_in_n_days():
    assert resolve_relative_date("in 3 days", REF_SUNDAY) == "2026-10-07"


def test_this_weekday_when_today_is_sunday():
    # Today is Sunday: "this Sunday" is today
    assert resolve_relative_date("this sunday", REF_SUNDAY) == "2026-10-04"
    # "this monday" is tomorrow
    assert resolve_relative_date("this monday", REF_SUNDAY) == "2026-10-05"
    # "this saturday" is upcoming Saturday
    assert resolve_relative_date("this saturday", REF_SUNDAY) == "2026-10-10"


def test_next_weekday_when_today_is_sunday():
    # Today is Sunday: "next Sunday" is 7 days ahead (strictly future)
    assert resolve_relative_date("next sunday", REF_SUNDAY) == "2026-10-11"
    # "next monday" is upcoming Monday (+1 day)
    assert resolve_relative_date("next monday", REF_SUNDAY) == "2026-10-05"
    # "next saturday" is upcoming Saturday (+6 days)
    assert resolve_relative_date("next saturday", REF_SUNDAY) == "2026-10-10"


def test_ambiguous_saturday_when_today_is_saturday():
    # Critical ambiguity edge case: said ON a Saturday (2026-10-10)
    # "this saturday" is today (2026-10-10)
    assert resolve_relative_date("this saturday", REF_SATURDAY) == "2026-10-10"
    # "next saturday" must NOT be today, it must be the upcoming Saturday 7 days away (2026-10-17)
    assert resolve_relative_date("next saturday", REF_SATURDAY) == "2026-10-17"


def test_past_or_invalid_expressions():
    with pytest.raises(ValueError, match="in the past"):
        resolve_relative_date("yesterday", REF_SUNDAY)
    with pytest.raises(ValueError, match="Unrecognized"):
        resolve_relative_date("some someday", REF_SUNDAY)
