"""Tests for the 5 PawCare tools (pure in-memory logic, no LLM)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pytest

from app import tools
from app.models import Booking, BookingStatus
from app.seed_data import BUSINESS_TZ
from app.tools import (
    BookingNotFoundError,
    InvalidDateError,
    InvalidPetNameError,
    InvalidServiceError,
    InvalidSlotError,
    PawCareError,
    SlotUnavailableError,
    cancel_booking,
    check_slots,
    create_booking,
    get_bookings,
    list_services,
)

# The clock is frozen at Sunday 2026-10-04 08:00 IST so day-of-week rules are deterministic.
FROZEN_NOW = datetime(2026, 10, 4, 8, 0, tzinfo=BUSINESS_TZ)
TODAY = "2026-10-04"      # Sunday
MONDAY = "2026-10-05"
TUESDAY = "2026-10-06"
WEDNESDAY = "2026-10-07"
THURSDAY = "2026-10-08"
FRIDAY = "2026-10-09"
SATURDAY = "2026-10-10"
SUNDAY = "2026-10-11"

ALL_HOURLY = [f"{h:02d}:00" for h in range(9, 18)]  # 09:00 .. 17:00


@pytest.fixture(autouse=True)
def _isolated_store_and_clock(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda: FROZEN_NOW)
    tools.reset_bookings()
    yield
    tools.reset_bookings()


def test_date_constants_have_expected_weekdays():
    days = [MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY]
    assert [date.fromisoformat(d).weekday() for d in days] == list(range(7))


# ------------------------------------------------------------------ list_services
def test_list_services_returns_all_four_with_catalog_values():
    services = {s.service_id: s for s in list_services()}
    assert len(services) == 4
    assert {sid: (s.name, s.duration_minutes, s.price_inr) for sid, s in services.items()} == {
        "dog_walk": ("Dog Walk", 30, 300),
        "grooming": ("Grooming", 90, 900),
        "vet_home_visit": ("Vet Home Visit", 60, 1200),
        "day_care": ("Day Care", 480, 1500),
    }


def test_service_serializes_with_camel_case_keys():
    dumped = list_services()[0].model_dump(mode="json")
    assert set(dumped) == {"serviceId", "name", "durationMinutes", "priceInr", "rule"}


# ------------------------------------------------------------------ check_slots rules
@pytest.mark.parametrize("day", [MONDAY, FRIDAY, SATURDAY, SUNDAY])
def test_dog_walk_hourly_every_day(day):
    assert check_slots("dog_walk", day) == ALL_HOURLY


@pytest.mark.parametrize("day", [MONDAY, WEDNESDAY, FRIDAY, SUNDAY])
def test_grooming_full_hourly_on_non_saturdays(day):
    assert check_slots("grooming", day) == ALL_HOURLY


def test_grooming_saturday_slots_after_3pm_are_full():
    slots = check_slots("grooming", SATURDAY)
    assert slots == ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]
    assert "15:00" in slots  # 3 PM itself is not "after 3 PM"
    assert not {"16:00", "17:00"} & set(slots)


@pytest.mark.parametrize("day", [MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY])
def test_vet_home_visit_available_on_weekdays(day):
    assert check_slots("vet_home_visit", day) == ALL_HOURLY


@pytest.mark.parametrize("day", [SATURDAY, SUNDAY])
def test_vet_home_visit_empty_on_weekends(day):
    assert check_slots("vet_home_visit", day) == []


@pytest.mark.parametrize("day", [MONDAY, SATURDAY, SUNDAY])
def test_day_care_only_9am(day):
    assert check_slots("day_care", day) == ["09:00"]


# ------------------------------------------------------------------ check_slots errors
@pytest.mark.parametrize("bad_id", ["spa_day", "", "   ", None, "Dog Walk"])
def test_check_slots_invalid_service_raises(bad_id):
    with pytest.raises(InvalidServiceError, match="Valid service ids: day_care, dog_walk"):
        check_slots(bad_id, MONDAY)


def test_service_id_is_case_and_whitespace_insensitive():
    assert check_slots("  DOG_WALK ", MONDAY) == ALL_HOURLY


def test_invalid_service_error_is_a_value_error():
    with pytest.raises(ValueError):
        check_slots("spa_day", MONDAY)


@pytest.mark.parametrize(
    "bad_date",
    ["2026/10/05", "05-10-2026", "20261005", "2026-W41-1", "tomorrow", "", None, "2026-10-5"],
)
def test_check_slots_malformed_date_raises(bad_date):
    with pytest.raises(InvalidDateError, match="YYYY-MM-DD"):
        check_slots("dog_walk", bad_date)


@pytest.mark.parametrize("impossible", ["2026-13-01", "2026-02-30", "2026-00-10"])
def test_check_slots_impossible_calendar_date_raises(impossible):
    with pytest.raises(InvalidDateError, match="no such calendar date"):
        check_slots("dog_walk", impossible)


def test_check_slots_past_date_raises():
    with pytest.raises(InvalidDateError, match="in the past"):
        check_slots("dog_walk", "2026-10-03")


def test_check_slots_today_hides_slots_already_started(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda: datetime(2026, 10, 4, 13, 30, tzinfo=BUSINESS_TZ))
    assert check_slots("dog_walk", TODAY) == ["14:00", "15:00", "16:00", "17:00"]
    assert check_slots("day_care", TODAY) == []


# ------------------------------------------------------------------ create_booking
def test_create_booking_succeeds_and_slot_disappears():
    booking = create_booking("Bruno", "grooming", MONDAY, "10:00", session_id="s1")

    assert isinstance(booking, Booking)
    assert uuid.UUID(booking.booking_id).version == 4
    assert booking.pet_name == "Bruno"
    assert booking.service_id == "grooming"
    assert booking.date == date(2026, 10, 5)
    assert booking.slot == "10:00"
    assert booking.status is BookingStatus.CONFIRMED
    assert booking.session_id == "s1"

    slots = check_slots("grooming", MONDAY)
    assert "10:00" not in slots
    assert len(slots) == len(ALL_HOURLY) - 1
    # Other services / other days are unaffected.
    assert check_slots("dog_walk", MONDAY) == ALL_HOURLY
    assert check_slots("grooming", TUESDAY) == ALL_HOURLY


def test_booking_serializes_with_spec_field_names():
    dumped = create_booking("Bruno", "dog_walk", MONDAY, "09:00").model_dump(mode="json")
    for key in ("bookingId", "petName", "serviceId", "date", "slot", "status"):
        assert key in dumped
    assert dumped["date"] == MONDAY and dumped["status"] == "confirmed"


def test_booking_ids_are_unique():
    a = create_booking("Bruno", "dog_walk", MONDAY, "09:00")
    b = create_booking("Max", "dog_walk", MONDAY, "10:00")
    assert a.booking_id != b.booking_id


def test_create_booking_fails_on_already_booked_slot():
    create_booking("Bruno", "dog_walk", MONDAY, "11:00")
    with pytest.raises(SlotUnavailableError, match="already booked"):
        create_booking("Max", "dog_walk", MONDAY, "11:00")
    assert len(get_bookings()) == 1


def test_create_booking_already_booked_error_lists_alternatives():
    create_booking("Bruno", "day_care", MONDAY, "09:00")
    with pytest.raises(SlotUnavailableError, match="Currently available slots: none"):
        create_booking("Max", "day_care", MONDAY, "09:00")


@pytest.mark.parametrize(
    "service_id, day, slot",
    [
        ("grooming", SATURDAY, "16:00"),     # Saturday after 3 PM
        ("vet_home_visit", SUNDAY, "10:00"),  # weekend
        ("day_care", MONDAY, "10:00"),       # day care only at 9
        ("dog_walk", MONDAY, "18:00"),       # outside hours
        ("dog_walk", MONDAY, "09:30"),       # not on the hour
    ],
)
def test_create_booking_rejects_slots_not_allowed_by_rules(service_id, day, slot):
    with pytest.raises(SlotUnavailableError, match="not offered"):
        create_booking("Bruno", service_id, day, slot)
    assert get_bookings() == []


def test_create_booking_rejects_slot_already_started_today(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda: datetime(2026, 10, 4, 13, 30, tzinfo=BUSINESS_TZ))
    with pytest.raises(SlotUnavailableError, match="already started"):
        create_booking("Bruno", "dog_walk", TODAY, "13:00")


def test_create_booking_normalizes_single_digit_hour():
    assert create_booking("Bruno", "day_care", MONDAY, "9:00").slot == "09:00"


@pytest.mark.parametrize("bad_slot", ["9am", "25:00", "10:60", "", None, "10"])
def test_create_booking_malformed_slot_raises(bad_slot):
    with pytest.raises(InvalidSlotError, match="HH:MM"):
        create_booking("Bruno", "dog_walk", MONDAY, bad_slot)


@pytest.mark.parametrize("bad_name", ["", "   ", None, "x" * 51])
def test_create_booking_invalid_pet_name_raises(bad_name):
    with pytest.raises(InvalidPetNameError):
        create_booking(bad_name, "dog_walk", MONDAY, "09:00")


def test_create_booking_invalid_service_raises():
    with pytest.raises(InvalidServiceError, match="Unknown service_id 'cat_walk'"):
        create_booking("Bruno", "cat_walk", MONDAY, "09:00")


def test_create_booking_invalid_date_raises():
    with pytest.raises(InvalidDateError):
        create_booking("Bruno", "dog_walk", "next monday", "09:00")


# ------------------------------------------------------------------ get_bookings
def test_get_bookings_scoped_by_session():
    a = create_booking("Bruno", "dog_walk", MONDAY, "09:00", session_id="alice")
    b = create_booking("Max", "dog_walk", MONDAY, "10:00", session_id="bob")
    c = create_booking("Bruno", "grooming", MONDAY, "09:00", session_id="alice")

    assert [x.booking_id for x in get_bookings("alice")] == [a.booking_id, c.booking_id]
    assert [x.booking_id for x in get_bookings("bob")] == [b.booking_id]
    assert get_bookings("nobody") == []
    assert len(get_bookings()) == 3


def test_get_bookings_sorted_by_date_then_slot():
    create_booking("A", "dog_walk", TUESDAY, "09:00")
    create_booking("B", "dog_walk", MONDAY, "15:00")
    create_booking("C", "dog_walk", MONDAY, "10:00")
    assert [(str(b.date), b.slot) for b in get_bookings()] == [
        (MONDAY, "10:00"), (MONDAY, "15:00"), (TUESDAY, "09:00"),
    ]


# ------------------------------------------------------------------ cancel_booking
def test_cancel_booking_frees_slot_back_up():
    booking = create_booking("Bruno", "vet_home_visit", MONDAY, "14:00")
    assert "14:00" not in check_slots("vet_home_visit", MONDAY)

    assert cancel_booking(booking.booking_id) is True

    assert "14:00" in check_slots("vet_home_visit", MONDAY)
    assert check_slots("vet_home_visit", MONDAY) == ALL_HOURLY
    [stored] = get_bookings()
    assert stored.status is BookingStatus.CANCELLED
    # And it can be booked again.
    rebooked = create_booking("Max", "vet_home_visit", MONDAY, "14:00")
    assert rebooked.booking_id != booking.booking_id


def test_cancel_booking_nonexistent_id_raises_clear_error():
    with pytest.raises(BookingNotFoundError, match="No booking found with id 'does-not-exist'"):
        cancel_booking("does-not-exist")


def test_cancel_booking_twice_returns_false():
    booking = create_booking("Bruno", "dog_walk", MONDAY, "09:00")
    assert cancel_booking(booking.booking_id) is True
    assert cancel_booking(booking.booking_id) is False


def test_cancel_booking_from_other_session_is_not_found():
    booking = create_booking("Bruno", "dog_walk", MONDAY, "09:00", session_id="alice")
    with pytest.raises(BookingNotFoundError):
        cancel_booking(booking.booking_id, session_id="bob")
    assert get_bookings("alice")[0].status is BookingStatus.CONFIRMED
    assert cancel_booking(booking.booking_id, session_id="alice") is True


def test_all_tool_errors_share_base_class():
    for exc in (InvalidServiceError, InvalidDateError, InvalidSlotError, InvalidPetNameError,
                SlotUnavailableError, BookingNotFoundError):
        assert issubclass(exc, PawCareError) and issubclass(exc, ValueError)


def test_stored_bookings_are_immutable():
    booking = create_booking("Bruno", "dog_walk", MONDAY, "09:00")
    with pytest.raises(Exception):
        booking.status = BookingStatus.CANCELLED  # type: ignore[misc]
    assert get_bookings()[0].status is BookingStatus.CONFIRMED
