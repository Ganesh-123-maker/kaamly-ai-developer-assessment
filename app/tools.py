"""The 5 PawCare tools as plain, in-memory Python functions.

Phase 2 wraps these for LLM tool calling. All user-facing failures raise a
subclass of ``PawCareError`` (itself a ``ValueError``) with a message that is
safe and useful to show to the model / end user.
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import date as Date
from datetime import datetime

from app.models import Booking, BookingStatus, Service
from app.seed_data import BOOKING_RULES, BUSINESS_TZ, SERVICES


# --------------------------------------------------------------------------- errors
class PawCareError(ValueError):
    """Base class for all expected, user-facing tool errors."""


class InvalidServiceError(PawCareError):
    pass


class InvalidDateError(PawCareError):
    pass


class InvalidSlotError(PawCareError):
    pass


class InvalidPetNameError(PawCareError):
    pass


class SlotUnavailableError(PawCareError):
    pass


class BookingNotFoundError(PawCareError):
    pass


# --------------------------------------------------------------------------- store
# booking_id -> Booking. Cancelled bookings are kept (status=cancelled) for history.
_BOOKINGS: dict[str, Booking] = {}
# Makes "check availability + insert" atomic if tools are called from a threadpool.
_LOCK = threading.RLock()

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLOT_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_MAX_PET_NAME_LEN = 50


def reset_bookings() -> None:
    """Clear the in-memory store (test helper)."""
    with _LOCK:
        _BOOKINGS.clear()


def _now() -> datetime:
    """Current business-local time. Isolated so tests can freeze the clock."""
    return datetime.now(BUSINESS_TZ)


# --------------------------------------------------------------------------- validation
def _validate_service_id(service_id: object) -> str:
    valid = ", ".join(sorted(BOOKING_RULES))
    if not isinstance(service_id, str) or not service_id.strip():
        raise InvalidServiceError(f"service_id is required. Valid service ids: {valid}.")
    normalized = service_id.strip().lower()
    if normalized not in BOOKING_RULES:
        raise InvalidServiceError(
            f"Unknown service_id '{service_id}'. Valid service ids: {valid}."
        )
    return normalized


def _parse_date(value: object) -> Date:
    # Python 3.11's fromisoformat also accepts '20261005', '2026-W41-1', ...; the
    # tool contract is strictly YYYY-MM-DD, so check the shape first.
    if not isinstance(value, str) or not _DATE_RE.match(value.strip()):
        raise InvalidDateError(f"Invalid date '{value}'. Expected ISO format YYYY-MM-DD.")
    try:
        day = Date.fromisoformat(value.strip())
    except ValueError:
        raise InvalidDateError(f"Invalid date '{value}': no such calendar date.") from None
    today = _now().date()
    if day < today:
        raise InvalidDateError(
            f"Date {day.isoformat()} is in the past (today is {today.isoformat()})."
        )
    return day


def _normalize_slot(value: object) -> str:
    """Accept 'HH:MM' (or 'H:MM') 24h time and return zero-padded 'HH:MM'."""
    match = _SLOT_RE.match(value.strip()) if isinstance(value, str) else None
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour < 24 and minute < 60:
            return f"{hour:02d}:{minute:02d}"
    raise InvalidSlotError(f"Invalid slot '{value}'. Expected 24h time HH:MM, e.g. '09:00'.")


def _validate_pet_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidPetNameError("pet_name is required.")
    name = " ".join(value.split())
    if len(name) > _MAX_PET_NAME_LEN:
        raise InvalidPetNameError(f"pet_name must be at most {_MAX_PET_NAME_LEN} characters.")
    return name


# --------------------------------------------------------------------------- slot logic
def _rule_slots(service_id: str, day: Date) -> list[str]:
    """Slots allowed by the service's rule alone (ignores bookings and the clock)."""
    rule = BOOKING_RULES[service_id]
    weekday = day.weekday()
    if weekday not in rule.days:
        return []
    cutoff = rule.full_after.get(weekday)
    return [
        f"{hour:02d}:00"
        for hour in rule.start_hours
        if cutoff is None or hour <= cutoff
    ]


def _booked_slots(service_id: str, day: Date) -> set[str]:
    return {
        b.slot
        for b in _BOOKINGS.values()
        if b.service_id == service_id and b.date == day and b.status is BookingStatus.CONFIRMED
    }


def _available_slots(service_id: str, day: Date) -> list[str]:
    booked = _booked_slots(service_id, day)
    now = _now()
    slots = [s for s in _rule_slots(service_id, day) if s not in booked]
    if day == now.date():
        # Don't offer slots that have already started today.
        slots = [s for s in slots if s > now.strftime("%H:%M")]
    return slots


# --------------------------------------------------------------------------- tools
def list_services() -> list[Service]:
    """Return the full service catalog."""
    return list(SERVICES)


def check_slots(service_id: str, date: str) -> list[str]:
    """Return available 'HH:MM' start times for ``service_id`` on ``date`` (YYYY-MM-DD).

    Applies the service's rule, removes already-booked slots and, for today,
    slots whose start time has passed. Raises on unknown service / bad date.
    """
    sid = _validate_service_id(service_id)
    day = _parse_date(date)
    with _LOCK:
        return _available_slots(sid, day)


def create_booking(
    pet_name: str,
    service_id: str,
    date: str,
    slot: str,
    session_id: str | None = None,
) -> Booking:
    """Book ``slot`` after re-validating it is actually available right now."""
    name = _validate_pet_name(pet_name)
    sid = _validate_service_id(service_id)
    day = _parse_date(date)
    time = _normalize_slot(slot)

    with _LOCK:  # availability check + insert must be atomic
        available = _available_slots(sid, day)
        if time not in available:
            if time in _booked_slots(sid, day):
                reason = "is already booked"
            elif time in _rule_slots(sid, day):
                reason = "has already started"
            else:
                reason = "is not offered for this service on that date"
            alternatives = ", ".join(available) if available else "none"
            raise SlotUnavailableError(
                f"Slot {time} on {day.isoformat()} for '{sid}' {reason}. "
                f"Currently available slots: {alternatives}."
            )

        booking = Booking(
            booking_id=str(uuid.uuid4()),
            pet_name=name,
            service_id=sid,
            date=day,
            slot=time,
            status=BookingStatus.CONFIRMED,
            session_id=session_id,
            created_at=_now(),
        )
        _BOOKINGS[booking.booking_id] = booking
        return booking


def get_bookings(session_id: str | None = None) -> list[Booking]:
    """Return bookings (incl. cancelled ones) for ``session_id``, or all if None.

    Sorted by date then slot.
    """
    with _LOCK:
        bookings = [
            b for b in _BOOKINGS.values() if session_id is None or b.session_id == session_id
        ]
    return sorted(bookings, key=lambda b: (b.date, b.slot, b.created_at))


def cancel_booking(booking_id: str, session_id: str | None = None) -> bool:
    """Cancel a booking.

    Returns True if it was cancelled now, False if it was already cancelled.
    Raises BookingNotFoundError if the id doesn't exist (or, when ``session_id``
    is given, belongs to another session - reported identically so ids from
    other sessions can't be probed).
    """
    key = booking_id.strip() if isinstance(booking_id, str) else ""
    with _LOCK:
        booking = _BOOKINGS.get(key)
        if booking is None or (session_id is not None and booking.session_id != session_id):
            raise BookingNotFoundError(f"No booking found with id '{booking_id}'.")
        if booking.status is BookingStatus.CANCELLED:
            return False
        _BOOKINGS[key] = booking.model_copy(update={"status": BookingStatus.CANCELLED})
        return True
