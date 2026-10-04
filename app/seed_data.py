"""Static seed data: the PawCare service catalog and its booking rules.

The rules are expressed as data (``BookingRule``) so ``tools.py`` can apply them
generically instead of hard-coding per-service ``if`` branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta, timezone

from app.models import Service

# PawCare operates in India (prices in INR). IST has no DST, so a fixed offset is
# exact and avoids needing the tzdata package on Windows.
BUSINESS_TZ = timezone(timedelta(hours=5, minutes=30), name="IST")

# Operating window "9 AM - 6 PM": hourly start times 09:00 ... 17:00.
# 6 PM is treated as closing time, so the last hourly slot *starts* at 17:00.
OPEN_HOUR = 9
CLOSE_HOUR = 18
HOURLY_START_HOURS: tuple[int, ...] = tuple(range(OPEN_HOUR, CLOSE_HOUR))

# datetime.date.weekday(): Monday == 0 ... Sunday == 6
ALL_DAYS: frozenset[int] = frozenset(range(7))
WEEKDAYS: frozenset[int] = frozenset(range(5))
SATURDAY = 5


@dataclass(frozen=True)
class BookingRule:
    """When a service can start.

    start_hours:  candidate hourly start times (24h clock).
    days:         weekdays (0=Mon) on which the service runs at all.
    full_after:   {weekday: hour} -> on that weekday, slots starting strictly
                  after ``hour`` are full/unavailable.
    """

    start_hours: tuple[int, ...]
    days: frozenset[int] = ALL_DAYS
    full_after: dict[int, int] = field(default_factory=dict)


SERVICES: list[Service] = [
    Service(
        service_id="dog_walk",
        name="Dog Walk",
        duration_minutes=30,
        price_inr=300,
        rule="Hourly slots 9 AM-6 PM, every day.",
    ),
    Service(
        service_id="grooming",
        name="Grooming",
        duration_minutes=90,
        price_inr=900,
        rule="Hourly slots 9 AM-6 PM. On Saturdays, slots after 3 PM are full.",
    ),
    Service(
        service_id="vet_home_visit",
        name="Vet Home Visit",
        duration_minutes=60,
        price_inr=1200,
        rule="Hourly slots 9 AM-6 PM, weekdays (Mon-Fri) only.",
    ),
    Service(
        service_id="day_care",
        name="Day Care",
        duration_minutes=8 * 60,
        price_inr=1500,
        rule="Starts at 9 AM only, every day.",
    ),
]

BOOKING_RULES: dict[str, BookingRule] = {
    "dog_walk": BookingRule(start_hours=HOURLY_START_HOURS),
    "grooming": BookingRule(start_hours=HOURLY_START_HOURS, full_after={SATURDAY: 15}),
    "vet_home_visit": BookingRule(start_hours=HOURLY_START_HOURS, days=WEEKDAYS),
    "day_care": BookingRule(start_hours=(9,)),
}

# Guard against the catalog and the rules drifting apart.
if {s.service_id for s in SERVICES} != set(BOOKING_RULES):
    raise RuntimeError("seed_data: SERVICES and BOOKING_RULES define different service ids")
