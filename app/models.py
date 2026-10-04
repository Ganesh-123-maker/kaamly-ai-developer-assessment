"""Pydantic models for the PawCare domain: services, slots and bookings.

Python attributes are snake_case; JSON (what the LLM / API sees) is camelCase
(``bookingId``, ``petName``, ``serviceId`` ...). Models accept either form on input.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        frozen=True,  # store entries can't be mutated behind the store's back
    )


class Service(_CamelModel):
    """A bookable service from the catalog."""

    service_id: str = Field(description="Stable identifier used by all tools, e.g. 'dog_walk'.")
    name: str
    duration_minutes: int = Field(gt=0)
    price_inr: int = Field(ge=0)
    rule: str = Field(description="Human-readable availability rule.")


class Slot(_CamelModel):
    """A concrete bookable start time for a service on a given date."""

    service_id: str
    date: dt.date
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="24h start time, 'HH:MM'.")


class BookingStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class Booking(_CamelModel):
    """A booking made by a pet owner."""

    booking_id: str
    pet_name: str = Field(min_length=1)
    service_id: str
    date: dt.date
    slot: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="24h start time, 'HH:MM'.")
    status: BookingStatus = BookingStatus.CONFIRMED
    session_id: str | None = Field(
        default=None, description="Owning chat session; used to scope 'my bookings'."
    )
    created_at: dt.datetime
