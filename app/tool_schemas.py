"""OpenAI function-calling schemas for the 5 tools in ``app/tools.py``.

Each schema's parameters mirror the Python signature exactly, EXCEPT the
optional ``session_id`` keyword on create_booking / get_bookings /
cancel_booking. That is deliberately NOT exposed to the model: which session a
request belongs to is decided by the server (injected by the agent loop), never
by the LLM - otherwise a prompt could read or cancel another user's bookings.
``tests/test_tool_schemas.py`` enforces this correspondence.

No service ids are enumerated here on purpose: the model must discover them via
list_services (tool errors also list valid ids if it guesses wrong).
"""

from __future__ import annotations

from typing import Any

# Parameters that exist on the Python functions but are injected server-side.
SERVER_INJECTED_PARAMS = frozenset({"session_id"})

_SOURCE_OF_TRUTH = (
    "Tool results are the ONLY source of truth: never state a service, price, "
    "duration or time slot that was not returned by a tool call."
)

_SERVICE_ID = {
    "type": "string",
    "description": "The serviceId exactly as returned by list_services (e.g. not the display name).",
}
_DATE = {
    "type": "string",
    "description": (
        "Date in ISO format YYYY-MM-DD. Resolve relative dates like 'tomorrow' or "
        "'next Saturday' using the current date given in the system prompt. Must not be in the past."
    ),
}


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function(
        "list_services",
        "List every PawCare service with its serviceId, name, duration (minutes), price (INR) "
        "and availability rule. Call this before mentioning any service, price or duration. "
        + _SOURCE_OF_TRUTH,
        properties={},
        required=[],
    ),
    _function(
        "check_slots",
        "Get the currently AVAILABLE start times (24h 'HH:MM') for one service on one date, "
        "with business rules applied and already-booked slots removed. An empty list means "
        "nothing is available that day. Call this before offering or confirming any time. "
        + _SOURCE_OF_TRUTH,
        properties={"service_id": _SERVICE_ID, "date": _DATE},
        required=["service_id", "date"],
    ),
    _function(
        "create_booking",
        "Create a booking. ONLY call this after the user has explicitly confirmed (said yes to) "
        "the pet name, service, date and time you summarised. The slot is re-validated; if it is "
        "no longer available an error with the current alternatives is returned. Returns the "
        "booking including its bookingId - never make up a bookingId.",
        properties={
            "pet_name": {"type": "string", "description": "The pet's name as given by the user."},
            "service_id": _SERVICE_ID,
            "date": _DATE,
            "slot": {
                "type": "string",
                "description": "Start time in 24h 'HH:MM' format, exactly as returned by check_slots (e.g. '15:00').",
            },
        },
        required=["pet_name", "service_id", "date", "slot"],
    ),
    _function(
        "get_bookings",
        "List the user's bookings (confirmed and cancelled) with bookingId, pet name, service, "
        "date, slot and status. Use this when the user asks about their bookings or wants to "
        "cancel one but doesn't know the bookingId.",
        properties={},
        required=[],
    ),
    _function(
        "cancel_booking",
        "Cancel a booking by its bookingId (from create_booking or get_bookings). Only call after "
        "the user confirms which booking to cancel. Returns true if cancelled, false if it was "
        "already cancelled; errors if no such booking exists.",
        properties={
            "booking_id": {
                "type": "string",
                "description": "The bookingId exactly as returned by a tool.",
            }
        },
        required=["booking_id"],
    ),
]

TOOL_NAMES: frozenset[str] = frozenset(s["function"]["name"] for s in TOOL_SCHEMAS)
