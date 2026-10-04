"""Deterministic date-resolution helper for relative date expressions.

Converts phrases like "today", "tomorrow", "this <weekday>", "next <weekday>"
to ISO date strings (YYYY-MM-DD) given a reference date.

Convention:
- "today": the reference date itself.
- "tomorrow": reference date + 1 day.
- "this <weekday>": the nearest upcoming occurrence of <weekday> (if reference date
  is that weekday, "this <weekday>" is the reference date itself).
- "next <weekday>": the upcoming occurrence of <weekday> strictly in the future (> reference date).
  If reference date is already that weekday, "next <weekday>" is exactly 7 days later (+7 days).
  If reference date is a different weekday, "next <weekday>" is the upcoming occurrence.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def resolve_relative_date(expression: str, reference_date: date | str) -> str:
    """Resolve a relative date expression to an ISO YYYY-MM-DD date string.

    Raises ValueError if expression cannot be parsed or if the date would be in the past.
    """
    ref = date.fromisoformat(reference_date) if isinstance(reference_date, str) else reference_date
    expr = expression.strip().lower()

    # Exact ISO date
    if re.match(r"^\d{4}-\d{2}-\d{2}$", expr):
        target = date.fromisoformat(expr)
        if target < ref:
            raise ValueError(f"Date {target.isoformat()} is in the past relative to {ref.isoformat()}.")
        return target.isoformat()

    if expr in ("today", "now"):
        return ref.isoformat()

    if expr == "tomorrow":
        return (ref + timedelta(days=1)).isoformat()

    if expr == "yesterday":
        target = ref - timedelta(days=1)
        raise ValueError(f"Yesterday ({target.isoformat()}) is in the past.")

    # Match "in N days"
    m_in_days = re.match(r"^in\s+(\d+)\s+days?$", expr)
    if m_in_days:
        days = int(m_in_days.group(1))
        return (ref + timedelta(days=days)).isoformat()

    # Match "this <weekday>", "next <weekday>", or plain "<weekday>"
    m_wd = re.match(r"^(?:(this|next)\s+)?([a-z]+)$", expr)
    if m_wd:
        prefix, day_name = m_wd.group(1), m_wd.group(2)
        if day_name in WEEKDAYS:
            target_weekday = WEEKDAYS[day_name]
            ref_weekday = ref.weekday()

            if prefix == "this":
                # Nearest upcoming or today
                days_ahead = (target_weekday - ref_weekday) % 7
                return (ref + timedelta(days=days_ahead)).isoformat()

            if prefix == "next":
                # Strictly in the future: if today is that weekday, jump 7 days
                days_ahead = (target_weekday - ref_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (ref + timedelta(days=days_ahead)).isoformat()

            # Plain weekday name without prefix: treat as nearest upcoming strictly in the future (or today if today)
            days_ahead = (target_weekday - ref_weekday) % 7
            return (ref + timedelta(days=days_ahead)).isoformat()

    raise ValueError(f"Unrecognized date expression: '{expression}'")
