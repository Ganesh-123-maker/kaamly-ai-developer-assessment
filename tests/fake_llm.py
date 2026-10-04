"""Deterministic stand-in for a real LLM, used when no API key is available.

``RuleBasedFakeLLM.chat(messages, tools)`` decides its reply ONLY from the
messages it is sent - it keeps no state of its own. That makes it a fair probe
of conversation memory and tool dispatch.

Hardened in Phase 4 to:
- Use `app.date_utils.resolve_relative_date` with strict conventions.
- Check slots when service and date are known even if pet/slot are still missing.
- When `check_slots` returns empty for a weekday-only service (like vet visit on Sunday),
  automatically call `check_slots` for the upcoming weekday to offer real verified alternatives.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from openai.types.chat import ChatCompletionMessage

from app.date_utils import resolve_relative_date

_SERVICE_KEYWORDS = [
    ("grooming", "grooming"), ("groom", "grooming"),
    ("dog walk", "dog_walk"), ("walk", "dog_walk"),
    ("vet", "vet_home_visit"),
    ("day care", "day_care"), ("daycare", "day_care"),
]
_PET_PATTERNS = [
    r"(?:dog|cat|pet|puppy)(?:'s|s)? name is ([A-Z][a-zA-Z]+)",
    r"(?:named|called) ([A-Z][a-zA-Z]+)",
    r"\bfor (?:my )?(?:dog|cat|pet|puppy )?([A-Z][a-zA-Z]+)",
]
_AFFIRMATIVE = re.compile(r"^\s*(yes|yep|yeah|confirm|confirmed|sure|go ahead|please do|please book)\b", re.I)


def _msg(content: str | None = None, tool_calls: list[tuple[str, dict]] | None = None):
    payload: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        payload["tool_calls"] = [
            {"id": f"fake_{name}_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return ChatCompletionMessage.model_validate(payload)


def _h12(slot: str) -> str:
    hour = int(slot[:2])
    return f"{(hour - 1) % 12 + 1}:{slot[3:]} {'AM' if hour < 12 else 'PM'}"


class RuleBasedFakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    # ---------------------------------------------------------------- extraction
    @staticmethod
    def _today(messages: list[dict]) -> date:
        m = re.search(r"Today is \w+, (\d{4}-\d{2}-\d{2})", messages[0]["content"])
        return date.fromisoformat(m.group(1))

    @staticmethod
    def _facts(user_texts: list[str], today: date) -> dict[str, str | None]:
        facts: dict[str, str | None] = {"service": None, "pet": None, "date": None, "slot": None}
        for text in user_texts:
            low = text.lower()
            for kw, sid in _SERVICE_KEYWORDS:
                if kw in low:
                    facts["service"] = sid
                    break
            for pat in _PET_PATTERNS:
                if m := re.search(pat, text):
                    facts["pet"] = m.group(1)
                    break

            # Date resolution
            if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text):
                facts["date"] = m.group(1)
            elif "tomorrow" in low:
                facts["date"] = resolve_relative_date("tomorrow", today)
            elif "today" in low:
                facts["date"] = resolve_relative_date("today", today)
            else:
                m_rel = re.search(r"\b(this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low)
                if m_rel:
                    facts["date"] = resolve_relative_date(m_rel.group(0), today)
                else:
                    m_plain = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low)
                    if m_plain:
                        facts["date"] = resolve_relative_date(m_plain.group(1), today)

            # Slot time resolution
            if m := re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low):
                hour = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
                facts["slot"] = f"{hour:02d}:{m.group(2) or '00'}"
            elif m := re.search(r"\b([01]\d|2[0-3]):([0-5]\d)\b", low):
                facts["slot"] = f"{m.group(1)}:{m.group(2)}"
        return facts

    @staticmethod
    def _tool_results_since_last_user(messages: list[dict]) -> tuple[dict[str, dict], list[dict]]:
        """Extract tool results that were generated in response to the latest user message."""
        names: dict[str, str] = {}
        results: dict[str, dict] = {}
        results_list: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                results = {}
                results_list = []
            elif m["role"] == "assistant":
                for tc in m.get("tool_calls", []):
                    names[tc["id"]] = tc["function"]["name"]
            elif m["role"] == "tool":
                res = json.loads(m["content"])
                t_name = names.get(m["tool_call_id"], "unknown")
                results[t_name] = res
                results_list.append({"name": t_name, "content": res})
        return results, results_list

    # ---------------------------------------------------------------- policy
    def chat(self, messages: list[dict], tools: list[dict]) -> ChatCompletionMessage:
        self.calls += 1
        today = self._today(messages)
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        facts = self._facts(user_texts, today)
        results, results_list = self._tool_results_since_last_user(messages)
        last_assistant = next(
            (m.get("content") or "" for m in reversed(messages)
             if m["role"] == "assistant" and m.get("content")), "")

        latest_user = user_texts[-1]

        # 1. Did we just invoke create_booking in this turn?
        if "create_booking" in results:
            r = results["create_booking"]
            if not r["ok"]:
                return _msg(f"Sorry, I couldn't book that: {r['error']}")
            b = r["result"]
            return _msg(f"Done! {b['petName']} is booked for {b['serviceId'].replace('_', ' ')} "
                        f"on {b['date']} at {_h12(b['slot'])}. Booking ID: {b['bookingId']}.")

        # 2. Did we just check slots in this turn?
        if "check_slots" in results:
            # Special case: check if this was a weekend vet check that returned empty
            if facts["service"] == "vet_home_visit" and facts["date"]:
                vet_day = date.fromisoformat(facts["date"])
                if vet_day.weekday() in (5, 6):  # Sat or Sun
                    monday_date = resolve_relative_date("next monday", today)
                    # Have we checked Monday yet in this turn?
                    checked_monday = any(
                        r["name"] == "check_slots" and r["content"].get("result")
                        for r in results_list
                    )
                    # If this was the Sunday call (result empty), call Monday!
                    if not results["check_slots"].get("result"):
                        return _msg(tool_calls=[("check_slots", {"service_id": "vet_home_visit", "date": monday_date})])
                    else:
                        # This is the Monday result
                        monday_slots = results["check_slots"].get("result", [])
                        offered = ", ".join(_h12(s) for s in monday_slots[:6]) + "..."
                        return _msg(
                            f"Vet Home Visits are available on weekdays only, so {vet_day:%A, %d %B %Y} has no available slots. "
                            f"However, I checked Monday ({monday_date}) and slots are available: {offered}. "
                            f"Would you like to book one of those? (Please also share your pet's name and preferred time)."
                        )

            r = results["check_slots"]
            if not r["ok"]:
                return _msg(f"Sorry: {r['error']}")
            services = {s["serviceId"]: s for s in results.get("list_services", {}).get("result", [])}
            svc = services.get(facts["service"], {"name": facts["service"].replace('_', ' ').title(), "priceInr": 300})
            day = date.fromisoformat(facts["date"])

            if facts["slot"] and facts["slot"] in r.get("result", []):
                pet_phrase = f" for {facts['pet']}" if facts["pet"] else ""
                return _msg(f"{svc['name']}{pet_phrase} on {day:%A, %d %B %Y} at "
                            f"{_h12(facts['slot'])} costs Rs {svc['priceInr']}. Shall I book it?")

            if r.get("result"):
                offered = ", ".join(_h12(s) for s in r["result"])
                missing_info = []
                if not facts["pet"]:
                    missing_info.append("your pet's name")
                if not facts["slot"]:
                    missing_info.append("a preferred time slot")
                missing_str = f" Please also tell me {', '.join(missing_info)}." if missing_info else ""
                return _msg(f"Available slots for {svc['name']} on {day:%A, %d %B %Y} are: {offered}.{missing_str}")
            else:
                return _msg(f"No slots are available for {svc['name']} on {day:%A, %d %B %Y}.")

        # 3. Memory inspection / isolation query
        if re.search(r"pet'?s name|what did i", latest_user, re.I):
            if facts["pet"] or facts["service"]:
                return _msg(f"So far: pet={facts['pet'] or 'not given'}, "
                            f"service={facts['service'] or 'not given'}.")
            return _msg("You haven't told me your pet's name or what you'd like to book in this "
                        "conversation yet. I can help with dog walks, grooming, vet home visits and day care.")

        # 4. Handle explicit confirmation
        if _AFFIRMATIVE.match(latest_user) and "Shall I book it?" in last_assistant and all(facts.values()):
            return _msg(tool_calls=[("create_booking", {
                "pet_name": facts["pet"], "service_id": facts["service"],
                "date": facts["date"], "slot": facts["slot"]})])

        # 5. Scope check
        if facts["service"] is None:
            return _msg("I can only help with PawCare pet services: dog walks, grooming, "
                        "vet home visits and day care. What would you like to book?")

        # 6. Do we have service and date? If so, check slots even if pet or time is missing!
        if facts["service"] and facts["date"] and "check_slots" not in results:
            tools_to_call = [("check_slots", {"service_id": facts["service"], "date": facts["date"]})]
            if "list_services" not in results:
                tools_to_call.insert(0, ("list_services", {}))
            return _msg(tool_calls=tools_to_call)

        # 7. If missing details, ask for them (DO NOT call create_booking)
        missing = [label for key, label in
                   (("pet", "your pet's name"), ("date", "the date"), ("slot", "the preferred time"))
                   if not facts[key]]
        if missing:
            return _msg(f"Happy to help with {facts['service'].replace('_', ' ')}! "
                        f"Could you tell me {', '.join(missing)}?")

        return _msg(tool_calls=[("list_services", {}),
                                ("check_slots", {"service_id": facts["service"], "date": facts["date"]})])
