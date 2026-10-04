"""Deterministic stand-in for a real LLM, used when no API key is available.

``RuleBasedFakeLLM.chat(messages, tools)`` decides its reply ONLY from the
messages it is sent - it keeps no state of its own. That makes it a fair probe
of conversation memory: if the server failed to send prior turns, the fake
would "forget" the pet name / service exactly like a real model would.

It is deliberately simple (keyword/regex extraction), covers only the booking
flow used in the demo/tests, and is NOT a substitute for evaluating a real model.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from openai.types.chat import ChatCompletionMessage

_SERVICE_KEYWORDS = [
    ("grooming", "grooming"), ("groom", "grooming"),
    ("dog walk", "dog_walk"), ("walk", "dog_walk"),
    ("vet", "vet_home_visit"),
    ("day care", "day_care"), ("daycare", "day_care"),
]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PET_PATTERNS = [
    r"(?:dog|cat|pet|puppy)(?:'s|s)? name is ([A-Z][a-zA-Z]+)",
    r"(?:named|called) ([A-Z][a-zA-Z]+)",
    r"\bfor my (?:dog|cat|pet|puppy) ([A-Z][a-zA-Z]+)",
]
_AFFIRMATIVE = re.compile(r"^\s*(yes|yep|yeah|confirm|confirmed|sure|go ahead|please do)\b", re.I)


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
        for text in user_texts:  # later turns override earlier ones
            low = text.lower()
            for kw, sid in _SERVICE_KEYWORDS:
                if kw in low:
                    facts["service"] = sid
                    break
            for pat in _PET_PATTERNS:
                if m := re.search(pat, text):
                    facts["pet"] = m.group(1)
                    break
            if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text):
                facts["date"] = m.group(1)
            elif "tomorrow" in low:
                facts["date"] = (today + timedelta(days=1)).isoformat()
            else:
                for i, wd in enumerate(_WEEKDAYS):
                    if re.search(rf"\b{wd}\b", low):
                        ahead = (i - today.weekday()) % 7 or 7
                        facts["date"] = (today + timedelta(days=ahead)).isoformat()
                        break
            if m := re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low):
                hour = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
                facts["slot"] = f"{hour:02d}:{m.group(2) or '00'}"
            elif m := re.search(r"\b([01]\d|2[0-3]):([0-5]\d)\b", low):
                facts["slot"] = f"{m.group(1)}:{m.group(2)}"
        return facts

    @staticmethod
    def _tool_results_since_last_user(messages: list[dict]) -> dict[str, dict]:
        names: dict[str, str] = {}
        results: dict[str, dict] = {}
        for m in messages:
            if m["role"] == "user":
                results = {}
            elif m["role"] == "assistant":
                for tc in m.get("tool_calls", []):
                    names[tc["id"]] = tc["function"]["name"]
            elif m["role"] == "tool":
                results[names[m["tool_call_id"]]] = json.loads(m["content"])
        return results

    # ---------------------------------------------------------------- policy
    def chat(self, messages: list[dict], tools: list[dict]) -> ChatCompletionMessage:
        self.calls += 1
        today = self._today(messages)
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        facts = self._facts(user_texts, today)
        results = self._tool_results_since_last_user(messages)
        last_assistant = next(
            (m.get("content") or "" for m in reversed(messages)
             if m["role"] == "assistant" and m.get("content")), "")

        if "create_booking" in results:
            r = results["create_booking"]
            if not r["ok"]:
                return _msg(f"Sorry, I couldn't book that: {r['error']}")
            b = r["result"]
            return _msg(f"Done! {b['petName']} is booked for {b['serviceId'].replace('_', ' ')} "
                        f"on {b['date']} at {_h12(b['slot'])}. Booking ID: {b['bookingId']}.")

        if "check_slots" in results:
            r = results["check_slots"]
            if not r["ok"]:
                return _msg(f"Sorry: {r['error']}")
            services = {s["serviceId"]: s for s in results.get("list_services", {}).get("result", [])}
            svc = services.get(facts["service"], {"name": facts["service"], "priceInr": "?"})
            day = date.fromisoformat(facts["date"])
            if facts["slot"] in r["result"]:
                return _msg(f"{svc['name']} for {facts['pet']} on {day:%A, %d %B %Y} at "
                            f"{_h12(facts['slot'])} costs Rs {svc['priceInr']}. Shall I book it?")
            offered = ", ".join(_h12(s) for s in r["result"]) or "none"
            return _msg(f"{_h12(facts['slot'])} isn't available that day. Available: {offered}.")

        latest = user_texts[-1]
        if re.search(r"pet'?s name|what did i", latest, re.I):
            if facts["pet"] or facts["service"]:
                return _msg(f"So far: pet={facts['pet'] or 'not given'}, "
                            f"service={facts['service'] or 'not given'}.")
            return _msg("You haven't told me your pet's name or what you'd like to book in this "
                        "conversation yet. I can help with dog walks, grooming, vet home visits and day care.")

        if _AFFIRMATIVE.match(latest) and "Shall I book it?" in last_assistant and all(facts.values()):
            return _msg(tool_calls=[("create_booking", {
                "pet_name": facts["pet"], "service_id": facts["service"],
                "date": facts["date"], "slot": facts["slot"]})])

        if facts["service"] is None:
            return _msg("I can only help with PawCare pet services: dog walks, grooming, "
                        "vet home visits and day care. What would you like to book?")

        missing = [label for key, label in
                   (("pet", "your pet's name"), ("date", "the date"), ("slot", "the preferred time"))
                   if not facts[key]]
        if missing:
            return _msg(f"Happy to help with {facts['service'].replace('_', ' ')}! "
                        f"Could you tell me {', '.join(missing)}?")

        return _msg(tool_calls=[("list_services", {}),
                                ("check_slots", {"service_id": facts["service"], "date": facts["date"]})])
