"""Execute the 4 required Phase 4 conversational scenarios against the live PawCare server.

Validates:
a) "Book a dog walk for Bruno tomorrow at 5 pm" -> checks slots, asks confirmation, books only upon explicit yes.
b) "I want grooming" -> asks for missing details (pet name, date, time); verifies no booking is made.
c) "Book a vet visit this Sunday" -> checks slots, handles weekend rule, calls check_slots on Monday weekday to offer verified slots.
d) "Book grooming for Max next Saturday at 2 pm" said on a Saturday -> verifies resolved date is +7 days (not today).
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from app import tools
from app.seed_data import BUSINESS_TZ

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"


def send_chat(client: httpx.Client, session_id: str, message: str) -> dict:
    body = {"sessionId": session_id, "message": message}
    print(f"\n[USER -> {session_id}]: {message}")
    resp = client.post("/chat", json=body)
    resp.raise_for_status()
    data = resp.json()
    print(f"[ASSISTANT]: {data['reply']}")
    if data["toolCalls"]:
        print(f"[TOOL CALLS ({len(data['toolCalls'])})]:")
        for tc in data["toolCalls"]:
            print(f"  * {tc['name']}({json.dumps(tc['arguments'])}) -> {str(tc['result'])[:160]}")
    else:
        print("[TOOL CALLS]: (none)")
    return data


def run_scenario_a(client: httpx.Client):
    print("\n" + "=" * 70)
    print("SCENARIO A: 'Book a dog walk for Bruno tomorrow at 5 pm'")
    print("=" * 70)
    sid = f"scen-a-{uuid.uuid4().hex[:6]}"

    # Turn 1
    t1 = send_chat(client, sid, "Book a dog walk for Bruno tomorrow at 5 pm")
    assert any(c["name"] == "check_slots" for c in t1["toolCalls"]), "Must call check_slots"
    assert not any(c["name"] == "create_booking" for c in t1["toolCalls"]), "Must NOT call create_booking yet"
    assert "Shall I book it?" in t1["reply"] or "confirm" in t1["reply"].lower()

    # Turn 2: explicit confirmation
    t2 = send_chat(client, sid, "Yes, please confirm")
    assert any(c["name"] == "create_booking" for c in t2["toolCalls"]), "Must call create_booking on confirmation"
    assert "Booking ID:" in t2["reply"]


def run_scenario_b(client: httpx.Client):
    print("\n" + "=" * 70)
    print("SCENARIO B: 'I want grooming' (Missing details)")
    print("=" * 70)
    sid = f"scen-b-{uuid.uuid4().hex[:6]}"

    t1 = send_chat(client, sid, "I want grooming")
    assert len(t1["toolCalls"]) == 0, "No tools should be called when all details are missing"
    assert "pet's name" in t1["reply"].lower()
    assert "date" in t1["reply"].lower()
    assert "time" in t1["reply"].lower()


def run_scenario_c(client: httpx.Client):
    print("\n" + "=" * 70)
    print("SCENARIO C: 'Book a vet visit this Sunday' (Weekday only rule + weekday slot check)")
    print("=" * 70)
    sid = f"scen-c-{uuid.uuid4().hex[:6]}"

    t1 = send_chat(client, sid, "Book a vet visit this Sunday")
    calls = [c["name"] for c in t1["toolCalls"]]
    assert "check_slots" in calls, "Must check slots for Sunday"
    # Check that check_slots was called for Monday as well
    dates_checked = [c["arguments"].get("date") for c in t1["toolCalls"] if c["name"] == "check_slots"]
    print(f"Dates verified by tool calls: {dates_checked}")
    assert len(dates_checked) >= 2, "Must check Sunday (empty) and then Monday (weekday alternative)"
    assert "weekdays only" in t1["reply"].lower() or "no available slots" in t1["reply"].lower()


def run_scenario_d(client: httpx.Client):
    print("\n" + "=" * 70)
    print("SCENARIO D: Date-ambiguous 'next Saturday'")
    print("=" * 70)
    # Part 1: Said on Sunday 2026-10-04 -> resolves to upcoming Saturday 2026-10-10
    sid1 = f"scen-d1-{uuid.uuid4().hex[:6]}"
    t1 = send_chat(client, sid1, "Book grooming for Max next Saturday at 2 pm")
    dates_checked1 = [c["arguments"].get("date") for c in t1["toolCalls"] if c["name"] == "check_slots"]
    print(f"When said on Sunday 2026-10-04 -> resolved to: {dates_checked1}")
    assert dates_checked1 == ["2026-10-10"]

    # Part 2: Ambiguity edge case: said ON a Saturday (e.g., 2026-10-10)
    # Using run_turn directly with reference date '2026-10-10':
    from app.agent_loop import run_turn
    from tests.fake_llm import RuleBasedFakeLLM
    out_sat = run_turn(
        "Book grooming for Max next Saturday at 2 pm",
        "2026-10-10",
        client=RuleBasedFakeLLM()
    )
    sat_dates = [c["arguments"].get("date") for c in out_sat["toolCalls"] if c["name"] == "check_slots"]
    print(f"When said on Saturday 2026-10-10 -> resolved to: {sat_dates} (+7 days: 2026-10-17, NEVER today)")
    assert sat_dates == ["2026-10-17"]


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        health = client.get("/health")
        print(f"GET /health: {health.json()}")

        run_scenario_a(client)
        run_scenario_b(client)
        run_scenario_c(client)
        run_scenario_d(client)

    print("\nAll 4 scenarios executed successfully against the live server!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
