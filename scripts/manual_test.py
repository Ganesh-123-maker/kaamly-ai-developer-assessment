"""Manual end-to-end check of single-turn tool calling against the configured LLM.

Not part of pytest. Usage (from the project root, with .env filled in):

    python scripts/manual_test.py
    python scripts/manual_test.py "Can I book a vet visit on Sunday?"
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent_loop import run_turn  # noqa: E402
from app.llm_client import LLMClient, LLMConfigError  # noqa: E402
from app.seed_data import BUSINESS_TZ  # noqa: E402

DEFAULT_MESSAGES = [
    "What services do you offer and how much do they cost?",  # expect list_services
    "Is grooming available next Saturday?",                   # expect check_slots
    "What's the capital of France?",                           # expect NO tool call
    "Book a dog walk for tomorrow at 10am",                    # expect: asks for pet name, no booking
]


def main() -> int:
    messages = sys.argv[1:] or DEFAULT_MESSAGES
    try:
        client = LLMClient()
    except LLMConfigError as exc:
        print(f"[config error] {exc}")
        return 1

    # The caller (not run_turn) decides "today"; the script uses the real IST date.
    current_date = datetime.now(BUSINESS_TZ).date().isoformat()
    print(f"Provider: {client.config!r}\nCurrent date: {current_date}\n")

    for i, text in enumerate(messages, 1):
        print("=" * 78)
        print(f"[{i}] USER: {text}")
        result = run_turn(text, current_date, client=client)
        if result["toolCalls"]:
            for call in result["toolCalls"]:
                print(f"  -> TOOL {call['name']}({json.dumps(call['arguments'])})")
                print(f"     RESULT {json.dumps(call['result'])[:300]}")
        else:
            print("  -> (no tool calls)")
        print(f"ASSISTANT: {result['reply']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
