"""Multi-turn + session-isolation demo against a running PawCare server over HTTP.

    python scripts/multi_turn_demo.py                      # http://127.0.0.1:8765
    python scripts/multi_turn_demo.py http://127.0.0.1:8000

Works the same against the real-LLM server (uvicorn app.main:app) or the
fake-LLM one (scripts/serve_fake_llm.py). Session ids are randomised per run.
"""

from __future__ import annotations

import json
import sys
import uuid

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"


def post_chat(client: httpx.Client, session_id: str, message: str) -> dict:
    body = {"sessionId": session_id, "message": message}
    print(f"\n>>> POST /chat {json.dumps(body)}")
    resp = client.post("/chat", json=body)
    print(f"<<< {resp.status_code} {json.dumps(resp.json(), indent=2)}")
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    run = uuid.uuid4().hex[:6]
    session_a, session_b = f"demo-A-{run}", f"demo-B-{run}"
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        health = client.get("/health")
        print(f">>> GET /health\n<<< {health.status_code} {health.json()}")

        print(f"\n======== Session A ({session_a}): 3-turn booking ========")
        post_chat(client, session_a, "I want to book grooming")
        post_chat(client, session_a, "My dog's name is Bruno. Next Saturday at 11am please.")
        post_chat(client, session_a, "Yes, please confirm")

        print(f"\n======== Session B ({session_b}): must not see session A ========")
        post_chat(client, session_b, "What's my pet's name and what did I want to book?")
        post_chat(client, session_b, "Book it for me")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
