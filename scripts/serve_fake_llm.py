"""Run the REAL FastAPI app under uvicorn, but with the LLM replaced by the
deterministic ``tests.fake_llm.RuleBasedFakeLLM`` (for environments without an
API key). Everything else - HTTP, session store, tool dispatch, bookings - is
the production code path.

    python scripts/serve_fake_llm.py --port 8765

With a real key in .env, use the normal server instead:  uvicorn app.main:app
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from app.main import app, get_llm_client  # noqa: E402
from tests.fake_llm import RuleBasedFakeLLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    fake = RuleBasedFakeLLM()
    app.dependency_overrides[get_llm_client] = lambda: fake
    print("[serve_fake_llm] LLM dependency overridden with RuleBasedFakeLLM", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
