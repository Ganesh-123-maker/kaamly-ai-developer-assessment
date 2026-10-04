"""In-memory per-session conversation history.

Each session's history is the full OpenAI-format message list:
``[system, user, assistant(tool_calls), tool, ..., assistant(reply), user, ...]``.
The system message at index 0 is re-rendered on every turn (fresh current date),
everything after it is preserved verbatim.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

Message = dict[str, Any]


class SessionStore:
    def __init__(self) -> None:
        self._histories: dict[str, list[Message]] = {}
        self._turn_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get_history(self, session_id: str) -> list[Message]:
        """Return a deep copy of the session's history ([] if new)."""
        with self._guard:
            return copy.deepcopy(self._histories.get(session_id, []))

    def save_history(self, session_id: str, history: list[Message]) -> None:
        with self._guard:
            self._histories[session_id] = copy.deepcopy(history)

    @contextmanager
    def turn(self, session_id: str) -> Iterator[None]:
        """Serialize turns within one session (concurrent sessions run in parallel)."""
        with self._guard:
            lock = self._turn_locks.setdefault(session_id, threading.Lock())
        with lock:
            yield

    def session_ids(self) -> list[str]:
        with self._guard:
            return list(self._histories)

    def reset(self, session_id: str | None = None) -> None:
        with self._guard:
            if session_id is None:
                self._histories.clear()
                self._turn_locks.clear()
            else:
                self._histories.pop(session_id, None)
                self._turn_locks.pop(session_id, None)


# Process-wide default store used by the API.
SESSION_STORE = SessionStore()
