"""Multi-turn session memory + /chat and /health endpoint tests (mocked LLM)."""

from __future__ import annotations

import copy
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app import tools
from app.agent_loop import run_turn
from app.main import app, current_server_date, get_llm_client
from app.seed_data import BUSINESS_TZ
from app.session_store import SESSION_STORE, SessionStore
from tests.fake_llm import RuleBasedFakeLLM

TODAY = "2026-10-04"  # Sunday -> "next Saturday" = 2026-10-10


class RecordingLLM:
    """Wraps another fake and records a deep copy of every request's messages."""

    def __init__(self, inner):
        self.inner = inner
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        return self.inner.chat(messages, tools)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda: datetime(2026, 10, 4, 8, 0, tzinfo=BUSINESS_TZ))
    tools.reset_bookings()
    SESSION_STORE.reset()
    yield
    tools.reset_bookings()
    SESSION_STORE.reset()
    app.dependency_overrides.clear()


@pytest.fixture
def llm():
    return RecordingLLM(RuleBasedFakeLLM())


@pytest.fixture
def client(llm):
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[current_server_date] = lambda: TODAY
    return TestClient(app)


def chat(client, session_id, message):
    resp = client.post("/chat", json={"sessionId": session_id, "message": message})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ endpoints
def test_health():
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize(
    "body", [{}, {"message": "hi"}, {"sessionId": "a"}, {"sessionId": "", "message": "hi"},
             {"sessionId": "a", "message": ""}],
)
def test_chat_validates_body(client, body):
    assert client.post("/chat", json=body).status_code == 422


def test_chat_response_shape(client):
    out = chat(client, "s1", "What services do you offer? I need grooming next Saturday at 11am "
                             "for my dog Bruno")
    assert set(out) == {"reply", "toolCalls"}
    assert [c["name"] for c in out["toolCalls"]] == ["list_services", "check_slots"]
    for call in out["toolCalls"]:
        assert set(call) == {"name", "arguments", "result"}


def test_missing_llm_config_returns_503(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.llm_client.load_dotenv", lambda: None)
    get_llm_client.cache_clear()
    resp = TestClient(app).post("/chat", json={"sessionId": "s", "message": "hi"})
    assert resp.status_code == 503
    assert "LLM_API_KEY" in resp.json()["detail"]
    get_llm_client.cache_clear()


# ------------------------------------------------------------------ multi-turn memory
def test_three_turn_booking_carries_context(client, llm):
    t1 = chat(client, "A", "I want to book grooming")
    assert "pet's name" in t1["reply"] and t1["toolCalls"] == []

    t2 = chat(client, "A", "My dog's name is Bruno. Next Saturday at 11am please.")
    assert "Shall I book it?" in t2["reply"]
    check = next(c for c in t2["toolCalls"] if c["name"] == "check_slots")
    assert check["arguments"] == {"service_id": "grooming", "date": "2026-10-10"}

    t3 = chat(client, "A", "Yes, please confirm")
    assert "pet's name" not in t3["reply"]
    [booking_call] = t3["toolCalls"]  # only this turn's tool calls are reported
    assert booking_call["name"] == "create_booking"
    assert booking_call["arguments"] == {
        "pet_name": "Bruno", "service_id": "grooming", "date": "2026-10-10", "slot": "11:00"}
    assert booking_call["result"]["result"]["bookingId"] in t3["reply"]

    # The LLM really received the full history on turn 3.
    last_request = llm.requests[-2]  # request that produced the create_booking call
    users = [m["content"] for m in last_request if m["role"] == "user"]
    assert users == ["I want to book grooming",
                     "My dog's name is Bruno. Next Saturday at 11am please.",
                     "Yes, please confirm"]

    # Booking is scoped to session A server-side.
    [stored] = tools.get_bookings("A")
    assert stored.session_id == "A" and stored.slot == "11:00"


def test_history_structure_is_saved_in_order(client):
    chat(client, "A", "I want to book grooming")
    chat(client, "A", "My dog's name is Bruno. Next Saturday at 11am please.")
    roles = [m["role"] for m in SESSION_STORE.get_history("A")]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool", "tool", "assistant"]
    assert sum(1 for r in roles if r == "system") == 1


def test_sessions_are_isolated(client, llm):
    chat(client, "A", "I want to book grooming")
    chat(client, "A", "My dog's name is Bruno. Next Saturday at 11am please.")

    out = chat(client, "B", "What's my pet's name and what did I want to book?")
    assert "Bruno" not in out["reply"] and "haven't told me" in out["reply"]
    # B's request contained only the system prompt and B's own message.
    assert [m["role"] for m in llm.requests[-1]] == ["system", "user"]
    assert all("Bruno" not in str(m) for m in llm.requests[-1])

    # Confirming in B must not book A's pending request.
    out = chat(client, "B", "Yes, please confirm")
    assert out["toolCalls"] == []
    assert tools.get_bookings() == []


def test_get_bookings_tool_is_scoped_to_session():
    tools.create_booking("Bruno", "dog_walk", "2026-10-05", "09:00", session_id="A")
    tools.create_booking("Max", "dog_walk", "2026-10-05", "10:00", session_id="B")
    from openai.types.chat import ChatCompletionMessage

    class AskBookings:
        def __init__(self):
            self.n = 0

        def chat(self, messages, tools):
            self.n += 1
            if self.n == 1:
                return ChatCompletionMessage.model_validate({"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "get_bookings", "arguments": "{}"}}]})
            return ChatCompletionMessage.model_validate({"role": "assistant", "content": "here"})

    out = run_turn("my bookings?", TODAY, session_id="B", client=AskBookings(), store=SessionStore())
    assert [b["petName"] for b in out["toolCalls"][0]["result"]["result"]] == ["Max"]


# ------------------------------------------------------------------ date + failure handling
def test_system_prompt_refreshed_with_current_date_each_turn(llm):
    store = SessionStore()
    run_turn("I want to book grooming", "2026-10-04", session_id="A", client=llm, store=store)
    run_turn("for my dog Bruno", "2026-10-05", session_id="A", client=llm, store=store)
    assert "Today is Sunday, 2026-10-04" in llm.requests[0][0]["content"]
    assert "Today is Monday, 2026-10-05" in llm.requests[1][0]["content"]
    history = store.get_history("A")
    assert [m["role"] for m in history].count("system") == 1
    assert "2026-10-05" in history[0]["content"]


def test_endpoint_uses_server_date_dependency(llm):
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[current_server_date] = lambda: "2026-12-25"
    chat(TestClient(app), "A", "hello")
    assert "Today is Friday, 2026-12-25" in llm.requests[0][0]["content"]


def test_current_server_date_is_ist_today():
    assert current_server_date() == datetime.now(BUSINESS_TZ).date().isoformat()


def test_failed_turn_does_not_corrupt_history(llm):
    store = SessionStore()
    run_turn("I want to book grooming", TODAY, session_id="A", client=llm, store=store)
    before = store.get_history("A")

    class Boom:
        def chat(self, messages, tools):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        run_turn("for my dog Bruno", TODAY, session_id="A", client=Boom(), store=store)
    assert store.get_history("A") == before


def test_stateless_mode_still_works_without_session(llm):
    store = SessionStore()
    run_turn("I want to book grooming", TODAY, client=llm, store=store)
    assert store.session_ids() == []
