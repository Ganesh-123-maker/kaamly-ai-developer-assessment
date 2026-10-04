"""Agent-loop tests with a scripted (mocked) LLM - no network, no API key.

Responses are built as real ``openai`` SDK ``ChatCompletionMessage`` objects so
the loop is exercised against the same types the live client returns.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletionMessage

from app import tools
from app.agent_loop import execute_tool_call, run_turn
from app.llm_client import LLMClient, LLMConfig, LLMConfigError
from app.seed_data import BUSINESS_TZ
from app.tool_schemas import TOOL_SCHEMAS

CURRENT_DATE = "2026-10-04"  # Sunday
SATURDAY = "2026-10-10"
MONDAY = "2026-10-05"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(tools, "_now", lambda: datetime(2026, 10, 4, 8, 0, tzinfo=BUSINESS_TZ))
    tools.reset_bookings()
    yield
    tools.reset_bookings()


# ------------------------------------------------------------------ helpers
def text_reply(content: str) -> ChatCompletionMessage:
    return ChatCompletionMessage.model_validate({"role": "assistant", "content": content})


def tool_reply(*calls: tuple[str, dict | str], content: str | None = None) -> ChatCompletionMessage:
    return ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                }
                for i, (name, args) in enumerate(calls)
            ],
        }
    )


class ScriptedLLM:
    """Returns pre-scripted messages in order and records every request."""

    def __init__(self, *responses: ChatCompletionMessage):
        self._responses = list(responses)
        self.requests: list[dict] = []

    def chat(self, messages, tools):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools})
        return self._responses.pop(0)


# ------------------------------------------------------------------ no tool call
def test_off_topic_message_returns_text_without_tool_calls():
    llm = ScriptedLLM(text_reply("I can only help with PawCare pet service bookings."))
    out = run_turn("What's the capital of France?", CURRENT_DATE, client=llm)

    assert out == {"reply": "I can only help with PawCare pet service bookings.", "toolCalls": []}
    assert len(llm.requests) == 1
    req = llm.requests[0]
    assert req["tools"] is TOOL_SCHEMAS
    assert req["messages"][0]["role"] == "system"
    assert req["messages"][1] == {"role": "user", "content": "What's the capital of France?"}


def test_current_date_is_injected_into_system_prompt():
    llm = ScriptedLLM(text_reply("ok"))
    run_turn("hi", CURRENT_DATE, client=llm)
    system = llm.requests[0]["messages"][0]["content"]
    assert "Today is Sunday, 2026-10-04" in system


def test_invalid_current_date_raises():
    with pytest.raises(ValueError):
        run_turn("hi", "04/10/2026", client=ScriptedLLM())


# ------------------------------------------------------------------ dispatch
def test_list_services_call_invokes_function_and_feeds_result_back(monkeypatch):
    spy = MagicMock(wraps=tools.list_services)
    monkeypatch.setattr(tools, "list_services", spy)
    llm = ScriptedLLM(tool_reply(("list_services", {})), text_reply("We offer 4 services."))

    out = run_turn("What services do you offer?", CURRENT_DATE, client=llm)

    spy.assert_called_once_with()
    assert out["reply"] == "We offer 4 services."
    [call] = out["toolCalls"]
    assert call["name"] == "list_services" and call["arguments"] == {}
    assert call["result"]["ok"] is True
    assert {s["serviceId"] for s in call["result"]["result"]} == {
        "dog_walk", "grooming", "vet_home_visit", "day_care"
    }

    # Second request carries the assistant tool_call + matching tool result.
    second = llm.requests[1]["messages"]
    assert second[2]["role"] == "assistant"
    assert second[2]["tool_calls"][0]["function"]["name"] == "list_services"
    assert "content" not in second[2]  # null content omitted
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "call_0"
    assert json.loads(second[3]["content"]) == call["result"]


def test_check_slots_called_with_model_arguments(monkeypatch):
    spy = MagicMock(wraps=tools.check_slots)
    monkeypatch.setattr(tools, "check_slots", spy)
    llm = ScriptedLLM(
        tool_reply(("check_slots", {"service_id": "grooming", "date": SATURDAY})),
        text_reply("Saturday grooming slots: 9 AM - 3 PM."),
    )

    out = run_turn("Is grooming available next Saturday?", CURRENT_DATE, client=llm)

    spy.assert_called_once_with(service_id="grooming", date=SATURDAY)
    assert out["toolCalls"][0]["result"] == {
        "ok": True,
        "result": ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"],
    }


def test_multi_step_chain_and_parallel_calls():
    llm = ScriptedLLM(
        tool_reply(("list_services", {})),
        tool_reply(
            ("check_slots", {"service_id": "vet_home_visit", "date": MONDAY}),
            ("check_slots", {"service_id": "day_care", "date": SATURDAY}),
        ),
        text_reply("Here is what's available."),
    )
    out = run_turn("Vet Monday or day care Saturday?", CURRENT_DATE, client=llm)

    assert [c["name"] for c in out["toolCalls"]] == ["list_services", "check_slots", "check_slots"]
    assert out["toolCalls"][2]["result"]["result"] == ["09:00"]
    # Both parallel tool results were sent back, each with its own id.
    last = llm.requests[2]["messages"]
    assert [m["tool_call_id"] for m in last if m["role"] == "tool"] == ["call_0", "call_0", "call_1"]


def test_create_booking_call_really_books():
    llm = ScriptedLLM(
        tool_reply(
            ("create_booking", {"pet_name": "Bruno", "service_id": "dog_walk",
                                "date": MONDAY, "slot": "10:00"})
        ),
        text_reply("Booked!"),
    )
    out = run_turn("Yes, book it.", CURRENT_DATE, client=llm)

    result = out["toolCalls"][0]["result"]
    assert result["ok"] is True
    [stored] = tools.get_bookings()
    assert result["result"]["bookingId"] == stored.booking_id
    assert stored.session_id is None  # server-injected, not model-chosen
    assert "10:00" not in tools.check_slots("dog_walk", MONDAY)


def test_tool_error_is_fed_back_and_loop_continues():
    llm = ScriptedLLM(
        tool_reply(("check_slots", {"service_id": "Grooming Deluxe", "date": SATURDAY})),
        tool_reply(("check_slots", {"service_id": "grooming", "date": SATURDAY})),
        text_reply("Grooming is available Saturday."),
    )
    out = run_turn("grooming saturday?", CURRENT_DATE, client=llm)

    first, second = out["toolCalls"]
    assert first["result"]["ok"] is False
    assert "Unknown service_id 'Grooming Deluxe'" in first["result"]["error"]
    assert second["result"]["ok"] is True
    assert json.loads(llm.requests[1]["messages"][-1]["content"])["ok"] is False


# ------------------------------------------------------------------ argument validation
def test_unknown_tool_returns_error():
    args, result = execute_tool_call("delete_everything", "{}")
    assert result["ok"] is False and "Unknown tool" in result["error"]


def test_malformed_json_arguments_do_not_call_function(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(tools, "check_slots", spy)
    _, result = execute_tool_call("check_slots", '{"service_id": "grooming", ')
    assert result["ok"] is False and "not valid JSON" in result["error"]
    spy.assert_not_called()


def test_missing_required_argument_returns_error(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(tools, "check_slots", spy)
    _, result = execute_tool_call("check_slots", json.dumps({"service_id": "grooming"}))
    assert result == {"ok": False, "error": "Invalid call to check_slots: missing required argument(s): date."}
    spy.assert_not_called()


def test_model_cannot_set_session_id():
    tools.create_booking("Bruno", "dog_walk", MONDAY, "09:00", session_id="alice")
    _, result = execute_tool_call("get_bookings", json.dumps({"session_id": "alice"}))
    assert result["ok"] is False and "unexpected argument(s): session_id" in result["error"]


def test_server_side_session_id_is_injected():
    tools.create_booking("Bruno", "dog_walk", MONDAY, "09:00", session_id="alice")
    tools.create_booking("Max", "dog_walk", MONDAY, "10:00", session_id="bob")
    _, result = execute_tool_call("get_bookings", "{}", session_id="alice")
    assert [b["petName"] for b in result["result"]] == ["Bruno"]


def test_empty_arguments_string_treated_as_no_args():
    _, result = execute_tool_call("list_services", "")
    assert result["ok"] is True and len(result["result"]) == 4


def test_cancel_booking_bool_result_and_not_found():
    booking = tools.create_booking("Bruno", "dog_walk", MONDAY, "09:00")
    _, ok = execute_tool_call("cancel_booking", json.dumps({"booking_id": booking.booking_id}))
    assert ok == {"ok": True, "result": True}
    _, missing = execute_tool_call("cancel_booking", json.dumps({"booking_id": "nope"}))
    assert missing["ok"] is False and "No booking found" in missing["error"]


# ------------------------------------------------------------------ llm_client config
def test_llm_config_missing_vars_raises(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMConfigError, match="LLM_API_KEY, LLM_BASE_URL, LLM_MODEL"):
        LLMConfig.from_env(load_env_file=False)


def test_llm_client_uses_env_config_and_hides_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://example-provider.test/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    config = LLMConfig.from_env(load_env_file=False)
    assert "sk-test-secret" not in repr(config)

    client = LLMClient(config)
    assert str(client._client.base_url).rstrip("/") == "https://example-provider.test/v1"
    assert client._client.api_key == "sk-test-secret"


def test_llm_client_chat_sends_model_tools_and_returns_message():
    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create.return_value.choices = [MagicMock(message="MSG")]
    config = LLMConfig(api_key="k", base_url="https://x.test/v1", model="m")
    msgs = [{"role": "user", "content": "hi"}]

    assert LLMClient(config, client=fake_sdk).chat(msgs, TOOL_SCHEMAS) == "MSG"
    kwargs = fake_sdk.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "m" and kwargs["tools"] is TOOL_SCHEMAS
    assert kwargs["messages"] is msgs and kwargs["tool_choice"] == "auto"
