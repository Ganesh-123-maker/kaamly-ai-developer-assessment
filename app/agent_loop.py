"""Single-turn tool-calling loop: one user message in, final reply + tool trace out.

No conversation memory (Phase 3) and no loop limit / safety rails (Phase 5) yet.
"""

from __future__ import annotations

import inspect
import json
from datetime import date as Date
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from app import tools
from app.system_prompt import build_system_prompt
from app.tool_schemas import SERVER_INJECTED_PARAMS, TOOL_SCHEMAS

_SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}


class ChatClient(Protocol):
    """Anything with LLMClient.chat's shape (lets tests inject a fake)."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _resolve_tool(name: str) -> Callable[..., Any]:
    # Looked up on the module at call time so tests can monkeypatch/spy on tools.
    return getattr(tools, name)


def execute_tool_call(
    name: str, raw_arguments: str | None, *, session_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and run one model-requested tool call.

    Returns ``(parsed_arguments, result_payload)`` where the payload is
    ``{"ok": True, "result": ...}`` or ``{"ok": False, "error": "..."}``.
    Expected failures (bad arguments, PawCareError) become error payloads so the
    model can read them and recover; genuine bugs still raise.
    """
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return {}, {"ok": False, "error": f"Unknown tool '{name}'. Available: {sorted(_SCHEMA_BY_NAME)}."}

    try:
        args = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return {}, {"ok": False, "error": f"Arguments for {name} are not valid JSON: {exc.msg}."}
    if not isinstance(args, dict):
        return {}, {"ok": False, "error": f"Arguments for {name} must be a JSON object."}

    allowed = set(schema["properties"])
    unexpected = sorted(set(args) - allowed)
    missing = sorted(set(schema["required"]) - set(args))
    if unexpected or missing:
        problems = []
        if missing:
            problems.append(f"missing required argument(s): {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected argument(s): {', '.join(unexpected)}")
        return args, {"ok": False, "error": f"Invalid call to {name}: {'; '.join(problems)}."}

    func = _resolve_tool(name)
    kwargs = dict(args)
    # Session scoping is decided server-side, never by the model.
    if SERVER_INJECTED_PARAMS & set(inspect.signature(func).parameters):
        kwargs["session_id"] = session_id

    try:
        result = func(**kwargs)
    except tools.PawCareError as exc:
        return args, {"ok": False, "error": str(exc)}
    return args, {"ok": True, "result": _to_jsonable(result)}


def _assistant_tool_call_entry(content: str | None, tool_calls: list[Any]) -> dict[str, Any]:
    """Re-serialize the assistant's tool-call message for the next request."""
    entry: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    }
    if content:  # content may legitimately be null alongside tool_calls
        entry["content"] = content
    return entry


def run_turn(user_message: str, current_date: str, *, client: ChatClient | None = None) -> dict:
    """Run one user message through the model, executing tool calls until a text reply.

    ``current_date`` (YYYY-MM-DD) is supplied by the caller and injected into the
    system prompt - this function never reads the system clock.

    Returns ``{"reply": str, "toolCalls": [{"name", "arguments", "result"}, ...]}``.
    """
    Date.fromisoformat(current_date)  # fail fast on a bad date (ValueError)
    if client is None:
        from app.llm_client import LLMClient  # lazy: tests never need real config

        client = LLMClient()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(current_date)},
        {"role": "user", "content": user_message},
    ]
    trace: list[dict[str, Any]] = []

    while True:  # Phase 5 adds an iteration cap
        message = client.chat(messages=messages, tools=TOOL_SCHEMAS)
        # Only function tools are defined; ignore any other tool-call types.
        tool_calls = [
            tc for tc in (message.tool_calls or []) if getattr(tc, "type", "function") == "function"
        ]
        if not tool_calls:
            return {"reply": message.content or "", "toolCalls": trace}

        messages.append(_assistant_tool_call_entry(message.content, tool_calls))
        for tc in tool_calls:
            args, result = execute_tool_call(tc.function.name, tc.function.arguments)
            trace.append({"name": tc.function.name, "arguments": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )
