"""Tool-calling loop: one user message in, final reply + tool trace out.

Stateless by default; pass ``session_id`` to continue a per-session conversation
(history kept in ``app.session_store``). No loop limit / safety rails yet (Phase 5).
"""

from __future__ import annotations

import inspect
import json
from datetime import date as Date
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from app import tools
from app.session_store import SESSION_STORE, SessionStore
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


def _run_tool_loop(
    client: ChatClient, messages: list[dict[str, Any]], *, session_id: str | None
) -> tuple[str, list[dict[str, Any]]]:
    """Call the model with ``messages`` until it gives a text reply.

    Appends every assistant tool-call message, tool result and the final
    assistant reply to ``messages`` in place. Returns ``(reply, trace)`` where
    ``trace`` lists only the tools invoked during this call.
    """
    trace: list[dict[str, Any]] = []
    while True:  # Phase 5 adds an iteration cap
        message = client.chat(messages=messages, tools=TOOL_SCHEMAS)
        # Only function tools are defined; ignore any other tool-call types.
        tool_calls = [
            tc for tc in (message.tool_calls or []) if getattr(tc, "type", "function") == "function"
        ]
        if not tool_calls:
            reply = message.content or ""
            messages.append({"role": "assistant", "content": reply})
            return reply, trace

        messages.append(_assistant_tool_call_entry(message.content, tool_calls))
        for tc in tool_calls:
            args, result = execute_tool_call(
                tc.function.name, tc.function.arguments, session_id=session_id
            )
            trace.append({"name": tc.function.name, "arguments": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )


def run_turn(
    user_message: str,
    current_date: str,
    *,
    session_id: str | None = None,
    client: ChatClient | None = None,
    store: SessionStore | None = None,
) -> dict:
    """Run one user message through the model, executing tool calls until a text reply.

    ``current_date`` (YYYY-MM-DD) is supplied by the caller and injected into the
    system prompt - this function never reads the system clock.

    Without ``session_id`` the turn is stateless (fresh conversation). With a
    ``session_id`` the session's full prior history is sent, the system prompt
    is refreshed with ``current_date``, tools are scoped to that session, and the
    updated history is saved - only if the turn completes successfully.

    Returns ``{"reply": str, "toolCalls": [{"name", "arguments", "result"}, ...]}``
    where ``toolCalls`` covers this turn only.
    """
    Date.fromisoformat(current_date)  # fail fast on a bad date (ValueError)
    if client is None:
        from app.llm_client import LLMClient  # lazy: tests never need real config

        client = LLMClient()
    system_message = {"role": "system", "content": build_system_prompt(current_date)}
    user_entry = {"role": "user", "content": user_message}

    if session_id is None:
        messages = [system_message, user_entry]
        reply, trace = _run_tool_loop(client, messages, session_id=None)
        return {"reply": reply, "toolCalls": trace}

    store = store if store is not None else SESSION_STORE
    with store.turn(session_id):
        history = store.get_history(session_id)
        # Index 0 is always the system prompt; refresh it so the date is current.
        messages = [system_message, *history[1:], user_entry]
        reply, trace = _run_tool_loop(client, messages, session_id=session_id)
        store.save_history(session_id, messages)
    return {"reply": reply, "toolCalls": trace}

