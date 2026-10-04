"""Tool-calling loop: one user message in, final reply + tool trace out.

Stateless by default; pass ``session_id`` to continue a per-session conversation
(history kept in ``app.session_store``). No loop limit / safety rails yet (Phase 5).
"""

from __future__ import annotations

import inspect
import json
import time
from datetime import date as Date, datetime, timezone
from pathlib import Path
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


import logging

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 6
LOOP_CAP_FALLBACK = "I'm having trouble completing that — could you confirm the details again?"


def execute_tool_call(
    name: str, raw_arguments: str | None, *, session_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and run one model-requested tool call.

    Returns ``(parsed_arguments, result_payload)`` where the payload is
    ``{"ok": True, "result": ...}`` or ``{"ok": False, "error": "..."}``.
    Catches PawCareError, ValueError, or any other tool failure and feeds
    the error description back to the model as JSON so it can explain and recover.
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
    except (tools.PawCareError, ValueError) as exc:
        return args, {"ok": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected error in tool %s: %s", name, exc)
        return args, {"ok": False, "error": f"Tool execution failed: {exc}"}
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


# Pricing reference: OpenAI GPT-4o-mini (openai.com/api/pricing)
# $0.150 / 1M prompt tokens, $0.600 / 1M completion tokens
INPUT_COST_PER_TOKEN = 0.150 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.600 / 1_000_000

METRICS_LOG_FILE = Path(__file__).resolve().parents[1] / "turn_metrics.jsonl"
LAST_TURN_METRICS: dict[str, Any] = {}


def get_last_metrics() -> dict[str, Any]:
    """Return a copy of the metrics dictionary from the most recently completed turn."""
    return dict(LAST_TURN_METRICS)


def _run_tool_loop(
    client: ChatClient, messages: list[dict[str, Any]], *, session_id: str | None
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Call the model with ``messages`` until it gives a text reply or hits the tool cap.

    Appends every assistant tool-call message, tool result and the final
    assistant reply to ``messages`` in place. Returns ``(reply, trace, metrics)``.
    """
    global LAST_TURN_METRICS
    trace: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0

    while True:
        if len(trace) >= MAX_TOOL_CALLS:
            logger.warning(
                "Tool-call loop cap (%d) reached in single turn. Halting tool loop.",
                MAX_TOOL_CALLS,
            )
            reply = LOOP_CAP_FALLBACK
            messages.append({"role": "assistant", "content": reply})
            break

        message = client.chat(messages=messages, tools=TOOL_SCHEMAS)
        usage = getattr(message, "usage", None)
        if usage:
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        else:
            # Fallback estimation for test mocks / offline testing
            prompt_tokens += sum(len(str(m.get("content", ""))) for m in messages) // 4 + 50
            completion_tokens += len(str(message.content or "")) // 4 + 10

        tool_calls = [
            tc for tc in (message.tool_calls or []) if getattr(tc, "type", "function") == "function"
        ]
        if not tool_calls:
            reply = message.content or ""
            messages.append({"role": "assistant", "content": reply})
            break

        remaining = MAX_TOOL_CALLS - len(trace)
        if len(tool_calls) > remaining:
            logger.warning(
                "Truncating tool calls from %d to remaining quota %d.", len(tool_calls), remaining
            )
            tool_calls = tool_calls[:remaining]

        messages.append(_assistant_tool_call_entry(message.content, tool_calls))
        for tc in tool_calls:
            args, result = execute_tool_call(
                tc.function.name, tc.function.arguments, session_id=session_id
            )
            trace.append({"name": tc.function.name, "arguments": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )

        if len(trace) >= MAX_TOOL_CALLS:
            logger.warning(
                "Tool-call loop cap (%d) reached in single turn. Returning fallback.",
                MAX_TOOL_CALLS,
            )
            reply = LOOP_CAP_FALLBACK
            messages.append({"role": "assistant", "content": reply})
            break

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    total_tokens = prompt_tokens + completion_tokens
    cost_usd = round(
        (prompt_tokens * INPUT_COST_PER_TOKEN) + (completion_tokens * OUTPUT_COST_PER_TOKEN), 8
    )

    metrics = {
        "latencyMs": elapsed_ms,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": total_tokens,
        "estimatedCostUsd": cost_usd,
    }
    LAST_TURN_METRICS = dict(metrics)

    logger.info(
        "Turn completed in %.2fms | Tokens: in=%d, out=%d, total=%d | Est Cost: $%.6f",
        elapsed_ms,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost_usd,
    )

    # Simple structured JSON log entry
    try:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sessionId": session_id,
            **metrics,
        }
        with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
    except Exception as exc:
        logger.debug("Failed writing turn metrics log: %s", exc)

    return reply, trace, metrics


def run_turn(
    user_message: str,
    current_date: str,
    *,
    session_id: str | None = None,
    client: ChatClient | None = None,
    store: SessionStore | None = None,
    return_metrics: bool = False,
) -> dict:
    """Run one user message through the model, executing tool calls until a text reply.

    Returns ``{"reply": str, "toolCalls": list}`` by default, or with
    ``"metrics": dict`` if ``return_metrics=True``.
    """
    Date.fromisoformat(current_date)
    if client is None:
        from app.llm_client import LLMClient

        client = LLMClient()
    system_message = {"role": "system", "content": build_system_prompt(current_date)}
    user_entry = {"role": "user", "content": user_message}

    if session_id is None:
        messages = [system_message, user_entry]
        reply, trace, metrics = _run_tool_loop(client, messages, session_id=None)
    else:
        store = store if store is not None else SESSION_STORE
        with store.turn(session_id):
            history = store.get_history(session_id)
            messages = [system_message, *history[1:], user_entry]
            reply, trace, metrics = _run_tool_loop(client, messages, session_id=session_id)
            store.save_history(session_id, messages)

    out: dict[str, Any] = {"reply": reply, "toolCalls": trace}
    if return_metrics:
        out["metrics"] = metrics
    return out

