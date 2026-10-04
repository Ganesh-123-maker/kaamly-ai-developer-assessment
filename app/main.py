"""FastAPI app: stateful multi-turn chat over the PawCare tools.

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.agent_loop import ChatClient, run_turn
from app.llm_client import LLMClient, LLMConfigError
from app.seed_data import BUSINESS_TZ
from app.session_store import SESSION_STORE

app = FastAPI(title="PawCare AI Assistant", version="0.3.0")


# --------------------------------------------------------------------------- schemas
class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, serialize_by_alias=True)


class ChatRequest(_CamelModel):
    session_id: str = Field(min_length=1, max_length=128, description="Client-chosen conversation id.")
    message: str = Field(min_length=1, max_length=4000)


class ToolCallRecord(_CamelModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(_CamelModel):
    reply: str
    tool_calls: list[ToolCallRecord]


# --------------------------------------------------------------------------- dependencies
@lru_cache(maxsize=1)
def get_llm_client() -> ChatClient:
    """Shared LLM client, created on first use (so /health works without config)."""
    return LLMClient()


def current_server_date() -> str:
    """Today's date on the server, in PawCare's business timezone (IST)."""
    return datetime.now(BUSINESS_TZ).date().isoformat()


FALLBACK_REPLY = "Sorry, I'm having trouble right now — please try again in a moment."


# --------------------------------------------------------------------------- errors
@app.exception_handler(LLMConfigError)
async def _llm_config_error(_: Request, exc: LLMConfigError) -> JSONResponse:
    return JSONResponse(
        status_code=200, content={"reply": FALLBACK_REPLY, "toolCalls": []}
    )


@app.exception_handler(OpenAIError)
async def _llm_provider_error(_: Request, exc: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=200, content={"reply": FALLBACK_REPLY, "toolCalls": []}
    )


# --------------------------------------------------------------------------- routes
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "serverDate": current_server_date(),
        "activeSessions": len(SESSION_STORE.session_ids()),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    client: ChatClient = Depends(get_llm_client),
    today: str = Depends(current_server_date),
) -> ChatResponse:
    # Sync endpoint -> runs in FastAPI's threadpool; SessionStore serializes turns
    # per session and tools.py locks the booking store.
    result = run_turn(request.message, today, session_id=request.session_id, client=client)
    return ChatResponse(reply=result["reply"], tool_calls=result["toolCalls"])
