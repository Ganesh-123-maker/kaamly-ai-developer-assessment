"""Thin wrapper around any OpenAI-compatible chat-completions API with tool calling.

Includes:
- Timeout configuration (default 15s).
- Exactly one retry on transient failure / timeout.
- Graceful fallback on permanent failure (never crashing callers).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessage

logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")

FALLBACK_REPLY = "Sorry, I'm having trouble right now — please try again in a moment."


class LLMConfigError(RuntimeError):
    """Raised when required LLM settings are missing."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    timeout: float = 15.0

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> "LLMConfig":
        """Build config from environment variables (optionally loading ``.env`` first)."""
        if load_env_file:
            load_dotenv()
        values = {name: (os.getenv(name) or "").strip() for name in REQUIRED_ENV_VARS}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise LLMConfigError(
                f"Missing LLM configuration: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your provider's values."
            )
        return cls(
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"],
            model=values["LLM_MODEL"],
        )

    def __repr__(self) -> str:
        return f"LLMConfig(base_url={self.base_url!r}, model={self.model!r}, timeout={self.timeout}, api_key='***')"


class LLMClient:
    """Sends chat requests with tools, timeout, 1 retry, and graceful degradation."""

    def __init__(self, config: LLMConfig | None = None, client: OpenAI | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = client or OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatCompletionMessage:
        """Call LLM with timeout and exactly ONE retry on transient errors.

        If all attempts fail, returns a friendly fallback message instead of raising.
        """
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                )
                return response.choices[0].message
            except (
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
                RateLimitError,
                OpenAIError,
            ) as exc:
                logger.warning(
                    "LLM chat call attempt %d failed with %s: %s",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )
                if attempt == 0:
                    continue  # Exactly one retry
                logger.error("LLM chat retry failed. Returning fallback message.")
                return ChatCompletionMessage.model_validate(
                    {"role": "assistant", "content": FALLBACK_REPLY}
                )
            except Exception as exc:
                logger.error("Unexpected error calling LLM: %s", exc)
                return ChatCompletionMessage.model_validate(
                    {"role": "assistant", "content": FALLBACK_REPLY}
                )
        return ChatCompletionMessage.model_validate(
            {"role": "assistant", "content": FALLBACK_REPLY}
        )
