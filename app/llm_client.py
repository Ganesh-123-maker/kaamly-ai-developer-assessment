"""Thin wrapper around any OpenAI-compatible chat-completions API with tool calling.

Provider-agnostic: point LLM_BASE_URL at any OpenAI-compatible endpoint (Groq,
OpenRouter, Gemini's OpenAI endpoint, Together, a local Ollama, ...), set
LLM_API_KEY and LLM_MODEL, and nothing else changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


class LLMConfigError(RuntimeError):
    """Raised when required LLM settings are missing."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2

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

    def __repr__(self) -> str:  # never leak the key into logs
        return f"LLMConfig(base_url={self.base_url!r}, model={self.model!r}, api_key='***')"


class LLMClient:
    """Sends a chat-completions request with tools and returns the assistant message."""

    def __init__(self, config: LLMConfig | None = None, client: OpenAI | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = client or OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatCompletionMessage:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config.temperature,
        )
        return response.choices[0].message
