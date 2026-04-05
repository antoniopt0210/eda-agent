"""LLM provider abstraction — supports Anthropic, OpenAI, and Google Gemini."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool invocation parsed from an LLM response."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a tool, sent back to the LLM."""
    tool_call_id: str
    content: str


@dataclass
class LLMResponse:
    """Normalised LLM response returned by every provider."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    raw: Any = None  # provider-specific raw response (kept for message history)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers.

    Each provider manages its own internal message history and converts
    between the canonical Anthropic-format tool definitions and whatever
    the underlying SDK expects.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def send(self, system_prompt: str, tools: list[dict]) -> LLMResponse:
        """Send the current conversation and return a normalised response.

        *tools* are always in Anthropic format (``name``, ``description``,
        ``input_schema``).  Each provider converts internally.
        """
        ...

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """Append a plain-text user message to the conversation."""
        ...

    @abstractmethod
    def add_response_and_tool_results(
        self,
        response: LLMResponse,
        tool_results: list[ToolResult],
    ) -> None:
        """Append the assistant turn **and** tool results to the conversation."""
        ...

    def reset(self) -> None:
        """Clear conversation history (provider may override)."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

PROVIDER_ALIASES: dict[str, str] = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
}

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}

# Suggested models shown in UIs
MODEL_OPTIONS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-20250115",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o3-mini",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-03-25",
    ],
}


def create_provider(provider_name: str, api_key: str, model: str | None = None) -> BaseLLMProvider:
    """Instantiate the requested LLM provider.

    Raises ``ValueError`` for unknown providers and ``ImportError`` if the
    required SDK package is not installed.
    """
    key = PROVIDER_ALIASES.get(provider_name.lower())
    if key is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Choose from: {', '.join(sorted(PROVIDER_ALIASES.keys()))}"
        )

    resolved_model = model or DEFAULT_MODELS[key]

    if key == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key, resolved_model)

    if key == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(api_key, resolved_model)

    if key == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(api_key, resolved_model)

    raise ValueError(f"Provider '{key}' is not implemented.")
