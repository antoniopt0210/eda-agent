"""Anthropic / Claude provider implementation."""

from __future__ import annotations

from typing import Any

from . import BaseLLMProvider, LLMResponse, ToolCall, ToolResult

MAX_TOKENS = 4096


class AnthropicProvider(BaseLLMProvider):
    """Claude via the Anthropic SDK (tool-use / messages API)."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with:  pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic = anthropic
        self._messages: list[dict[str, Any]] = []
        self._raw_response: Any = None

    # -- BaseLLMProvider interface ------------------------------------------

    def send(self, system_prompt: str, tools: list[dict]) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=tools,  # already in Anthropic format
                messages=self._messages,
            )
        except self._anthropic.APIError:
            raise

        self._raw_response = response

        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        text_parts = [
            b.text for b in response.content if hasattr(b, "text") and b.text
        ]

        return LLMResponse(
            tool_calls=tool_calls,
            text=" ".join(text_parts),
            raw=response,
        )

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_response_and_tool_results(
        self,
        response: LLMResponse,
        tool_results: list[ToolResult],
    ) -> None:
        # Append assistant turn (raw Anthropic content blocks)
        self._messages.append({
            "role": "assistant",
            "content": self._raw_response.content,
        })
        # Append tool results as a user message (Anthropic convention)
        self._messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                }
                for r in tool_results
            ],
        })

    def reset(self) -> None:
        self._messages.clear()
        self._raw_response = None
