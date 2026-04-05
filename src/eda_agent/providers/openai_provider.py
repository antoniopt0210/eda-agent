"""OpenAI / GPT provider implementation."""

from __future__ import annotations

import json
from typing import Any

from . import BaseLLMProvider, LLMResponse, ToolCall, ToolResult

MAX_TOKENS = 4096


class OpenAIProvider(BaseLLMProvider):
    """GPT models via the OpenAI SDK (chat completions with tools)."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with:  pip install openai"
            )
        self._client = OpenAI(api_key=api_key)
        self._messages: list[dict[str, Any]] = []
        self._raw_message: Any = None  # last assistant message object

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Anthropic format -> OpenAI function-calling format."""
        out: list[dict] = []
        for t in tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            })
        return out

    # -- BaseLLMProvider interface ------------------------------------------

    def send(self, system_prompt: str, tools: list[dict]) -> LLMResponse:
        all_messages = [{"role": "system", "content": system_prompt}] + self._messages

        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            tools=self._convert_tools(tools),
            max_tokens=MAX_TOKENS,
        )

        msg = response.choices[0].message
        self._raw_message = msg

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                ))

        return LLMResponse(
            tool_calls=tool_calls,
            text=msg.content or "",
            raw=msg,
        )

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_response_and_tool_results(
        self,
        response: LLMResponse,
        tool_results: list[ToolResult],
    ) -> None:
        # Append assistant message (must include tool_calls for OpenAI)
        msg_dict: dict[str, Any] = {"role": "assistant", "content": self._raw_message.content}
        if self._raw_message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self._raw_message.tool_calls
            ]
        self._messages.append(msg_dict)

        # Each tool result is a separate message with role "tool"
        for result in tool_results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            })

    def reset(self) -> None:
        self._messages.clear()
        self._raw_message = None
