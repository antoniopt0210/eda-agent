"""Google Gemini provider implementation (google-genai SDK)."""

from __future__ import annotations

import uuid
from typing import Any

from . import BaseLLMProvider, LLMResponse, ToolCall, ToolResult


class GeminiProvider(BaseLLMProvider):
    """Gemini models via the ``google-genai`` SDK.

    Uses the stateless ``client.models.generate_content`` API with manual
    conversation history so we stay in full control (identical pattern to
    the Anthropic and OpenAI providers).
    """

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required for the Gemini provider. "
                "Install it with:  pip install google-genai"
            )
        self._client = genai.Client(api_key=api_key)
        self._genai = genai
        self._types = types
        self._history: list[Any] = []  # list of types.Content

    # -- helpers ------------------------------------------------------------

    def _convert_tools(self, tools: list[dict]) -> list[Any]:
        """Anthropic-format tools -> Gemini Tool declarations."""
        declarations = []
        for t in tools:
            declarations.append(self._types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=self._convert_schema(t["input_schema"]),
            ))
        return [self._types.Tool(function_declarations=declarations)]

    def _convert_schema(self, schema: dict) -> dict[str, Any]:
        """Convert JSON Schema dict to a Gemini-compatible schema dict.

        The ``google-genai`` SDK accepts plain dicts with uppercase type names.
        """
        type_map = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        result: dict[str, Any] = {
            "type": type_map.get(schema.get("type", "string"), "STRING"),
        }

        if "description" in schema:
            result["description"] = schema["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]

        if schema.get("type") == "object" and "properties" in schema:
            result["properties"] = {
                k: self._convert_schema(v) for k, v in schema["properties"].items()
            }
            if "required" in schema:
                result["required"] = schema["required"]

        if schema.get("type") == "array" and "items" in schema:
            result["items"] = self._convert_schema(schema["items"])

        return result

    # -- BaseLLMProvider interface ------------------------------------------

    def send(self, system_prompt: str, tools: list[dict]) -> LLMResponse:
        gemini_tools = self._convert_tools(tools)

        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=self._history,
            config=config,
        )

        # Parse response parts
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call and part.function_call.name:
                    args = dict(part.function_call.args) if part.function_call.args else {}
                    tool_calls.append(ToolCall(
                        id=f"{part.function_call.name}_{uuid.uuid4().hex[:8]}",
                        name=part.function_call.name,
                        input=args,
                    ))
                elif part.text:
                    text_parts.append(part.text)

        return LLMResponse(
            tool_calls=tool_calls,
            text=" ".join(text_parts),
            raw=response,
        )

    def add_user_message(self, content: str) -> None:
        self._history.append(self._types.Content(
            role="user",
            parts=[self._types.Part(text=content)],
        ))

    def add_response_and_tool_results(
        self,
        response: LLMResponse,
        tool_results: list[ToolResult],
    ) -> None:
        # Append assistant turn from raw response
        if response.raw and response.raw.candidates and response.raw.candidates[0].content:
            self._history.append(response.raw.candidates[0].content)

        # Build function-response parts
        parts = []
        for result in tool_results:
            tc = next((t for t in response.tool_calls if t.id == result.tool_call_id), None)
            name = tc.name if tc else "unknown"
            parts.append(self._types.Part(
                function_response=self._types.FunctionResponse(
                    name=name,
                    response={"result": result.content},
                ),
            ))

        self._history.append(self._types.Content(role="user", parts=parts))

    def reset(self) -> None:
        self._history.clear()
