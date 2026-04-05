"""Tests for the multi-provider abstraction layer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eda_agent.providers import (
    DEFAULT_MODELS,
    MODEL_OPTIONS,
    PROVIDER_ALIASES,
    BaseLLMProvider,
    LLMResponse,
    ToolCall,
    ToolResult,
    create_provider,
)


class TestProviderFactory:
    def test_create_anthropic(self) -> None:
        provider = create_provider("anthropic", "fake-key")
        assert isinstance(provider, BaseLLMProvider)
        assert provider.model == DEFAULT_MODELS["anthropic"]

    def test_create_openai(self) -> None:
        provider = create_provider("openai", "fake-key")
        assert isinstance(provider, BaseLLMProvider)
        assert provider.model == DEFAULT_MODELS["openai"]

    def test_create_gemini(self) -> None:
        provider = create_provider("gemini", "fake-key")
        assert isinstance(provider, BaseLLMProvider)
        assert provider.model == DEFAULT_MODELS["gemini"]

    def test_alias_claude(self) -> None:
        provider = create_provider("claude", "fake-key")
        assert provider.model == DEFAULT_MODELS["anthropic"]

    def test_alias_gpt(self) -> None:
        provider = create_provider("gpt", "fake-key")
        assert provider.model == DEFAULT_MODELS["openai"]

    def test_alias_google(self) -> None:
        provider = create_provider("google", "fake-key")
        assert provider.model == DEFAULT_MODELS["gemini"]

    def test_custom_model(self) -> None:
        provider = create_provider("anthropic", "fake-key", model="claude-opus-4-20250115")
        assert provider.model == "claude-opus-4-20250115"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown-ai", "fake-key")

    def test_case_insensitive(self) -> None:
        provider = create_provider("OpenAI", "fake-key")
        assert isinstance(provider, BaseLLMProvider)


class TestCommonTypes:
    def test_tool_call(self) -> None:
        tc = ToolCall(id="tc_1", name="run_python_code", input={"code": "print(1)"})
        assert tc.name == "run_python_code"

    def test_tool_result(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", content="output")
        assert tr.content == "output"

    def test_llm_response_defaults(self) -> None:
        r = LLMResponse()
        assert r.tool_calls == []
        assert r.text == ""
        assert r.raw is None


class TestModelOptions:
    def test_all_providers_have_options(self) -> None:
        for key in DEFAULT_MODELS:
            assert key in MODEL_OPTIONS
            assert len(MODEL_OPTIONS[key]) > 0

    def test_default_model_in_options(self) -> None:
        for key, default in DEFAULT_MODELS.items():
            assert default in MODEL_OPTIONS[key]
