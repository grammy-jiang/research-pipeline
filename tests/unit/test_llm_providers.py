"""Tests for LLM provider implementations."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

from research_pipeline.config.models import LLMConfig
from research_pipeline.llm.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    create_llm_provider,
)

# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


class TestOllamaProviderModelName:
    """OllamaProvider.model_name() returns the configured model."""

    def test_default_model(self) -> None:
        provider = OllamaProvider()
        assert provider.model_name() == "llama3.2"

    def test_custom_model(self) -> None:
        provider = OllamaProvider(model="mistral")
        assert provider.model_name() == "mistral"


class TestOllamaProviderCall:
    """OllamaProvider.call() HTTP interaction."""

    def test_successful_call(self) -> None:
        inner = json.dumps({"answer": 42})
        body = json.dumps({"response": inner}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            provider = OllamaProvider(base_url="http://test:11434")
            result = provider.call("hello", schema_id="test_schema", temperature=0.1)

        assert result == {"answer": 42}
        called_req = mock_open.call_args[0][0]
        assert called_req.full_url == "http://test:11434/api/generate"
        payload = json.loads(called_req.data)
        assert payload["model"] == "llama3.2"
        assert payload["format"] == "json"
        assert payload["stream"] is False

    def test_connection_error_returns_empty_dict(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            provider = OllamaProvider()
            result = provider.call("hello", schema_id="s")

        assert result == {}

    def test_invalid_json_response_returns_empty_dict(self) -> None:
        body = json.dumps({"response": "not-json{"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            provider = OllamaProvider()
            result = provider.call("hello", schema_id="s")

        assert result == {}

    def test_timeout_error_returns_empty_dict(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            provider = OllamaProvider()
            result = provider.call("hello", schema_id="s")

        assert result == {}


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider
# ---------------------------------------------------------------------------


class TestOpenAICompatibleProviderModelName:
    """OpenAICompatibleProvider.model_name() returns the configured model."""

    def test_default_model(self) -> None:
        provider = OpenAICompatibleProvider()
        assert provider.model_name() == "gpt-4o-mini"

    def test_custom_model(self) -> None:
        provider = OpenAICompatibleProvider(model="gpt-4o")
        assert provider.model_name() == "gpt-4o"


class TestOpenAICompatibleProviderCall:
    """OpenAICompatibleProvider.call() HTTP interaction."""

    def test_successful_call(self) -> None:
        content = json.dumps({"summary": "test"})
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            provider = OpenAICompatibleProvider(
                base_url="https://api.example.com/v1",
                api_key="sk-test",
                model="gpt-4o",
            )
            result = provider.call("hello", schema_id="test_schema")

        assert result == {"summary": "test"}
        called_req = mock_open.call_args[0][0]
        assert called_req.full_url == "https://api.example.com/v1/chat/completions"
        assert called_req.get_header("Authorization") == "Bearer sk-test"
        payload = json.loads(called_req.data)
        assert payload["model"] == "gpt-4o"
        assert payload["response_format"] == {"type": "json_object"}

    def test_connection_error_returns_empty_dict(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            provider = OpenAICompatibleProvider(api_key="sk-test")
            result = provider.call("hello", schema_id="s")

        assert result == {}

    def test_invalid_json_content_returns_empty_dict(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "not valid json{"}}]}
        ).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            provider = OpenAICompatibleProvider(api_key="sk-test")
            result = provider.call("hello", schema_id="s")

        assert result == {}

    def test_no_api_key_omits_auth_header(self) -> None:
        content = json.dumps({"ok": True})
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            provider = OpenAICompatibleProvider(api_key="")
            provider.call("hello", schema_id="s")

        called_req = mock_open.call_args[0][0]
        assert called_req.get_header("Authorization") is None


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateLLMProvider:
    """create_llm_provider() factory tests."""

    def test_disabled_returns_none(self) -> None:
        config = LLMConfig(enabled=False)
        assert create_llm_provider(config) is None

    def test_ollama_provider(self) -> None:
        config = LLMConfig(enabled=True, provider="ollama", model="phi3")
        provider = create_llm_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name() == "phi3"

    def test_openai_provider(self) -> None:
        config = LLMConfig(
            enabled=True,
            provider="openai",
            api_key="sk-test",
            model="gpt-4o",
        )
        provider = create_llm_provider(config)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.model_name() == "gpt-4o"

    def test_unknown_provider_returns_none(self) -> None:
        config = LLMConfig(enabled=True, provider="unknown")
        assert create_llm_provider(config) is None

    def test_ollama_default_model_when_empty(self) -> None:
        config = LLMConfig(enabled=True, provider="ollama", model="")
        provider = create_llm_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name() == "llama3.2"

    def test_openai_default_model_when_empty(self) -> None:
        config = LLMConfig(enabled=True, provider="openai", model="")
        provider = create_llm_provider(config)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.model_name() == "gpt-4o-mini"

    def test_custom_base_url(self) -> None:
        config = LLMConfig(
            enabled=True,
            provider="openai",
            base_url="http://local:8080/v1",
        )
        provider = create_llm_provider(config)
        assert isinstance(provider, OpenAICompatibleProvider)
