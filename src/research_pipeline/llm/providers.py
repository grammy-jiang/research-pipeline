"""Concrete LLM provider implementations."""

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from research_pipeline.config.models import LLMConfig
from research_pipeline.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance.

    Args:
        base_url: Ollama API base URL.
        model: Model name to use for generation.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_OLLAMA_URL,
        model: str = _DEFAULT_OLLAMA_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens

    def call(
        self,
        prompt: str,
        schema_id: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Make a JSON-mode generation request to Ollama.

        Args:
            prompt: The prompt text.
            schema_id: ID of the expected output schema.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON response dict, or empty dict on failure.
        """
        url = f"{self._base_url}/api/generate"
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": self._max_tokens,
                },
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
                body = json.loads(resp.read().decode())
            raw = body.get("response", "{}")
            return json.loads(raw)  # type: ignore[no-any-return]
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Ollama connection error: %s", exc)
            return {}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Ollama response parse error: %s", exc)
            return {}

    def model_name(self) -> str:
        """Return the Ollama model identifier."""
        return self._model


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for OpenAI-compatible chat completion APIs.

    Args:
        base_url: API base URL (e.g. ``https://api.openai.com/v1``).
        api_key: Bearer token for authentication.
        model: Model name to request.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_OPENAI_URL,
        api_key: str = "",
        model: str = _DEFAULT_OPENAI_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens

    def call(
        self,
        prompt: str,
        schema_id: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Make a chat completion request to an OpenAI-compatible API.

        Args:
            prompt: The prompt text.
            schema_id: ID of the expected output schema.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON response dict, or empty dict on failure.
        """
        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": self._max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
                body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)  # type: ignore[no-any-return]
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("OpenAI-compatible API connection error: %s", exc)
            return {}
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
            logger.warning("OpenAI-compatible API parse error: %s", exc)
            return {}

    def model_name(self) -> str:
        """Return the OpenAI-compatible model identifier."""
        return self._model


def create_llm_provider(config: LLMConfig) -> LLMProvider | None:
    """Instantiate an LLM provider from pipeline configuration.

    Args:
        config: The LLM section of the pipeline configuration.

    Returns:
        A concrete ``LLMProvider``, or ``None`` when LLM integration is
        disabled or the provider name is unrecognised.
    """
    if not config.enabled:
        logger.debug("LLM integration disabled")
        return None

    provider = config.provider.lower()

    if provider == "ollama":
        base_url = config.base_url or _DEFAULT_OLLAMA_URL
        model = config.model or _DEFAULT_OLLAMA_MODEL
        logger.info("Creating Ollama provider: %s @ %s", model, base_url)
        return OllamaProvider(
            base_url=base_url,
            model=model,
            max_tokens=config.max_tokens,
        )

    if provider == "openai":
        base_url = config.base_url or _DEFAULT_OPENAI_URL
        model = config.model or _DEFAULT_OPENAI_MODEL
        logger.info("Creating OpenAI-compatible provider: %s @ %s", model, base_url)
        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=config.api_key,
            model=model,
            max_tokens=config.max_tokens,
        )

    logger.warning("Unknown LLM provider: %s", provider)
    return None
