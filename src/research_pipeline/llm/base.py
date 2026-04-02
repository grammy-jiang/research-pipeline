"""LLM provider interface."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def call(
        self,
        prompt: str,
        schema_id: str,
        temperature: float = 0.0,
    ) -> dict:  # type: ignore[type-arg]
        """Make an LLM call with schema-constrained output.

        Args:
            prompt: The prompt text.
            schema_id: ID of the expected output schema.
            temperature: Sampling temperature.

        Returns:
            Parsed response dict conforming to the schema.
        """

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
