"""Phase-aware model routing.

Classifies pipeline stages into complexity tiers and routes LLM calls
to the appropriate provider/model per tier.

Tier definitions (from deep research, Pattern 2):
- MECHANICAL: Query formatting, dedup, metadata parsing, checkpointing
  → Local/cheap model (Ollama, small model)
- INTELLIGENT: Analysis, synthesis, gap identification, query generation
  → Cloud API (GPT-4, Claude)
- CRITICAL_SAFETY: Final report, security-sensitive operations
  → Multi-model consensus or premium model
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from research_pipeline.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class PhaseTier(StrEnum):
    """Pipeline phase complexity tier."""

    MECHANICAL = "mechanical"
    INTELLIGENT = "intelligent"
    CRITICAL_SAFETY = "critical_safety"


# Default classification of pipeline stages to tiers.
# Users can override via config.
STAGE_TIER_MAP: dict[str, PhaseTier] = {
    # Mechanical stages — deterministic, formatting, bookkeeping
    "plan": PhaseTier.MECHANICAL,
    "search": PhaseTier.MECHANICAL,
    "download": PhaseTier.MECHANICAL,
    "convert": PhaseTier.MECHANICAL,
    "convert_rough": PhaseTier.MECHANICAL,
    "convert_fine": PhaseTier.MECHANICAL,
    "extract": PhaseTier.MECHANICAL,
    "index": PhaseTier.MECHANICAL,
    "export_html": PhaseTier.MECHANICAL,
    # Intelligent stages — require deep reasoning
    "screen": PhaseTier.INTELLIGENT,
    "summarize": PhaseTier.INTELLIGENT,
    "expand": PhaseTier.INTELLIGENT,
    "quality": PhaseTier.INTELLIGENT,
    "analyze": PhaseTier.INTELLIGENT,
    "aggregate": PhaseTier.INTELLIGENT,
    # Critical safety — final outputs, security-sensitive
    "validate": PhaseTier.CRITICAL_SAFETY,
    "compare": PhaseTier.CRITICAL_SAFETY,
    "security_gate": PhaseTier.CRITICAL_SAFETY,
}


class ModelRouter:
    """Route LLM calls to the appropriate provider based on phase tier.

    Supports three provider slots (mechanical, intelligent, critical_safety).
    Falls back to the next-higher tier if a slot is empty.

    Args:
        mechanical: Provider for MECHANICAL phases (cheap/local).
        intelligent: Provider for INTELLIGENT phases (capable/cloud).
        critical_safety: Provider for CRITICAL_SAFETY phases (premium).
        stage_overrides: Custom stage→tier mappings that override defaults.
    """

    def __init__(
        self,
        mechanical: LLMProvider | None = None,
        intelligent: LLMProvider | None = None,
        critical_safety: LLMProvider | None = None,
        stage_overrides: dict[str, str] | None = None,
    ) -> None:
        self._providers: dict[PhaseTier, LLMProvider | None] = {
            PhaseTier.MECHANICAL: mechanical,
            PhaseTier.INTELLIGENT: intelligent,
            PhaseTier.CRITICAL_SAFETY: critical_safety,
        }
        self._stage_map = dict(STAGE_TIER_MAP)
        if stage_overrides:
            for stage, tier_str in stage_overrides.items():
                try:
                    self._stage_map[stage] = PhaseTier(tier_str)
                except ValueError:
                    logger.warning(
                        "Unknown tier %r for stage %r, ignoring", tier_str, stage
                    )

    @property
    def stage_map(self) -> dict[str, PhaseTier]:
        """Return the current stage→tier mapping (read-only copy)."""
        return dict(self._stage_map)

    def get_tier(self, stage: str) -> PhaseTier:
        """Classify a pipeline stage into a tier.

        Args:
            stage: Pipeline stage name.

        Returns:
            The tier for the stage, defaulting to INTELLIGENT for unknown stages.
        """
        return self._stage_map.get(stage, PhaseTier.INTELLIGENT)

    def get_provider(self, stage: str) -> LLMProvider | None:
        """Get the appropriate LLM provider for a pipeline stage.

        Looks up the tier for the stage, then returns the provider for
        that tier. Falls back through the tier hierarchy:
        MECHANICAL → INTELLIGENT → CRITICAL_SAFETY.

        Args:
            stage: Pipeline stage name.

        Returns:
            An LLM provider, or None if no provider is configured.
        """
        tier = self.get_tier(stage)
        fallback_order = _FALLBACK_CHAINS[tier]

        for candidate_tier in fallback_order:
            provider = self._providers.get(candidate_tier)
            if provider is not None:
                if candidate_tier != tier:
                    logger.debug(
                        "Stage %r: tier %s fell back to %s provider",
                        stage,
                        tier.value,
                        candidate_tier.value,
                    )
                return provider

        logger.debug("No provider available for stage %r (tier %s)", stage, tier.value)
        return None

    def call(
        self,
        stage: str,
        prompt: str,
        schema_id: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Make a routed LLM call for the given pipeline stage.

        Args:
            stage: Pipeline stage name.
            prompt: The prompt text.
            schema_id: ID of the expected output schema.
            temperature: Sampling temperature.

        Returns:
            Parsed response dict, or empty dict if no provider available.
        """
        provider = self.get_provider(stage)
        if provider is None:
            logger.warning(
                "No LLM provider for stage %r (tier %s), returning empty",
                stage,
                self.get_tier(stage).value,
            )
            return {}

        tier = self.get_tier(stage)
        logger.info(
            "Routing %r (tier=%s) → %s",
            stage,
            tier.value,
            provider.model_name(),
        )
        return provider.call(prompt, schema_id, temperature)

    def has_provider(self, stage: str) -> bool:
        """Check if a provider is available for the given stage.

        Args:
            stage: Pipeline stage name.

        Returns:
            True if a provider can handle this stage.
        """
        return self.get_provider(stage) is not None

    def summary(self) -> dict[str, str | None]:
        """Return a summary of provider assignments per tier.

        Returns:
            Dict mapping tier names to model names (or None).
        """
        return {
            tier.value: (p.model_name() if p else None)
            for tier, p in self._providers.items()
        }


# Fallback chains: if the primary tier has no provider, try others.
_FALLBACK_CHAINS: dict[PhaseTier, list[PhaseTier]] = {
    PhaseTier.MECHANICAL: [
        PhaseTier.MECHANICAL,
        PhaseTier.INTELLIGENT,
        PhaseTier.CRITICAL_SAFETY,
    ],
    PhaseTier.INTELLIGENT: [
        PhaseTier.INTELLIGENT,
        PhaseTier.CRITICAL_SAFETY,
        PhaseTier.MECHANICAL,
    ],
    PhaseTier.CRITICAL_SAFETY: [
        PhaseTier.CRITICAL_SAFETY,
        PhaseTier.INTELLIGENT,
    ],
}


def create_model_router(
    config: Any,
) -> ModelRouter:
    """Create a ModelRouter from pipeline configuration.

    Reads ``config.llm`` for the default provider and optional
    ``config.llm_routing`` for per-tier overrides.

    Args:
        config: PipelineConfig instance.

    Returns:
        Configured ModelRouter.
    """
    from research_pipeline.llm.providers import (
        create_llm_provider,
    )

    # Build providers per tier from routing config
    routing_cfg = getattr(config, "llm_routing", None)

    if routing_cfg is None or not getattr(routing_cfg, "enabled", False):
        # No routing config — use the single LLM provider for all tiers
        single = create_llm_provider(config.llm)
        return ModelRouter(
            mechanical=single,
            intelligent=single,
            critical_safety=single,
        )

    providers: dict[PhaseTier, LLMProvider | None] = {}
    for tier in PhaseTier:
        tier_cfg = getattr(routing_cfg, tier.value, None)
        if tier_cfg is None:
            providers[tier] = None
            continue
        providers[tier] = _create_tier_provider(tier_cfg)

    stage_overrides = getattr(routing_cfg, "stage_overrides", None)
    override_dict = dict(stage_overrides) if stage_overrides else None

    return ModelRouter(
        mechanical=providers.get(PhaseTier.MECHANICAL),
        intelligent=providers.get(PhaseTier.INTELLIGENT),
        critical_safety=providers.get(PhaseTier.CRITICAL_SAFETY),
        stage_overrides=override_dict,
    )


def _create_tier_provider(tier_cfg: Any) -> LLMProvider | None:
    """Create a provider from a tier-level config block.

    Args:
        tier_cfg: A TierProviderConfig with provider, base_url, etc.

    Returns:
        An LLMProvider instance, or None if not configured.
    """
    from research_pipeline.llm.providers import (
        OllamaProvider,
        OpenAICompatibleProvider,
    )

    provider_name = getattr(tier_cfg, "provider", "").lower()
    if not provider_name:
        return None

    if provider_name == "ollama":
        return OllamaProvider(
            base_url=getattr(tier_cfg, "base_url", "http://localhost:11434"),
            model=getattr(tier_cfg, "model", "llama3.2"),
            max_tokens=getattr(tier_cfg, "max_tokens", 4096),
        )

    if provider_name == "openai":
        return OpenAICompatibleProvider(
            base_url=getattr(tier_cfg, "base_url", "https://api.openai.com/v1"),
            api_key=getattr(tier_cfg, "api_key", ""),
            model=getattr(tier_cfg, "model", "gpt-4o-mini"),
            max_tokens=getattr(tier_cfg, "max_tokens", 4096),
        )

    logger.warning("Unknown tier provider: %s", provider_name)
    return None
