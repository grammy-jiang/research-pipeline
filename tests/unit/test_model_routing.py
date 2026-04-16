"""Tests for phase-aware model routing."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.llm.base import LLMProvider
from research_pipeline.llm.routing import (
    STAGE_TIER_MAP,
    ModelRouter,
    PhaseTier,
    _create_tier_provider,
    create_model_router,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider(LLMProvider):
    """Test LLM provider with configurable responses."""

    def __init__(self, name: str = "fake", response: dict[str, Any] | None = None):
        self._name = name
        self._response = response or {"result": "ok"}
        self.calls: list[tuple[str, str, float]] = []

    def call(
        self, prompt: str, schema_id: str, temperature: float = 0.0
    ) -> dict[str, Any]:
        self.calls.append((prompt, schema_id, temperature))
        return self._response

    def model_name(self) -> str:
        return self._name


def _make_config(
    llm_enabled: bool = True,
    routing_enabled: bool = False,
    mechanical_provider: str = "",
    intelligent_provider: str = "",
    critical_provider: str = "",
    stage_overrides: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock PipelineConfig."""
    cfg = MagicMock()
    cfg.llm.enabled = llm_enabled
    cfg.llm.provider = "ollama"
    cfg.llm.base_url = "http://localhost:11434"
    cfg.llm.api_key = ""
    cfg.llm.model = "test-model"
    cfg.llm.max_tokens = 1024

    routing = MagicMock()
    routing.enabled = routing_enabled
    routing.stage_overrides = stage_overrides or {}

    for tier, prov_str in [
        ("mechanical", mechanical_provider),
        ("intelligent", intelligent_provider),
        ("critical_safety", critical_provider),
    ]:
        tier_cfg = MagicMock()
        tier_cfg.provider = prov_str
        tier_cfg.base_url = ""
        tier_cfg.api_key = ""
        tier_cfg.model = f"{prov_str}-model" if prov_str else ""
        tier_cfg.max_tokens = 2048
        setattr(routing, tier, tier_cfg)

    cfg.llm_routing = routing
    return cfg


# ---------------------------------------------------------------------------
# PhaseTier enum tests
# ---------------------------------------------------------------------------


class TestPhaseTier:
    """Tests for PhaseTier enum."""

    def test_values(self) -> None:
        assert PhaseTier.MECHANICAL.value == "mechanical"
        assert PhaseTier.INTELLIGENT.value == "intelligent"
        assert PhaseTier.CRITICAL_SAFETY.value == "critical_safety"

    def test_from_string(self) -> None:
        assert PhaseTier("mechanical") is PhaseTier.MECHANICAL
        assert PhaseTier("intelligent") is PhaseTier.INTELLIGENT
        assert PhaseTier("critical_safety") is PhaseTier.CRITICAL_SAFETY

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            PhaseTier("invalid")

    def test_str_comparison(self) -> None:
        assert PhaseTier.MECHANICAL == "mechanical"
        assert PhaseTier.INTELLIGENT == "intelligent"

    def test_all_tiers_in_enum(self) -> None:
        assert len(PhaseTier) == 3


# ---------------------------------------------------------------------------
# STAGE_TIER_MAP tests
# ---------------------------------------------------------------------------


class TestStageTierMap:
    """Tests for the default stage→tier mapping."""

    def test_mechanical_stages(self) -> None:
        mechanical = [
            "plan",
            "search",
            "download",
            "convert",
            "convert_rough",
            "convert_fine",
            "extract",
            "index",
            "export_html",
        ]
        for stage in mechanical:
            assert STAGE_TIER_MAP[stage] == PhaseTier.MECHANICAL, stage

    def test_intelligent_stages(self) -> None:
        intelligent = [
            "screen",
            "summarize",
            "expand",
            "quality",
            "analyze",
            "aggregate",
        ]
        for stage in intelligent:
            assert STAGE_TIER_MAP[stage] == PhaseTier.INTELLIGENT, stage

    def test_critical_safety_stages(self) -> None:
        critical = ["validate", "compare", "security_gate"]
        for stage in critical:
            assert STAGE_TIER_MAP[stage] == PhaseTier.CRITICAL_SAFETY, stage

    def test_all_stages_classified(self) -> None:
        assert len(STAGE_TIER_MAP) >= 18

    def test_all_values_are_phase_tiers(self) -> None:
        for stage, tier in STAGE_TIER_MAP.items():
            assert isinstance(tier, PhaseTier), stage


# ---------------------------------------------------------------------------
# ModelRouter tests
# ---------------------------------------------------------------------------


class TestModelRouterInit:
    """Tests for ModelRouter initialization."""

    def test_empty_init(self) -> None:
        router = ModelRouter()
        assert router.summary() == {
            "mechanical": None,
            "intelligent": None,
            "critical_safety": None,
        }

    def test_single_provider(self) -> None:
        prov = _FakeProvider("single")
        router = ModelRouter(intelligent=prov)
        assert router.summary()["intelligent"] == "single"
        assert router.summary()["mechanical"] is None

    def test_all_providers(self) -> None:
        m = _FakeProvider("mech")
        i = _FakeProvider("intel")
        c = _FakeProvider("crit")
        router = ModelRouter(mechanical=m, intelligent=i, critical_safety=c)
        summary = router.summary()
        assert summary["mechanical"] == "mech"
        assert summary["intelligent"] == "intel"
        assert summary["critical_safety"] == "crit"

    def test_stage_overrides(self) -> None:
        router = ModelRouter(stage_overrides={"plan": "intelligent"})
        assert router.get_tier("plan") == PhaseTier.INTELLIGENT

    def test_invalid_stage_override_ignored(self) -> None:
        router = ModelRouter(stage_overrides={"plan": "nonexistent"})
        # Should keep the default mapping
        assert router.get_tier("plan") == PhaseTier.MECHANICAL


class TestModelRouterGetTier:
    """Tests for get_tier method."""

    def test_known_stage(self) -> None:
        router = ModelRouter()
        assert router.get_tier("plan") == PhaseTier.MECHANICAL
        assert router.get_tier("screen") == PhaseTier.INTELLIGENT
        assert router.get_tier("validate") == PhaseTier.CRITICAL_SAFETY

    def test_unknown_stage_defaults_to_intelligent(self) -> None:
        router = ModelRouter()
        assert router.get_tier("unknown_stage") == PhaseTier.INTELLIGENT

    def test_overridden_stage(self) -> None:
        router = ModelRouter(stage_overrides={"plan": "critical_safety"})
        assert router.get_tier("plan") == PhaseTier.CRITICAL_SAFETY

    def test_stage_map_property_returns_copy(self) -> None:
        router = ModelRouter()
        stage_map = router.stage_map
        stage_map["plan"] = PhaseTier.CRITICAL_SAFETY
        # Original should not change
        assert router.get_tier("plan") == PhaseTier.MECHANICAL


class TestModelRouterGetProvider:
    """Tests for get_provider with fallback logic."""

    def test_direct_match(self) -> None:
        mech = _FakeProvider("mech")
        router = ModelRouter(mechanical=mech)
        assert router.get_provider("plan") is mech

    def test_mechanical_falls_back_to_intelligent(self) -> None:
        intel = _FakeProvider("intel")
        router = ModelRouter(intelligent=intel)
        # plan is MECHANICAL, but no mechanical provider → fall back to intelligent
        assert router.get_provider("plan") is intel

    def test_mechanical_falls_back_to_critical(self) -> None:
        crit = _FakeProvider("crit")
        router = ModelRouter(critical_safety=crit)
        assert router.get_provider("plan") is crit

    def test_intelligent_falls_back_to_critical(self) -> None:
        crit = _FakeProvider("crit")
        router = ModelRouter(critical_safety=crit)
        assert router.get_provider("screen") is crit

    def test_intelligent_falls_back_to_mechanical(self) -> None:
        mech = _FakeProvider("mech")
        router = ModelRouter(mechanical=mech)
        # intelligent has no provider, critical has none → fall back to mechanical
        assert router.get_provider("screen") is mech

    def test_critical_does_not_fall_back_to_mechanical(self) -> None:
        mech = _FakeProvider("mech")
        router = ModelRouter(mechanical=mech)
        # critical_safety only falls back to intelligent, NOT mechanical
        assert router.get_provider("validate") is None

    def test_critical_falls_back_to_intelligent(self) -> None:
        intel = _FakeProvider("intel")
        router = ModelRouter(intelligent=intel)
        assert router.get_provider("validate") is intel

    def test_no_providers_returns_none(self) -> None:
        router = ModelRouter()
        assert router.get_provider("plan") is None
        assert router.get_provider("screen") is None
        assert router.get_provider("validate") is None


class TestModelRouterCall:
    """Tests for the call method."""

    def test_routes_to_correct_provider(self) -> None:
        mech = _FakeProvider("mech", {"phase": "mechanical"})
        intel = _FakeProvider("intel", {"phase": "intelligent"})
        router = ModelRouter(mechanical=mech, intelligent=intel)

        result = router.call("plan", "prompt", "schema1")
        assert result == {"phase": "mechanical"}
        assert len(mech.calls) == 1

        result = router.call("screen", "prompt", "schema2")
        assert result == {"phase": "intelligent"}
        assert len(intel.calls) == 1

    def test_call_with_no_provider_returns_empty(self) -> None:
        router = ModelRouter()
        result = router.call("plan", "prompt", "schema")
        assert result == {}

    def test_call_passes_parameters(self) -> None:
        prov = _FakeProvider("test")
        router = ModelRouter(intelligent=prov)
        router.call("screen", "my prompt", "my_schema", temperature=0.7)
        assert prov.calls == [("my prompt", "my_schema", 0.7)]

    def test_call_fallback_route(self) -> None:
        intel = _FakeProvider("intel", {"via": "fallback"})
        router = ModelRouter(intelligent=intel)
        # MECHANICAL stage, but only intelligent provider → fallback
        result = router.call("plan", "prompt", "schema")
        assert result == {"via": "fallback"}
        assert len(intel.calls) == 1


class TestModelRouterHasProvider:
    """Tests for has_provider method."""

    def test_has_direct_provider(self) -> None:
        mech = _FakeProvider("mech")
        router = ModelRouter(mechanical=mech)
        assert router.has_provider("plan") is True

    def test_has_fallback_provider(self) -> None:
        intel = _FakeProvider("intel")
        router = ModelRouter(intelligent=intel)
        # plan is MECHANICAL but can fall back to intel
        assert router.has_provider("plan") is True

    def test_no_provider(self) -> None:
        router = ModelRouter()
        assert router.has_provider("plan") is False

    def test_critical_no_fallback_to_mechanical(self) -> None:
        mech = _FakeProvider("mech")
        router = ModelRouter(mechanical=mech)
        assert router.has_provider("validate") is False


class TestModelRouterSummary:
    """Tests for summary method."""

    def test_empty_summary(self) -> None:
        router = ModelRouter()
        assert router.summary() == {
            "mechanical": None,
            "intelligent": None,
            "critical_safety": None,
        }

    def test_full_summary(self) -> None:
        router = ModelRouter(
            mechanical=_FakeProvider("m"),
            intelligent=_FakeProvider("i"),
            critical_safety=_FakeProvider("c"),
        )
        assert router.summary() == {
            "mechanical": "m",
            "intelligent": "i",
            "critical_safety": "c",
        }


# ---------------------------------------------------------------------------
# create_model_router tests
# ---------------------------------------------------------------------------


class TestCreateModelRouter:
    """Tests for create_model_router factory function."""

    @patch("research_pipeline.llm.providers.create_llm_provider")
    def test_no_routing_uses_single_provider(self, mock_create: MagicMock) -> None:
        prov = _FakeProvider("single")
        mock_create.return_value = prov
        cfg = _make_config(routing_enabled=False)
        router = create_model_router(cfg)
        summary = router.summary()
        assert summary["mechanical"] == "single"
        assert summary["intelligent"] == "single"
        assert summary["critical_safety"] == "single"

    @patch("research_pipeline.llm.providers.create_llm_provider")
    def test_no_routing_llm_disabled(self, mock_create: MagicMock) -> None:
        mock_create.return_value = None
        cfg = _make_config(llm_enabled=False, routing_enabled=False)
        router = create_model_router(cfg)
        assert all(v is None for v in router.summary().values())

    def test_routing_enabled_ollama_mechanical(self) -> None:
        cfg = _make_config(routing_enabled=True, mechanical_provider="ollama")
        with patch("research_pipeline.llm.providers.OllamaProvider") as MockOllama:
            MockOllama.return_value = _FakeProvider("oll")
            router = create_model_router(cfg)
            assert router.summary()["mechanical"] == "oll"

    def test_routing_enabled_openai_intelligent(self) -> None:
        cfg = _make_config(routing_enabled=True, intelligent_provider="openai")
        with patch(
            "research_pipeline.llm.providers.OpenAICompatibleProvider"
        ) as MockOAI:
            MockOAI.return_value = _FakeProvider("oai")
            router = create_model_router(cfg)
            assert router.summary()["intelligent"] == "oai"

    def test_routing_with_stage_overrides(self) -> None:
        cfg = _make_config(
            routing_enabled=True,
            intelligent_provider="openai",
            stage_overrides={"plan": "intelligent"},
        )
        with patch(
            "research_pipeline.llm.providers.OpenAICompatibleProvider"
        ) as MockOAI:
            MockOAI.return_value = _FakeProvider("oai")
            router = create_model_router(cfg)
            assert router.get_tier("plan") == PhaseTier.INTELLIGENT

    @patch("research_pipeline.llm.providers.create_llm_provider")
    def test_no_routing_attr_uses_single(self, mock_create: MagicMock) -> None:
        """Config without llm_routing attribute uses single provider."""
        prov = _FakeProvider("single")
        mock_create.return_value = prov
        cfg = MagicMock(spec=[])  # no attributes
        cfg.llm = MagicMock()
        cfg.llm.enabled = True
        # Accessing llm_routing should raise AttributeError → fallback to single
        router = create_model_router(cfg)
        assert router.summary()["mechanical"] == "single"


# ---------------------------------------------------------------------------
# _create_tier_provider tests
# ---------------------------------------------------------------------------


class TestCreateTierProvider:
    """Tests for _create_tier_provider."""

    def test_empty_provider_returns_none(self) -> None:
        cfg = MagicMock()
        cfg.provider = ""
        assert _create_tier_provider(cfg) is None

    def test_ollama_provider(self) -> None:
        cfg = MagicMock()
        cfg.provider = "ollama"
        cfg.base_url = "http://my-ollama:11434"
        cfg.model = "mistral"
        cfg.max_tokens = 2048
        with patch("research_pipeline.llm.providers.OllamaProvider") as MockOllama:
            MockOllama.return_value = _FakeProvider("oll")
            result = _create_tier_provider(cfg)
            assert result is not None
            MockOllama.assert_called_once()

    def test_openai_provider(self) -> None:
        cfg = MagicMock()
        cfg.provider = "openai"
        cfg.base_url = "https://api.openai.com/v1"
        cfg.api_key = "sk-test"
        cfg.model = "gpt-4o"
        cfg.max_tokens = 4096
        with patch(
            "research_pipeline.llm.providers.OpenAICompatibleProvider"
        ) as MockOAI:
            MockOAI.return_value = _FakeProvider("oai")
            result = _create_tier_provider(cfg)
            assert result is not None
            MockOAI.assert_called_once()

    def test_unknown_provider_returns_none(self) -> None:
        cfg = MagicMock()
        cfg.provider = "unknown_provider"
        assert _create_tier_provider(cfg) is None

    def test_case_insensitive_provider(self) -> None:
        cfg = MagicMock()
        cfg.provider = "OLLAMA"
        cfg.base_url = ""
        cfg.model = ""
        cfg.max_tokens = 4096
        with patch("research_pipeline.llm.providers.OllamaProvider") as MockOllama:
            MockOllama.return_value = _FakeProvider("oll")
            result = _create_tier_provider(cfg)
            assert result is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for model routing."""

    def test_multiple_calls_same_stage(self) -> None:
        prov = _FakeProvider("test")
        router = ModelRouter(intelligent=prov)
        for _ in range(10):
            router.call("screen", "prompt", "schema")
        assert len(prov.calls) == 10

    def test_all_stages_in_map_can_be_routed(self) -> None:
        prov = _FakeProvider("universal")
        router = ModelRouter(mechanical=prov, intelligent=prov, critical_safety=prov)
        for stage in STAGE_TIER_MAP:
            assert router.has_provider(stage) is True

    def test_custom_stage_via_override(self) -> None:
        prov = _FakeProvider("test")
        router = ModelRouter(
            intelligent=prov,
            stage_overrides={"my_custom_stage": "intelligent"},
        )
        assert router.get_tier("my_custom_stage") == PhaseTier.INTELLIGENT
        assert router.has_provider("my_custom_stage") is True

    def test_concurrent_routers_independent(self) -> None:
        p1 = _FakeProvider("r1")
        p2 = _FakeProvider("r2")
        r1 = ModelRouter(mechanical=p1)
        r2 = ModelRouter(mechanical=p2)
        r1.call("plan", "prompt1", "schema")
        r2.call("plan", "prompt2", "schema")
        assert len(p1.calls) == 1
        assert len(p2.calls) == 1
        assert p1.calls[0][0] == "prompt1"
        assert p2.calls[0][0] == "prompt2"
