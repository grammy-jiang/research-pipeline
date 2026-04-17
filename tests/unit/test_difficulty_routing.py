"""Tests for llm.difficulty_routing — adaptive difficulty routing."""

from __future__ import annotations

import pytest

from research_pipeline.llm.difficulty_routing import (
    ComplexityFeatures,
    DifficultyLevel,
    DifficultyRouter,
    DifficultyScore,
    RoutingDecision,
    RoutingTarget,
    extract_features,
    score_difficulty,
)


# ---------------------------------------------------------------------------
# ComplexityFeatures
# ---------------------------------------------------------------------------


class TestComplexityFeatures:
    def test_default(self) -> None:
        f = ComplexityFeatures()
        assert f.token_count == 0
        assert f.has_code is False

    def test_to_dict(self) -> None:
        f = ComplexityFeatures(token_count=10, unique_ratio=0.8)
        d = f.to_dict()
        assert d["token_count"] == 10
        assert d["unique_ratio"] == 0.8


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_empty(self) -> None:
        f = extract_features("")
        assert f.token_count == 0
        assert f.unique_ratio == 0.0

    def test_simple_query(self) -> None:
        f = extract_features("what is a transformer model?")
        assert f.token_count == 5
        assert f.question_count == 1
        assert f.technical_density > 0

    def test_code_detection(self) -> None:
        f = extract_features("```python\ndef foo():\n  pass\n```")
        assert f.has_code is True

    def test_no_code(self) -> None:
        f = extract_features("simple question")
        assert f.has_code is False

    def test_technical_terms(self) -> None:
        f = extract_features(
            "transformer attention embedding gradient backpropagation"
        )
        assert f.technical_density > 0.5

    def test_reasoning_indicators(self) -> None:
        f = extract_features(
            "because therefore however although evaluate synthesize"
        )
        assert f.reasoning_depth >= 4

    def test_unique_ratio(self) -> None:
        f = extract_features("the the the the")
        assert f.unique_ratio == 0.25

    def test_avg_word_length(self) -> None:
        f = extract_features("ab cd ef")
        assert f.avg_word_length == 2.0


# ---------------------------------------------------------------------------
# score_difficulty
# ---------------------------------------------------------------------------


class TestScoreDifficulty:
    def test_trivial_query(self) -> None:
        s = score_difficulty("hi")
        assert s.level in (DifficultyLevel.TRIVIAL, DifficultyLevel.EASY)
        assert s.target == RoutingTarget.LOCAL

    def test_hard_query(self) -> None:
        s = score_difficulty(
            "Compare the trade-offs between transformer and recurrent "
            "encoder architectures for embedding generation, analyzing "
            "gradient flow, regularization strategies, and cross-entropy "
            "loss convergence. Evaluate the implications of attention "
            "mechanism depth on benchmark performance because deeper "
            "models demonstrate better correlation with human evaluation. "
            "What are the necessary and sufficient conditions?"
        )
        assert s.level in (DifficultyLevel.HARD, DifficultyLevel.EXPERT)
        assert s.target in (RoutingTarget.CLOUD, RoutingTarget.PREMIUM)

    def test_score_range(self) -> None:
        s = score_difficulty("medium complexity query about transformers")
        assert 0.0 <= s.score <= 1.0

    def test_confidence_range(self) -> None:
        s = score_difficulty("test query")
        assert 0.0 <= s.confidence <= 1.0

    def test_to_dict(self) -> None:
        s = score_difficulty("test")
        d = s.to_dict()
        assert "level" in d
        assert "score" in d
        assert "target" in d
        assert "features" in d

    def test_reasoning_non_empty_for_complex(self) -> None:
        s = score_difficulty(
            "evaluate the methodology and analyze the implications "
            "of transformer attention mechanism however the results "
            "demonstrate correlation because of gradient regularization"
        )
        assert s.reasoning != ""

    def test_code_increases_difficulty(self) -> None:
        s1 = score_difficulty("explain this")
        s2 = score_difficulty("explain this ```python\ndef foo(): pass```")
        assert s2.score >= s1.score


# ---------------------------------------------------------------------------
# DifficultyLevel and RoutingTarget
# ---------------------------------------------------------------------------


class TestEnums:
    def test_difficulty_levels(self) -> None:
        assert len(DifficultyLevel) == 5

    def test_routing_targets(self) -> None:
        assert len(RoutingTarget) == 3


# ---------------------------------------------------------------------------
# DifficultyRouter
# ---------------------------------------------------------------------------


class TestDifficultyRouter:
    def test_basic_routing(self) -> None:
        router = DifficultyRouter()
        decision = router.route("hello")
        assert decision.target == RoutingTarget.LOCAL
        assert not decision.override

    def test_stage_override(self) -> None:
        router = DifficultyRouter(
            stage_overrides={"security_gate": RoutingTarget.PREMIUM}
        )
        decision = router.route("simple check", stage="security_gate")
        assert decision.target == RoutingTarget.PREMIUM
        assert decision.override is True

    def test_custom_routing(self) -> None:
        router = DifficultyRouter(
            custom_routing={DifficultyLevel.TRIVIAL: RoutingTarget.CLOUD}
        )
        decision = router.route("hi")
        assert decision.target == RoutingTarget.CLOUD

    def test_history_tracking(self) -> None:
        router = DifficultyRouter()
        router.route("hello")
        router.route("world")
        assert len(router.history) == 2

    def test_query_hash(self) -> None:
        router = DifficultyRouter()
        d = router.route("test")
        assert len(d.query_hash) == 16

    def test_no_override_without_stage(self) -> None:
        router = DifficultyRouter(
            stage_overrides={"screen": RoutingTarget.PREMIUM}
        )
        decision = router.route("test")
        assert not decision.override


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_to_dict(self) -> None:
        features = ComplexityFeatures(token_count=5)
        difficulty = DifficultyScore(
            level=DifficultyLevel.EASY,
            score=0.3,
            features=features,
            target=RoutingTarget.LOCAL,
        )
        d = RoutingDecision(
            query_hash="abc123",
            target=RoutingTarget.LOCAL,
            difficulty=difficulty,
        )
        result = d.to_dict()
        assert result["target"] == "local"
        assert result["query_hash"] == "abc123"


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------


class TestCostSummary:
    def test_empty(self) -> None:
        router = DifficultyRouter()
        s = router.cost_summary()
        assert s["total_queries"] == 0

    def test_all_local_saves_cost(self) -> None:
        router = DifficultyRouter()
        for _ in range(5):
            router.route("hi")
        s = router.cost_summary()
        assert s["local"] == 5
        assert s["estimated_savings"] > 0.0

    def test_mixed_routing(self) -> None:
        router = DifficultyRouter(
            stage_overrides={"validate": RoutingTarget.PREMIUM}
        )
        router.route("hello")
        router.route("complex transformer analysis", stage="validate")
        s = router.cost_summary()
        assert s["total_queries"] == 2
        assert s["overrides"] == 1
