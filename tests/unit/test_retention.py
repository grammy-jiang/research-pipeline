"""Tests for memory.retention — retention regularization for drift prevention."""

from __future__ import annotations

import pytest

from research_pipeline.memory.retention import (
    DEFAULT_THRESHOLDS,
    DriftScore,
    DriftSeverity,
    DriftThresholds,
    DriftTrend,
    RetentionRegularizer,
    StabilisationAction,
    classify_drift,
    cosine_similarity,
    measure_drift,
    recommend_action,
    text_similarity,
)

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical(self) -> None:
        v = {"a": 0.5, "b": 0.5}
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)

    def test_empty_a(self) -> None:
        assert cosine_similarity({}, {"a": 1.0}) == 0.0

    def test_empty_b(self) -> None:
        assert cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_both_empty(self) -> None:
        assert cosine_similarity({}, {}) == 0.0

    def test_partial_overlap(self) -> None:
        va = {"a": 0.5, "b": 0.5}
        vb = {"a": 0.5, "c": 0.5}
        sim = cosine_similarity(va, vb)
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# text_similarity
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def test_identical_text(self) -> None:
        t = "knowledge graph evaluation metrics"
        assert text_similarity(t, t) == pytest.approx(1.0)

    def test_similar_text(self) -> None:
        a = "knowledge graph evaluation metrics for scientific papers"
        b = "evaluation metrics for knowledge graphs in research papers"
        sim = text_similarity(a, b)
        assert sim > 0.5

    def test_different_text(self) -> None:
        a = "quantum computing entanglement"
        b = "machine learning neural networks"
        sim = text_similarity(a, b)
        assert sim < 0.3

    def test_empty_text(self) -> None:
        assert text_similarity("", "hello") == 0.0
        assert text_similarity("hello", "") == 0.0


# ---------------------------------------------------------------------------
# classify_drift
# ---------------------------------------------------------------------------


class TestClassifyDrift:
    def test_none(self) -> None:
        assert classify_drift(0.9) == DriftSeverity.NONE

    def test_low(self) -> None:
        assert classify_drift(0.7) == DriftSeverity.LOW

    def test_medium(self) -> None:
        assert classify_drift(0.5) == DriftSeverity.MEDIUM

    def test_high(self) -> None:
        assert classify_drift(0.3) == DriftSeverity.HIGH

    def test_critical(self) -> None:
        assert classify_drift(0.1) == DriftSeverity.CRITICAL

    def test_boundary_low(self) -> None:
        assert classify_drift(0.80) == DriftSeverity.NONE
        assert classify_drift(0.799) == DriftSeverity.LOW

    def test_custom_thresholds(self) -> None:
        t = DriftThresholds(low=0.9, medium=0.7, high=0.5, critical=0.3)
        assert classify_drift(0.85, t) == DriftSeverity.LOW
        assert classify_drift(0.95, t) == DriftSeverity.NONE


# ---------------------------------------------------------------------------
# recommend_action
# ---------------------------------------------------------------------------


class TestRecommendAction:
    def test_none_severity(self) -> None:
        assert recommend_action(DriftSeverity.NONE) == StabilisationAction.NONE

    def test_low_severity(self) -> None:
        assert recommend_action(DriftSeverity.LOW) == StabilisationAction.NONE

    def test_medium_severity(self) -> None:
        assert recommend_action(DriftSeverity.MEDIUM) == StabilisationAction.REVIEW

    def test_high_severity(self) -> None:
        assert recommend_action(DriftSeverity.HIGH) == StabilisationAction.CONSOLIDATE

    def test_critical_severity(self) -> None:
        assert recommend_action(DriftSeverity.CRITICAL) == StabilisationAction.ROLLBACK


# ---------------------------------------------------------------------------
# measure_drift
# ---------------------------------------------------------------------------


class TestMeasureDrift:
    def test_basic(self) -> None:
        a = "knowledge graph evaluation metrics papers"
        b = "knowledge graph evaluation metrics papers"
        score = measure_drift(a, b, session_a="run1", session_b="run2")
        assert score.similarity == pytest.approx(1.0)
        assert score.severity == DriftSeverity.NONE

    def test_drift_detected(self) -> None:
        a = "knowledge graph evaluation metrics papers"
        b = "quantum computing entanglement physics"
        score = measure_drift(a, b, session_a="run1", session_b="run2", stage="screen")
        assert score.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
        assert score.stage == "screen"

    def test_drift_magnitude(self) -> None:
        score = measure_drift("hello world", "hello world")
        assert score.drift_magnitude == pytest.approx(0.0)

    def test_to_dict(self) -> None:
        score = measure_drift("a b c", "d e f", session_a="s1", session_b="s2")
        d = score.to_dict()
        assert "similarity" in d
        assert "severity" in d
        assert d["session_a"] == "s1"


# ---------------------------------------------------------------------------
# DriftScore
# ---------------------------------------------------------------------------


class TestDriftScore:
    def test_drift_magnitude(self) -> None:
        ds = DriftScore(
            similarity=0.7,
            severity=DriftSeverity.LOW,
            recommendation=StabilisationAction.NONE,
            session_a="a",
            session_b="b",
        )
        assert ds.drift_magnitude == pytest.approx(0.3)

    def test_to_dict(self) -> None:
        ds = DriftScore(
            similarity=0.5,
            severity=DriftSeverity.MEDIUM,
            recommendation=StabilisationAction.REVIEW,
            session_a="a",
            session_b="b",
            stage="screen",
        )
        d = ds.to_dict()
        assert d["severity"] == "medium"
        assert d["recommendation"] == "review"
        assert d["stage"] == "screen"


# ---------------------------------------------------------------------------
# DriftTrend
# ---------------------------------------------------------------------------


class TestDriftTrend:
    def test_to_dict(self) -> None:
        trend = DriftTrend(
            scores=[0.8, 0.7, 0.6],
            direction="degrading",
            average_similarity=0.7,
            min_similarity=0.6,
            max_similarity=0.8,
        )
        d = trend.to_dict()
        assert d["direction"] == "degrading"
        assert d["measurement_count"] == 3


# ---------------------------------------------------------------------------
# RetentionRegularizer
# ---------------------------------------------------------------------------


class TestRetentionRegularizer:
    def test_record(self) -> None:
        reg = RetentionRegularizer()
        score = reg.record("hello world", "hello world")
        assert score.severity == DriftSeverity.NONE
        assert len(reg.history) == 1

    def test_record_drift(self) -> None:
        reg = RetentionRegularizer()
        score = reg.record(
            "knowledge graph evaluation",
            "quantum computing physics",
            session_a="run1",
            session_b="run2",
        )
        assert score.severity != DriftSeverity.NONE

    def test_history_limit(self) -> None:
        reg = RetentionRegularizer(history_limit=3)
        for i in range(5):
            reg.record(f"text {i}", f"text {i + 1}")
        assert len(reg.history) == 3

    def test_compute_trend_empty(self) -> None:
        reg = RetentionRegularizer()
        assert reg.compute_trend() is None

    def test_compute_trend_single(self) -> None:
        reg = RetentionRegularizer()
        reg.record("hello world", "hello world")
        trend = reg.compute_trend()
        assert trend is not None
        assert trend.direction == "stable"

    def test_compute_trend_stable(self) -> None:
        reg = RetentionRegularizer()
        base = "knowledge graph evaluation metrics"
        for _ in range(5):
            reg.record(base, base)
        trend = reg.compute_trend()
        assert trend is not None
        assert trend.direction == "stable"

    def test_compute_trend_degrading(self) -> None:
        reg = RetentionRegularizer()
        texts = [
            "knowledge graph evaluation metrics papers",
            "knowledge evaluation metrics papers review",
            "evaluation metrics review analysis",
            "metrics review analysis systems",
            "review analysis systems networks",
            "analysis systems networks quantum",
            "systems networks quantum computing",
            "networks quantum computing physics",
        ]
        for i in range(len(texts) - 1):
            reg.record(texts[0], texts[i + 1], session_a="base", session_b=f"s{i}")
        trend = reg.compute_trend()
        assert trend is not None
        assert trend.direction == "degrading"

    def test_compute_trend_window(self) -> None:
        reg = RetentionRegularizer()
        base = "hello world"
        for _ in range(10):
            reg.record(base, base)
        trend = reg.compute_trend(window=3)
        assert trend is not None
        assert len(trend.scores) == 3

    def test_get_stage_drift(self) -> None:
        reg = RetentionRegularizer()
        reg.record("a b", "a b", stage="screen")
        reg.record("c d", "c d", stage="download")
        reg.record("e f", "e f", stage="screen")
        assert len(reg.get_stage_drift("screen")) == 2
        assert len(reg.get_stage_drift("download")) == 1

    def test_needs_stabilisation_false(self) -> None:
        reg = RetentionRegularizer()
        reg.record("hello world", "hello world")
        assert not reg.needs_stabilisation()

    def test_needs_stabilisation_true(self) -> None:
        reg = RetentionRegularizer()
        reg.record(
            "knowledge graphs papers",
            "quantum computing entanglement physics chemistry",
        )
        assert reg.needs_stabilisation()

    def test_compute_regularization_penalty_no_drift(self) -> None:
        reg = RetentionRegularizer()
        penalty = reg.compute_regularization_penalty("hello world", "hello world")
        assert penalty == pytest.approx(0.0)

    def test_compute_regularization_penalty_full_drift(self) -> None:
        reg = RetentionRegularizer()
        penalty = reg.compute_regularization_penalty("", "hello")
        assert penalty == pytest.approx(1.0)

    def test_compute_regularization_penalty_alpha(self) -> None:
        reg = RetentionRegularizer()
        penalty = reg.compute_regularization_penalty("", "hello", alpha=0.5)
        assert penalty == pytest.approx(0.5)

    def test_regularize_score_no_drift(self) -> None:
        reg = RetentionRegularizer()
        adj = reg.regularize_score(0.9, "hello world", "hello world")
        assert adj == pytest.approx(0.9)

    def test_regularize_score_with_drift(self) -> None:
        reg = RetentionRegularizer()
        adj = reg.regularize_score(
            0.9, "knowledge graphs", "quantum physics", alpha=0.5
        )
        assert adj < 0.9

    def test_regularize_score_floor_zero(self) -> None:
        reg = RetentionRegularizer()
        adj = reg.regularize_score(0.1, "", "hello", alpha=1.0)
        assert adj >= 0.0

    def test_summary(self) -> None:
        reg = RetentionRegularizer()
        reg.record("hello world", "hello world")
        s = reg.summary()
        assert s["total_measurements"] == 1
        assert isinstance(s["needs_stabilisation"], bool)
        assert s["trend"] is not None

    def test_thresholds_property(self) -> None:
        t = DriftThresholds(low=0.9, medium=0.7, high=0.5, critical=0.3)
        reg = RetentionRegularizer(t)
        assert reg.thresholds.low == 0.9


# ---------------------------------------------------------------------------
# DriftThresholds
# ---------------------------------------------------------------------------


class TestDriftThresholds:
    def test_defaults(self) -> None:
        t = DEFAULT_THRESHOLDS
        assert t.low == 0.80
        assert t.medium == 0.60
        assert t.high == 0.40
        assert t.critical == 0.20

    def test_custom(self) -> None:
        t = DriftThresholds(low=0.95, medium=0.85, high=0.70, critical=0.50)
        assert t.low == 0.95
        assert t.critical == 0.50
