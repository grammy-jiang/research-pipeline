"""Tests for evaluation.coherence_eval — 7-Dimension Coherence Evaluation."""

from __future__ import annotations

import pytest

from research_pipeline.evaluation.coherence_eval import (
    DEFAULT_WEIGHTS,
    Assertion,
    CoherenceDimension,
    CoherenceEvaluator,
    CoherenceIssue,
    CoherenceReport,
    DimensionScore,
    Severity,
    evaluate_coherence_degradation,
    evaluate_contradiction_detection,
    evaluate_cross_session_reasoning,
    evaluate_factual_consistency,
    evaluate_knowledge_update_fidelity,
    evaluate_memory_pressure_resilience,
    evaluate_temporal_ordering,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_dimension_count(self) -> None:
        assert len(CoherenceDimension) == 7

    def test_severity_count(self) -> None:
        assert len(Severity) == 4


# ---------------------------------------------------------------------------
# CoherenceIssue
# ---------------------------------------------------------------------------


class TestCoherenceIssue:
    def test_to_dict(self) -> None:
        issue = CoherenceIssue(
            dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
            severity=Severity.HIGH,
            description="test issue",
            evidence="some evidence",
            session_id="s1",
        )
        d = issue.to_dict()
        assert d["dimension"] == "factual_consistency"
        assert d["severity"] == "high"
        assert d["session_id"] == "s1"

    def test_frozen(self) -> None:
        issue = CoherenceIssue(
            dimension=CoherenceDimension.TEMPORAL_ORDERING,
            severity=Severity.LOW,
            description="x",
        )
        with pytest.raises(AttributeError):
            issue.description = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DimensionScore
# ---------------------------------------------------------------------------


class TestDimensionScore:
    def test_to_dict(self) -> None:
        ds = DimensionScore(
            dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
            score=0.85,
            details="good",
        )
        d = ds.to_dict()
        assert d["score"] == 0.85
        assert d["num_issues"] == 0


# ---------------------------------------------------------------------------
# CoherenceReport
# ---------------------------------------------------------------------------


class TestCoherenceReport:
    def test_composite_score(self) -> None:
        report = CoherenceReport()
        report.dimension_scores[CoherenceDimension.FACTUAL_CONSISTENCY] = (
            DimensionScore(dimension=CoherenceDimension.FACTUAL_CONSISTENCY, score=0.8)
        )
        report.dimension_scores[CoherenceDimension.TEMPORAL_ORDERING] = DimensionScore(
            dimension=CoherenceDimension.TEMPORAL_ORDERING, score=1.0
        )
        # Composite should be weighted average of the dimensions present
        assert 0.0 < report.composite_score <= 1.0

    def test_empty_report(self) -> None:
        report = CoherenceReport()
        assert report.composite_score == 0.0
        assert report.all_issues == []

    def test_critical_issues(self) -> None:
        issue = CoherenceIssue(
            dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
            severity=Severity.CRITICAL,
            description="critical problem",
        )
        report = CoherenceReport()
        report.dimension_scores[CoherenceDimension.FACTUAL_CONSISTENCY] = (
            DimensionScore(
                dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
                score=0.2,
                issues=(issue,),
            )
        )
        assert len(report.critical_issues) == 1

    def test_to_dict(self) -> None:
        report = CoherenceReport()
        d = report.to_dict()
        assert "composite_score" in d
        assert "dimensions" in d


# ---------------------------------------------------------------------------
# evaluate_factual_consistency
# ---------------------------------------------------------------------------


class TestFactualConsistency:
    def test_no_assertions(self) -> None:
        ds = evaluate_factual_consistency([])
        assert ds.score == 1.0

    def test_no_contradictions(self) -> None:
        assertions = [
            Assertion("BERT is a language model"),
            Assertion("GPT is a language model"),
        ]
        ds = evaluate_factual_consistency(assertions, contradiction_pairs=[])
        assert ds.score == 1.0

    def test_with_contradictions(self) -> None:
        assertions = [
            Assertion("Model A outperforms Model B"),
            Assertion("Model B outperforms Model A"),
        ]
        ds = evaluate_factual_consistency(assertions, [(0, 1)])
        assert ds.score < 1.0
        assert len(ds.issues) == 1

    def test_many_contradictions(self) -> None:
        assertions = [Assertion(f"claim {i}") for i in range(10)]
        # Every consecutive pair contradicts
        pairs = [(i, i + 1) for i in range(9)]
        ds = evaluate_factual_consistency(assertions, pairs)
        assert ds.score == 0.0  # heavily penalised


# ---------------------------------------------------------------------------
# evaluate_temporal_ordering
# ---------------------------------------------------------------------------


class TestTemporalOrdering:
    def test_fewer_than_two_events(self) -> None:
        ds = evaluate_temporal_ordering([("evt", 1.0)])
        assert ds.score == 1.0

    def test_correct_order(self) -> None:
        events = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
        ds = evaluate_temporal_ordering(events)
        assert ds.score == 1.0
        assert len(ds.issues) == 0

    def test_one_inversion(self) -> None:
        events = [("b", 2.0), ("a", 1.0), ("c", 3.0)]
        ds = evaluate_temporal_ordering(events)
        assert ds.score < 1.0
        assert len(ds.issues) >= 1

    def test_fully_reversed(self) -> None:
        events = [("c", 3.0), ("b", 2.0), ("a", 1.0)]
        ds = evaluate_temporal_ordering(events)
        assert ds.score == 0.0


# ---------------------------------------------------------------------------
# evaluate_knowledge_update_fidelity
# ---------------------------------------------------------------------------


class TestKnowledgeUpdateFidelity:
    def test_no_updates(self) -> None:
        ds = evaluate_knowledge_update_fidelity([])
        assert ds.score == 1.0

    def test_all_integrated(self) -> None:
        updates = [
            {"old_claim": "X is Y", "new_claim": "X is Z", "integrated": True},
            {"old_claim": "A is B", "new_claim": "A is C", "integrated": True},
        ]
        ds = evaluate_knowledge_update_fidelity(updates)
        assert ds.score == 1.0

    def test_none_integrated(self) -> None:
        updates = [
            {"old_claim": "X", "new_claim": "Y", "integrated": False},
        ]
        ds = evaluate_knowledge_update_fidelity(updates)
        assert ds.score == 0.0
        assert len(ds.issues) == 1

    def test_partial_integration(self) -> None:
        updates = [
            {"old_claim": "A", "new_claim": "B", "integrated": True},
            {"old_claim": "C", "new_claim": "D", "integrated": False},
        ]
        ds = evaluate_knowledge_update_fidelity(updates)
        assert ds.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# evaluate_cross_session_reasoning
# ---------------------------------------------------------------------------


class TestCrossSessionReasoning:
    def test_no_conclusions(self) -> None:
        ds = evaluate_cross_session_reasoning([])
        assert ds.score == 1.0

    def test_all_consistent(self) -> None:
        conclusions = [
            {"session_id": "s1", "conclusion": "X", "consistent_with_prior": True},
            {"session_id": "s2", "conclusion": "Y", "consistent_with_prior": True},
        ]
        ds = evaluate_cross_session_reasoning(conclusions)
        assert ds.score == 1.0

    def test_some_inconsistent(self) -> None:
        conclusions = [
            {"session_id": "s1", "conclusion": "A", "consistent_with_prior": True},
            {"session_id": "s2", "conclusion": "B", "consistent_with_prior": False},
        ]
        ds = evaluate_cross_session_reasoning(conclusions)
        assert ds.score == 0.5
        assert len(ds.issues) == 1
        assert ds.issues[0].session_id == "s2"


# ---------------------------------------------------------------------------
# evaluate_contradiction_detection
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    def test_no_contradictions(self) -> None:
        ds = evaluate_contradiction_detection(0, 0, 0)
        assert ds.score == 1.0

    def test_perfect_detection(self) -> None:
        ds = evaluate_contradiction_detection(5, 5, 0)
        assert ds.score == 1.0

    def test_missed_some(self) -> None:
        ds = evaluate_contradiction_detection(10, 5, 0)
        assert ds.score < 1.0

    def test_false_positives(self) -> None:
        ds = evaluate_contradiction_detection(5, 5, 5)
        assert ds.score < 1.0
        assert any("false positive" in i.description for i in ds.issues)

    def test_no_detections(self) -> None:
        ds = evaluate_contradiction_detection(5, 0, 0)
        assert ds.score == 0.0


# ---------------------------------------------------------------------------
# evaluate_coherence_degradation
# ---------------------------------------------------------------------------


class TestCoherenceDegradation:
    def test_insufficient_data(self) -> None:
        ds = evaluate_coherence_degradation([0.9])
        assert ds.score == 1.0

    def test_no_degradation(self) -> None:
        ds = evaluate_coherence_degradation([0.9, 0.9, 0.9])
        assert ds.score == 1.0

    def test_minor_degradation(self) -> None:
        ds = evaluate_coherence_degradation([0.9, 0.85, 0.82])
        assert 0.9 < ds.score < 1.0

    def test_severe_degradation(self) -> None:
        ds = evaluate_coherence_degradation([0.9, 0.5, 0.2])
        assert ds.score < 0.5
        assert len(ds.issues) >= 1

    def test_zero_start(self) -> None:
        ds = evaluate_coherence_degradation([0.0, 0.0])
        assert ds.score == 1.0  # no drop from zero


# ---------------------------------------------------------------------------
# evaluate_memory_pressure_resilience
# ---------------------------------------------------------------------------


class TestMemoryPressureResilience:
    def test_zero_baseline(self) -> None:
        ds = evaluate_memory_pressure_resilience(0.0, 0.0)
        assert ds.score == 0.0

    def test_no_degradation(self) -> None:
        ds = evaluate_memory_pressure_resilience(0.9, 0.9)
        assert ds.score == 1.0

    def test_minor_drop(self) -> None:
        ds = evaluate_memory_pressure_resilience(0.9, 0.8)
        assert 0.85 < ds.score < 0.95

    def test_severe_drop(self) -> None:
        ds = evaluate_memory_pressure_resilience(0.9, 0.3)
        assert ds.score < 0.5
        assert len(ds.issues) >= 1

    def test_improvement_capped(self) -> None:
        ds = evaluate_memory_pressure_resilience(0.5, 0.6)
        assert ds.score == 1.0  # retention > 1 capped to 1


# ---------------------------------------------------------------------------
# CoherenceEvaluator
# ---------------------------------------------------------------------------


class TestCoherenceEvaluator:
    def test_default_evaluation(self) -> None:
        evaluator = CoherenceEvaluator()
        report = evaluator.evaluate()
        assert isinstance(report, CoherenceReport)
        assert len(report.dimension_scores) == 7
        # All empty inputs → all scores should be high
        assert report.composite_score >= 0.5

    def test_custom_weights(self) -> None:
        weights = {
            CoherenceDimension.FACTUAL_CONSISTENCY: 1.0,
            CoherenceDimension.TEMPORAL_ORDERING: 0.0,
            CoherenceDimension.KNOWLEDGE_UPDATE_FIDELITY: 0.0,
            CoherenceDimension.CROSS_SESSION_REASONING: 0.0,
            CoherenceDimension.CONTRADICTION_DETECTION: 0.0,
            CoherenceDimension.COHERENCE_DEGRADATION: 0.0,
            CoherenceDimension.MEMORY_PRESSURE_RESILIENCE: 0.0,
        }
        evaluator = CoherenceEvaluator(weights=weights)
        assertions = [Assertion("A"), Assertion("B")]
        report = evaluator.evaluate(assertions=assertions, contradiction_pairs=[(0, 1)])
        # Composite should only reflect factual consistency (which has a contradiction)
        assert report.composite_score < 1.0

    def test_full_evaluation(self) -> None:
        evaluator = CoherenceEvaluator()
        report = evaluator.evaluate(
            assertions=[Assertion("A"), Assertion("B")],
            contradiction_pairs=[],
            events=[("e1", 1.0), ("e2", 2.0)],
            knowledge_updates=[
                {"old_claim": "X", "new_claim": "Y", "integrated": True}
            ],
            session_conclusions=[
                {"session_id": "s1", "conclusion": "C", "consistent_with_prior": True}
            ],
            known_contradictions=3,
            detected_contradictions=2,
            false_positive_contradictions=1,
            scores_over_scale=[0.9, 0.88, 0.85],
            baseline_score=0.9,
            constrained_score=0.8,
            memory_limit_fraction=0.5,
        )
        assert 0.0 < report.composite_score <= 1.0
        d = report.to_dict()
        assert "composite_score" in d
        assert len(d["dimensions"]) == 7

    def test_report_serializable(self) -> None:
        evaluator = CoherenceEvaluator()
        report = evaluator.evaluate()
        d = report.to_dict()
        assert isinstance(d["composite_score"], float)
        assert isinstance(d["dimensions"], dict)


# ---------------------------------------------------------------------------
# DEFAULT_WEIGHTS
# ---------------------------------------------------------------------------


class TestDefaultWeights:
    def test_weights_sum_to_one(self) -> None:
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_all_dimensions_covered(self) -> None:
        for dim in CoherenceDimension:
            assert dim in DEFAULT_WEIGHTS
