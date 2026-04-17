"""Tests for llm.consensus — multi-model consensus engine."""

from __future__ import annotations

import pytest

from research_pipeline.llm.consensus import (
    AggregationStrategy,
    ConsensusEngine,
    ConsensusResult,
    DisagreementSeverity,
    ModelResponse,
    classify_disagreement,
    hash_prompt,
)


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_creation(self) -> None:
        r = ModelResponse(model_id="gpt-4o", verdict=True, confidence=0.9)
        assert r.model_id == "gpt-4o"
        assert r.verdict is True
        assert r.confidence == 0.9

    def test_frozen(self) -> None:
        r = ModelResponse(model_id="a", verdict=False)
        with pytest.raises(AttributeError):
            r.model_id = "b"  # type: ignore[misc]

    def test_with_reasoning(self) -> None:
        r = ModelResponse(model_id="m", verdict=0.8, reasoning="because X")
        assert r.reasoning == "because X"


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------


class TestConsensusResult:
    def test_to_dict(self) -> None:
        r = ConsensusResult(
            final_verdict=True,
            strategy=AggregationStrategy.MAJORITY,
            agreement_ratio=1.0,
            responses=[
                ModelResponse(model_id="a", verdict=True),
                ModelResponse(model_id="b", verdict=True),
            ],
        )
        d = r.to_dict()
        assert d["final_verdict"] is True
        assert d["strategy"] == "majority"
        assert d["num_models"] == 2
        assert d["model_ids"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Disagreement classification
# ---------------------------------------------------------------------------


class TestClassifyDisagreement:
    def test_empty(self) -> None:
        assert classify_disagreement([]) == DisagreementSeverity.NONE

    def test_binary_unanimous(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.NONE

    def test_binary_split(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=False),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.HIGH

    def test_binary_slight_disagreement(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
            ModelResponse(model_id="c", verdict=True),
            ModelResponse(model_id="d", verdict=True),
            ModelResponse(model_id="e", verdict=False),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.LOW

    def test_numeric_close(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict=0.85),
            ModelResponse(model_id="b", verdict=0.87),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.NONE

    def test_numeric_moderate(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict=0.9),
            ModelResponse(model_id="b", verdict=0.6),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.MODERATE

    def test_numeric_high(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict=0.95),
            ModelResponse(model_id="b", verdict=0.2),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.HIGH

    def test_label_unanimous(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict="high"),
            ModelResponse(model_id="b", verdict="high"),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.NONE

    def test_label_mixed(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict="high"),
            ModelResponse(model_id="b", verdict="low"),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.HIGH

    def test_label_partial(self) -> None:
        responses = [
            ModelResponse(model_id="a", verdict="high"),
            ModelResponse(model_id="b", verdict="high"),
            ModelResponse(model_id="c", verdict="low"),
        ]
        assert classify_disagreement(responses) == DisagreementSeverity.MODERATE

    def test_single_response(self) -> None:
        responses = [ModelResponse(model_id="a", verdict=0.5)]
        assert classify_disagreement(responses) == DisagreementSeverity.NONE


# ---------------------------------------------------------------------------
# hash_prompt
# ---------------------------------------------------------------------------


class TestHashPrompt:
    def test_deterministic(self) -> None:
        h1 = hash_prompt("test prompt")
        h2 = hash_prompt("test prompt")
        assert h1 == h2

    def test_different_prompts(self) -> None:
        h1 = hash_prompt("prompt A")
        h2 = hash_prompt("prompt B")
        assert h1 != h2

    def test_length(self) -> None:
        assert len(hash_prompt("x")) == 16


# ---------------------------------------------------------------------------
# ConsensusEngine — binary verdicts
# ---------------------------------------------------------------------------


class TestConsensusEngineBinary:
    def test_majority_true(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.MAJORITY)
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
            ModelResponse(model_id="c", verdict=False),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict is True
        assert result.agreement_ratio > 0.5

    def test_majority_false(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.MAJORITY)
        responses = [
            ModelResponse(model_id="a", verdict=False),
            ModelResponse(model_id="b", verdict=False),
            ModelResponse(model_id="c", verdict=True),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict is False

    def test_weighted_high_confidence_wins(self) -> None:
        engine = ConsensusEngine(
            strategy=AggregationStrategy.WEIGHTED,
            model_weights={"a": 2.0, "b": 1.0},
        )
        responses = [
            ModelResponse(model_id="a", verdict=True, confidence=0.95),
            ModelResponse(model_id="b", verdict=False, confidence=0.5),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict is True

    def test_unanimous_all_agree(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.UNANIMOUS)
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict is True
        assert result.disagreement == DisagreementSeverity.NONE

    def test_unanimous_disagree_fallback(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.UNANIMOUS)
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
            ModelResponse(model_id="c", verdict=False),
        ]
        result = engine.evaluate(responses)
        # Falls back to majority → True
        assert result.final_verdict is True


# ---------------------------------------------------------------------------
# ConsensusEngine — numeric verdicts
# ---------------------------------------------------------------------------


class TestConsensusEngineNumeric:
    def test_median(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.MEDIAN)
        responses = [
            ModelResponse(model_id="a", verdict=0.3),
            ModelResponse(model_id="b", verdict=0.8),
            ModelResponse(model_id="c", verdict=0.9),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict == 0.8

    def test_weighted_mean(self) -> None:
        engine = ConsensusEngine(
            strategy=AggregationStrategy.WEIGHTED,
            model_weights={"a": 2.0, "b": 1.0},
        )
        responses = [
            ModelResponse(model_id="a", verdict=0.8, confidence=1.0),
            ModelResponse(model_id="b", verdict=0.4, confidence=1.0),
        ]
        result = engine.evaluate(responses)
        # weighted: (0.8*2 + 0.4*1) / (2+1) = 2.0/3 ≈ 0.667
        assert 0.65 < result.final_verdict < 0.68  # type: ignore[operator]

    def test_majority_defaults_to_median(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.MAJORITY)
        responses = [
            ModelResponse(model_id="a", verdict=0.5),
            ModelResponse(model_id="b", verdict=0.7),
            ModelResponse(model_id="c", verdict=0.9),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict == 0.7


# ---------------------------------------------------------------------------
# ConsensusEngine — label verdicts
# ---------------------------------------------------------------------------


class TestConsensusEngineLabel:
    def test_majority_label(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.MAJORITY)
        responses = [
            ModelResponse(model_id="a", verdict="high"),
            ModelResponse(model_id="b", verdict="high"),
            ModelResponse(model_id="c", verdict="low"),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict == "high"

    def test_unanimous_label(self) -> None:
        engine = ConsensusEngine(strategy=AggregationStrategy.UNANIMOUS)
        responses = [
            ModelResponse(model_id="a", verdict="yes"),
            ModelResponse(model_id="b", verdict="yes"),
        ]
        result = engine.evaluate(responses)
        assert result.final_verdict == "yes"


# ---------------------------------------------------------------------------
# ConsensusEngine — error handling
# ---------------------------------------------------------------------------


class TestConsensusEngineErrors:
    def test_too_few_models(self) -> None:
        engine = ConsensusEngine(min_models=3)
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
        ]
        with pytest.raises(ValueError, match="at least 3"):
            engine.evaluate(responses)

    def test_min_models_1(self) -> None:
        engine = ConsensusEngine(min_models=1)
        responses = [ModelResponse(model_id="a", verdict=True)]
        result = engine.evaluate(responses)
        assert result.final_verdict is True


# ---------------------------------------------------------------------------
# ConsensusEngine — human review
# ---------------------------------------------------------------------------


class TestHumanReview:
    def test_no_review_on_agreement(self) -> None:
        engine = ConsensusEngine(
            human_review_threshold=DisagreementSeverity.HIGH,
        )
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
        ]
        result = engine.evaluate(responses)
        assert not result.needs_human_review

    def test_review_on_high_disagreement(self) -> None:
        engine = ConsensusEngine(
            human_review_threshold=DisagreementSeverity.HIGH,
        )
        responses = [
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=False),
        ]
        result = engine.evaluate(responses)
        assert result.needs_human_review

    def test_review_threshold_moderate(self) -> None:
        engine = ConsensusEngine(
            human_review_threshold=DisagreementSeverity.MODERATE,
        )
        responses = [
            ModelResponse(model_id="a", verdict=0.9),
            ModelResponse(model_id="b", verdict=0.6),
        ]
        result = engine.evaluate(responses)
        assert result.needs_human_review


# ---------------------------------------------------------------------------
# ConsensusEngine — history and summary
# ---------------------------------------------------------------------------


class TestConsensusHistory:
    def test_history_tracking(self) -> None:
        engine = ConsensusEngine()
        for _ in range(3):
            engine.evaluate([
                ModelResponse(model_id="a", verdict=True),
                ModelResponse(model_id="b", verdict=True),
            ])
        assert len(engine.history) == 3

    def test_summary_empty(self) -> None:
        engine = ConsensusEngine()
        s = engine.summary()
        assert s["total_evaluations"] == 0

    def test_summary_with_data(self) -> None:
        engine = ConsensusEngine()
        engine.evaluate([
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=True),
        ])
        engine.evaluate([
            ModelResponse(model_id="a", verdict=True),
            ModelResponse(model_id="b", verdict=False),
        ])
        s = engine.summary()
        assert s["total_evaluations"] == 2
        assert "mean_agreement" in s
        assert "disagreement_distribution" in s


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_full_agreement_full_confidence(self) -> None:
        engine = ConsensusEngine()
        responses = [
            ModelResponse(model_id="a", verdict=True, confidence=1.0),
            ModelResponse(model_id="b", verdict=True, confidence=1.0),
        ]
        result = engine.evaluate(responses)
        assert result.confidence == 1.0

    def test_disagreement_lowers_confidence(self) -> None:
        engine = ConsensusEngine()
        responses = [
            ModelResponse(model_id="a", verdict=True, confidence=0.9),
            ModelResponse(model_id="b", verdict=False, confidence=0.9),
        ]
        result = engine.evaluate(responses)
        assert result.confidence < 0.9

    def test_low_self_confidence(self) -> None:
        engine = ConsensusEngine()
        responses = [
            ModelResponse(model_id="a", verdict=True, confidence=0.3),
            ModelResponse(model_id="b", verdict=True, confidence=0.3),
        ]
        result = engine.evaluate(responses)
        assert result.confidence < 0.5
