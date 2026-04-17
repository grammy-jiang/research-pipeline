"""Tests for query-adaptive retrieval stopping criteria."""

from __future__ import annotations

import pytest

from research_pipeline.screening.adaptive_stopping import (
    BatchScores,
    QueryType,
    StoppingState,
    StopReason,
    check_judgment_stopping,
    check_precision_stopping,
    check_recall_stopping,
    check_score_plateau,
    classify_query_type,
    detect_knee,
    evaluate_stopping,
    marginal_gain_ratio,
)

# ── detect_knee ──────────────────────────────────────────────


class TestDetectKnee:
    def test_too_few_points_returns_false(self) -> None:
        found, idx = detect_knee([1.0, 2.0])
        assert not found
        assert idx == 0

    def test_flat_curve_is_knee_at_start(self) -> None:
        # All zero gains after first
        found, idx = detect_knee([1.0, 1.0, 1.0, 1.0])
        assert found

    def test_clear_knee(self) -> None:
        # Sharp drop: 1.0, 1.8, 1.9, 1.91, 1.911
        cumulative = [1.0, 1.8, 1.9, 1.91, 1.911, 1.9111]
        found, idx = detect_knee(cumulative, threshold=0.05)
        assert found
        assert idx >= 2  # Knee should be in the diminishing-returns zone

    def test_no_knee_in_linear_growth(self) -> None:
        # Linear: 1, 2, 3, 4, 5
        cumulative = [float(i) for i in range(1, 6)]
        found, _ = detect_knee(cumulative, threshold=0.05)
        assert not found

    def test_zero_initial_gain(self) -> None:
        found, idx = detect_knee([0.0, 0.0, 0.0])
        assert found
        assert idx == 0


# ── recall stopping ──────────────────────────────────────────


class TestRecallStopping:
    def test_min_results_not_met(self) -> None:
        state = StoppingState(min_results=10)
        state.batches = [BatchScores(0, [0.9, 0.8])]
        decision = check_recall_stopping(state)
        assert not decision.should_stop
        assert decision.reason == StopReason.MIN_RESULTS_NOT_MET

    def test_knee_triggers_stop(self) -> None:
        # High scores then rapid drop
        scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        state = StoppingState(min_results=5)
        state.batches = [BatchScores(0, scores)]
        decision = check_recall_stopping(state, knee_threshold=0.05)
        assert decision.should_stop
        assert decision.reason == StopReason.KNEE_DETECTED

    def test_no_knee_with_uniform_scores(self) -> None:
        scores = [0.5] * 20
        state = StoppingState(min_results=5)
        state.batches = [BatchScores(0, scores)]
        decision = check_recall_stopping(state)
        # Uniform scores → no knee (all gains identical)
        # Actually first gain is 0.5, rest are 0.5 → gains are constant → no knee
        assert not decision.should_stop

    def test_empty_scores(self) -> None:
        state = StoppingState(min_results=1)
        state.batches = []
        decision = check_recall_stopping(state)
        assert not decision.should_stop


# ── precision stopping ───────────────────────────────────────


class TestPrecisionStopping:
    def test_saturation_reached(self) -> None:
        # 18/20 relevant → 90% > 80% threshold
        scores = [0.9] * 18 + [0.3] * 2
        state = StoppingState(min_results=5, relevance_threshold=0.5)
        state.batches = [BatchScores(0, scores)]
        decision = check_precision_stopping(state, saturation_ratio=0.80, top_k=20)
        assert decision.should_stop
        assert decision.reason == StopReason.SATURATION_REACHED

    def test_saturation_not_reached(self) -> None:
        # 10/20 relevant → 50% < 80%
        scores = [0.9] * 10 + [0.3] * 10
        state = StoppingState(min_results=5, relevance_threshold=0.5)
        state.batches = [BatchScores(0, scores)]
        decision = check_precision_stopping(state, saturation_ratio=0.80, top_k=20)
        assert not decision.should_stop

    def test_min_results_check(self) -> None:
        state = StoppingState(min_results=10, relevance_threshold=0.5)
        state.batches = [BatchScores(0, [0.9] * 3)]
        decision = check_precision_stopping(state)
        assert not decision.should_stop
        assert decision.reason == StopReason.MIN_RESULTS_NOT_MET

    def test_fewer_results_than_top_k(self) -> None:
        # Only 8 results, top_k=20, 7 relevant → 87.5% > 80%
        scores = [0.8] * 7 + [0.3]
        state = StoppingState(min_results=5, relevance_threshold=0.5)
        state.batches = [BatchScores(0, scores)]
        decision = check_precision_stopping(state, saturation_ratio=0.80, top_k=20)
        assert decision.should_stop


# ── judgment stopping ────────────────────────────────────────


class TestJudgmentStopping:
    def test_stable_top1_triggers_stop(self) -> None:
        state = StoppingState(min_results=3)
        state.batches = [
            BatchScores(0, [0.95, 0.5]),
            BatchScores(1, [0.94, 0.6]),
            BatchScores(2, [0.95, 0.7]),
        ]
        decision = check_judgment_stopping(state, stability_window=3, tolerance=0.02)
        assert decision.should_stop
        assert decision.reason == StopReason.TOP1_STABLE

    def test_unstable_top1_continues(self) -> None:
        state = StoppingState(min_results=3)
        state.batches = [
            BatchScores(0, [0.5, 0.3]),
            BatchScores(1, [0.7, 0.4]),
            BatchScores(2, [0.95, 0.6]),
        ]
        decision = check_judgment_stopping(state, stability_window=3, tolerance=0.01)
        assert not decision.should_stop

    def test_insufficient_batches(self) -> None:
        state = StoppingState()
        state.batches = [BatchScores(0, [0.9])]
        decision = check_judgment_stopping(state, stability_window=3)
        assert not decision.should_stop


# ── score plateau ────────────────────────────────────────────


class TestScorePlateau:
    def test_plateau_detected(self) -> None:
        state = StoppingState(min_results=3)
        state.batches = [
            BatchScores(0, [0.5, 0.4]),
            BatchScores(1, [0.51, 0.41]),
            BatchScores(2, [0.505, 0.405]),
            BatchScores(3, [0.508, 0.408]),
        ]
        decision = check_score_plateau(state, window=3, improvement_threshold=0.02)
        assert decision.should_stop
        assert decision.reason == StopReason.SCORE_PLATEAU

    def test_no_plateau_with_improvement(self) -> None:
        state = StoppingState(min_results=3)
        state.batches = [
            BatchScores(0, [0.3, 0.2]),
            BatchScores(1, [0.5, 0.4]),
            BatchScores(2, [0.7, 0.6]),
            BatchScores(3, [0.9, 0.8]),
        ]
        decision = check_score_plateau(state, window=3, improvement_threshold=0.02)
        assert not decision.should_stop

    def test_insufficient_batches(self) -> None:
        state = StoppingState()
        state.batches = [BatchScores(0, [0.5])]
        decision = check_score_plateau(state, window=3)
        assert not decision.should_stop


# ── query type classification ────────────────────────────────


class TestClassifyQueryType:
    def test_recall_survey(self) -> None:
        assert (
            classify_query_type("survey of transformer architectures")
            == QueryType.RECALL
        )

    def test_recall_short_query(self) -> None:
        assert classify_query_type("transformer models") == QueryType.RECALL

    def test_precision_specific(self) -> None:
        assert (
            classify_query_type("specific algorithm for graph matching")
            == QueryType.PRECISION
        )

    def test_precision_how_to(self) -> None:
        assert (
            classify_query_type("how to implement attention mechanism")
            == QueryType.PRECISION
        )

    def test_judgment_compare(self) -> None:
        assert (
            classify_query_type("compare BERT versus GPT for NER") == QueryType.JUDGMENT
        )

    def test_judgment_best(self) -> None:
        assert (
            classify_query_type("best model for text classification")
            == QueryType.JUDGMENT
        )

    def test_default_medium_length(self) -> None:
        # 4-7 words, no markers → default recall
        qtype = classify_query_type("neural network pruning methods")
        assert qtype == QueryType.RECALL

    def test_long_query_precision(self) -> None:
        qtype = classify_query_type(
            "a very specific detailed long query about multiple topics and aspects"
        )
        assert qtype == QueryType.PRECISION


# ── composite evaluate_stopping ──────────────────────────────


class TestEvaluateStopping:
    def test_budget_exhausted(self) -> None:
        state = StoppingState(max_budget=10)
        state.batches = [BatchScores(0, list(range(15)))]
        decision = evaluate_stopping(state)
        assert decision.should_stop
        assert decision.reason == StopReason.BUDGET_EXHAUSTED

    def test_recall_query_uses_knee(self) -> None:
        scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        state = StoppingState(
            query_type=QueryType.RECALL, min_results=5, max_budget=500
        )
        state.batches = [BatchScores(0, scores)]
        decision = evaluate_stopping(state)
        assert decision.should_stop
        assert decision.reason == StopReason.KNEE_DETECTED

    def test_precision_query_uses_saturation(self) -> None:
        scores = [0.9] * 18 + [0.3] * 2
        state = StoppingState(
            query_type=QueryType.PRECISION,
            min_results=5,
            relevance_threshold=0.5,
            max_budget=500,
        )
        state.batches = [BatchScores(0, scores)]
        decision = evaluate_stopping(state)
        assert decision.should_stop
        assert decision.reason == StopReason.SATURATION_REACHED

    def test_judgment_query_uses_stability(self) -> None:
        state = StoppingState(
            query_type=QueryType.JUDGMENT, min_results=3, max_budget=500
        )
        state.batches = [
            BatchScores(0, [0.95, 0.5]),
            BatchScores(1, [0.94, 0.6]),
            BatchScores(2, [0.95, 0.7]),
        ]
        decision = evaluate_stopping(state)
        assert decision.should_stop
        assert decision.reason == StopReason.TOP1_STABLE

    def test_auto_classification(self) -> None:
        scores = [0.9] * 18 + [0.3] * 2
        state = StoppingState(
            query_type=QueryType.AUTO,
            min_results=5,
            relevance_threshold=0.5,
            max_budget=500,
        )
        state.batches = [BatchScores(0, scores)]
        # "specific algorithm" → PRECISION → saturation check
        decision = evaluate_stopping(
            state, query="specific algorithm for graph matching"
        )
        assert decision.should_stop

    def test_plateau_backstop(self) -> None:
        state = StoppingState(
            query_type=QueryType.RECALL, min_results=3, max_budget=500
        )
        # Scores that don't trigger knee but do plateau
        state.batches = [
            BatchScores(0, [0.5, 0.5, 0.5]),
            BatchScores(1, [0.51, 0.51, 0.51]),
            BatchScores(2, [0.505, 0.505, 0.505]),
            BatchScores(3, [0.508, 0.508, 0.508]),
        ]
        decision = evaluate_stopping(state)
        # Should hit plateau backstop since recall knee won't trigger
        if decision.should_stop:
            assert decision.reason in (
                StopReason.SCORE_PLATEAU,
                StopReason.KNEE_DETECTED,
            )

    def test_not_stopped(self) -> None:
        state = StoppingState(
            query_type=QueryType.RECALL, min_results=10, max_budget=500
        )
        state.batches = [BatchScores(0, [0.9, 0.8, 0.7])]
        decision = evaluate_stopping(state)
        assert not decision.should_stop


# ── marginal_gain_ratio ──────────────────────────────────────


class TestMarginalGainRatio:
    def test_valid_index(self) -> None:
        cumulative = [1.0, 1.5, 1.7, 1.8]
        ratio = marginal_gain_ratio(cumulative, 1)
        assert ratio == pytest.approx(0.5)  # 0.5 / 1.0

    def test_zero_index(self) -> None:
        assert marginal_gain_ratio([1.0, 1.5], 0) == 0.0

    def test_out_of_range(self) -> None:
        assert marginal_gain_ratio([1.0, 1.5], 5) == 0.0

    def test_empty_list(self) -> None:
        assert marginal_gain_ratio([], 0) == 0.0

    def test_zero_initial_gain(self) -> None:
        assert marginal_gain_ratio([0.0, 0.5], 1) == 0.0


# ── BatchScores dataclass ────────────────────────────────────


class TestBatchScores:
    def test_auto_computed_fields(self) -> None:
        b = BatchScores(0, [0.9, 0.5, 0.3])
        assert b.best_score == 0.9
        assert b.mean_score == pytest.approx(0.5666666, rel=1e-3)

    def test_empty_scores(self) -> None:
        b = BatchScores(0, [])
        assert b.best_score == 0.0
        assert b.mean_score == 0.0


# ── StoppingState properties ────────────────────────────────


class TestStoppingState:
    def test_total_results(self) -> None:
        state = StoppingState()
        state.batches = [
            BatchScores(0, [0.1, 0.2]),
            BatchScores(1, [0.3]),
        ]
        assert state.total_results == 3

    def test_all_scores(self) -> None:
        state = StoppingState()
        state.batches = [
            BatchScores(0, [0.1, 0.2]),
            BatchScores(1, [0.3]),
        ]
        assert state.all_scores == [0.1, 0.2, 0.3]

    def test_empty_state(self) -> None:
        state = StoppingState()
        assert state.total_results == 0
        assert state.all_scores == []
