"""Tests for screening.typed_stopping — query-typed retrieval stopping."""

from __future__ import annotations

from research_pipeline.screening.typed_stopping import (
    DEFAULT_PROFILES,
    EXPLORATORY_PROFILE,
    JUDGMENT_PROFILE,
    PRECISION_PROFILE,
    RECALL_PROFILE,
    VERIFICATION_PROFILE,
    ExtendedQueryType,
    StoppingProfile,
    TypedStoppingEvaluator,
    TypedStoppingResult,
    classify_query_type,
    estimate_cost,
    get_profile,
)

# ---------------------------------------------------------------------------
# ExtendedQueryType
# ---------------------------------------------------------------------------


class TestExtendedQueryType:
    def test_values(self) -> None:
        assert ExtendedQueryType.RECALL.value == "recall"
        assert ExtendedQueryType.VERIFICATION.value == "verification"

    def test_all_types(self) -> None:
        assert len(ExtendedQueryType) == 5


# ---------------------------------------------------------------------------
# StoppingProfile
# ---------------------------------------------------------------------------


class TestStoppingProfile:
    def test_recall_defaults(self) -> None:
        p = RECALL_PROFILE
        assert p.knee_threshold == 0.02
        assert p.cost_weight == 0.3
        assert p.max_batches == 30

    def test_precision_defaults(self) -> None:
        p = PRECISION_PROFILE
        assert p.knee_threshold == 0.15
        assert p.cost_weight == 2.0
        assert p.max_batches == 10

    def test_verification_aggressive(self) -> None:
        p = VERIFICATION_PROFILE
        assert p.cost_weight > RECALL_PROFILE.cost_weight
        assert p.max_batches < RECALL_PROFILE.max_batches

    def test_to_dict(self) -> None:
        d = RECALL_PROFILE.to_dict()
        assert d["query_type"] == "recall"
        assert "knee_threshold" in d
        assert "description" in d


# ---------------------------------------------------------------------------
# classify_query_type
# ---------------------------------------------------------------------------


class TestClassifyQueryType:
    def test_recall_keywords(self) -> None:
        assert classify_query_type("systematic review of NLP methods") == ExtendedQueryType.RECALL

    def test_precision_keywords(self) -> None:
        assert classify_query_type("what is the specific algorithm for BERT") == ExtendedQueryType.PRECISION

    def test_judgment_keywords(self) -> None:
        assert classify_query_type("compare transformer versus RNN") == ExtendedQueryType.JUDGMENT

    def test_exploratory_keywords(self) -> None:
        assert classify_query_type("explore emerging trends in AI") == ExtendedQueryType.EXPLORATORY

    def test_verification_keywords(self) -> None:
        assert classify_query_type("verify the claim about accuracy") == ExtendedQueryType.VERIFICATION

    def test_empty_query(self) -> None:
        assert classify_query_type("") == ExtendedQueryType.RECALL

    def test_ambiguous_defaults_recall(self) -> None:
        assert classify_query_type("machine learning") == ExtendedQueryType.RECALL

    def test_mixed_keywords(self) -> None:
        # verification wins over others (priority order)
        result = classify_query_type("verify this comprehensive review claim")
        assert result in (ExtendedQueryType.VERIFICATION, ExtendedQueryType.RECALL)


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_default_profiles(self) -> None:
        p = get_profile(ExtendedQueryType.PRECISION)
        assert p.query_type == ExtendedQueryType.PRECISION
        assert p.cost_weight == 2.0

    def test_custom_profiles(self) -> None:
        custom = {
            ExtendedQueryType.PRECISION: StoppingProfile(
                query_type=ExtendedQueryType.PRECISION,
                cost_weight=5.0,
            )
        }
        p = get_profile(ExtendedQueryType.PRECISION, custom)
        assert p.cost_weight == 5.0

    def test_fallback_to_recall(self) -> None:
        custom = {ExtendedQueryType.PRECISION: PRECISION_PROFILE}
        p = get_profile(ExtendedQueryType.EXPLORATORY, custom)
        assert p.query_type == ExtendedQueryType.RECALL


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_recall_baseline(self) -> None:
        est = estimate_cost("comprehensive survey of all methods")
        assert est.query_type == ExtendedQueryType.RECALL
        assert est.estimated_savings >= 0.0

    def test_precision_saves_cost(self) -> None:
        est = estimate_cost("what is the specific technique for X")
        assert est.query_type == ExtendedQueryType.PRECISION
        assert est.estimated_savings > 0.0

    def test_to_dict(self) -> None:
        est = estimate_cost("test query")
        d = est.to_dict()
        assert "query_type" in d
        assert "estimated_savings" in d

    def test_verification_most_savings(self) -> None:
        est = estimate_cost("verify this claim is correct")
        assert est.estimated_savings > 0.3


# ---------------------------------------------------------------------------
# TypedStoppingEvaluator
# ---------------------------------------------------------------------------


class TestTypedStoppingEvaluator:
    def test_below_min_batches(self) -> None:
        ev = TypedStoppingEvaluator(query="compare models")
        ev.add_batch([0.9, 0.8, 0.7])
        result = ev.evaluate()
        assert not result.should_stop
        assert result.reason == "below_min_batches"

    def test_budget_exhausted(self) -> None:
        ev = TypedStoppingEvaluator(
            query_type=ExtendedQueryType.VERIFICATION,
        )
        for _ in range(10):
            ev.add_batch([0.5, 0.4])
        result = ev.evaluate()
        assert result.should_stop
        assert result.reason == "budget_exhausted"

    def test_saturation_stop(self) -> None:
        ev = TypedStoppingEvaluator(
            query_type=ExtendedQueryType.PRECISION,
        )
        for _ in range(3):
            ev.add_batch([0.9, 0.95, 0.85, 0.92])
        result = ev.evaluate()
        assert result.should_stop
        assert "saturation" in result.reason

    def test_knee_detection(self) -> None:
        ev = TypedStoppingEvaluator(
            query_type=ExtendedQueryType.RECALL,
        )
        # Scores mostly below quality floor (0.3) so saturation won't trigger
        ev.add_batch([0.5, 0.2, 0.1])
        ev.add_batch([0.3, 0.15, 0.05])
        ev.add_batch([0.15, 0.1, 0.05])
        ev.add_batch([0.14, 0.1, 0.06])  # gain ~0.003 < knee 0.02
        result = ev.evaluate()
        assert result.should_stop
        assert "knee" in result.reason

    def test_top1_stable(self) -> None:
        ev = TypedStoppingEvaluator(
            query_type=ExtendedQueryType.JUDGMENT,
        )
        # Top-1 is always 0.85 (stable), but means vary enough to avoid knee
        # and most scores below quality_floor=0.5 to avoid saturation
        ev.add_batch([0.85, 0.3, 0.1])
        ev.add_batch([0.85, 0.1, 0.1])
        ev.add_batch([0.85, 0.3, 0.2])  # gain=|0.45-0.35|=0.10 > 0.08
        result = ev.evaluate()
        assert result.should_stop
        assert "top1_stable" in result.reason

    def test_continue_when_improving(self) -> None:
        ev = TypedStoppingEvaluator(
            query_type=ExtendedQueryType.RECALL,
        )
        # Increasing means, mostly below quality floor → no saturation, no knee
        ev.add_batch([0.1, 0.05, 0.02])
        ev.add_batch([0.2, 0.15, 0.1])
        ev.add_batch([0.4, 0.25, 0.15])
        result = ev.evaluate()
        assert not result.should_stop

    def test_query_type_property(self) -> None:
        ev = TypedStoppingEvaluator(query="survey of methods")
        assert ev.query_type == ExtendedQueryType.RECALL

    def test_explicit_query_type(self) -> None:
        ev = TypedStoppingEvaluator(
            query="anything", query_type=ExtendedQueryType.VERIFICATION,
        )
        assert ev.query_type == ExtendedQueryType.VERIFICATION

    def test_batches_processed(self) -> None:
        ev = TypedStoppingEvaluator()
        assert ev.batches_processed == 0
        ev.add_batch([0.5])
        assert ev.batches_processed == 1

    def test_summary(self) -> None:
        ev = TypedStoppingEvaluator(query="test")
        ev.add_batch([0.8, 0.7])
        s = ev.summary()
        assert s["batches_processed"] == 1
        assert s["total_scores"] == 2
        assert "profile" in s

    def test_cost_tracking(self) -> None:
        ev = TypedStoppingEvaluator(
            query_type=ExtendedQueryType.PRECISION,
        )
        ev.add_batch([0.9, 0.8])
        ev.add_batch([0.7, 0.6])
        result = ev.evaluate()
        assert result.cost_so_far == 2 * PRECISION_PROFILE.cost_weight


# ---------------------------------------------------------------------------
# TypedStoppingResult
# ---------------------------------------------------------------------------


class TestTypedStoppingResult:
    def test_to_dict(self) -> None:
        r = TypedStoppingResult(
            should_stop=True,
            query_type=ExtendedQueryType.PRECISION,
            profile=PRECISION_PROFILE,
            reason="saturation",
            batches_processed=3,
            cost_so_far=6.0,
        )
        d = r.to_dict()
        assert d["should_stop"] is True
        assert d["query_type"] == "precision"


# ---------------------------------------------------------------------------
# DEFAULT_PROFILES completeness
# ---------------------------------------------------------------------------


class TestDefaultProfiles:
    def test_all_types_covered(self) -> None:
        for qt in ExtendedQueryType:
            assert qt in DEFAULT_PROFILES

    def test_cost_ordering(self) -> None:
        # Verification should be most aggressive, recall least
        assert VERIFICATION_PROFILE.cost_weight > PRECISION_PROFILE.cost_weight
        assert PRECISION_PROFILE.cost_weight > JUDGMENT_PROFILE.cost_weight
        assert JUDGMENT_PROFILE.cost_weight > RECALL_PROFILE.cost_weight
        assert RECALL_PROFILE.cost_weight > EXPLORATORY_PROFILE.cost_weight

    def test_max_batches_ordering(self) -> None:
        # Exploratory/recall should allow more batches than precision/verification
        assert RECALL_PROFILE.max_batches > PRECISION_PROFILE.max_batches
        assert EXPLORATORY_PROFILE.max_batches > JUDGMENT_PROFILE.max_batches
