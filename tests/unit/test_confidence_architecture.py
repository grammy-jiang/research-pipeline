"""Tests for the 4-layer confidence architecture."""

from __future__ import annotations

from research_pipeline.confidence.architecture import (
    ArchitectureConfig,
    CalibrationMethod,
    CalibrationReport,
    ConfidenceLayer,
    GranularityDecision,
    L1Result,
    LayeredConfidenceResult,
    PlattParams,
    _run_l1,
    _run_l2,
    _run_l3,
    _run_l4,
    batch_calibration_report,
    compute_auroc,
    compute_brier,
    compute_ece,
    damped_fusion,
    dinco_calibrate,
    estimate_claim_complexity,
    evaluate_calibration,
    fit_platt_scaling,
    nli_triage,
    score_batch_layered,
    score_claim_layered,
)
from research_pipeline.models.claim import (
    AtomicClaim,
    ClaimEvidence,
    EvidenceClass,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_claim(
    claim_id: str = "CL-001",
    paper_id: str = "2401.00001",
    statement: str = "Transformers outperform RNNs on long-range tasks.",
    evidence_class: EvidenceClass = EvidenceClass.SUPPORTED,
    n_evidence: int = 3,
    relevance: float = 0.8,
) -> AtomicClaim:
    evidence = [
        ClaimEvidence(
            chunk_id=f"chunk-{i}",
            relevance_score=relevance,
            quote=f"Evidence quote {i}",
        )
        for i in range(n_evidence)
    ]
    return AtomicClaim(
        claim_id=claim_id,
        paper_id=paper_id,
        source_type="finding",
        statement=statement,
        evidence_class=evidence_class,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Calibration Metrics Tests
# ---------------------------------------------------------------------------


class TestComputeECE:
    def test_perfect_calibration(self) -> None:
        preds = [0.0, 0.5, 1.0]
        actuals = [0.0, 0.5, 1.0]
        ece = compute_ece(preds, actuals, n_bins=10)
        assert ece < 0.1

    def test_worst_calibration(self) -> None:
        preds = [1.0, 1.0, 1.0]
        actuals = [0.0, 0.0, 0.0]
        ece = compute_ece(preds, actuals, n_bins=10)
        assert ece > 0.9

    def test_empty_inputs(self) -> None:
        assert compute_ece([], [], n_bins=10) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert compute_ece([0.5], [0.0, 1.0]) == 0.0


class TestComputeBrier:
    def test_perfect_prediction(self) -> None:
        preds = [1.0, 0.0, 1.0]
        actuals = [1.0, 0.0, 1.0]
        assert compute_brier(preds, actuals) == 0.0

    def test_worst_prediction(self) -> None:
        preds = [1.0, 0.0]
        actuals = [0.0, 1.0]
        assert compute_brier(preds, actuals) == 1.0

    def test_empty_inputs(self) -> None:
        assert compute_brier([], []) == 0.0

    def test_moderate_prediction(self) -> None:
        preds = [0.7, 0.3]
        actuals = [1.0, 0.0]
        brier = compute_brier(preds, actuals)
        assert 0 < brier < 0.15


class TestComputeAUROC:
    def test_perfect_separation(self) -> None:
        preds = [0.9, 0.8, 0.2, 0.1]
        actuals = [1.0, 1.0, 0.0, 0.0]
        auroc = compute_auroc(preds, actuals)
        assert auroc == 1.0

    def test_random_separation(self) -> None:
        preds = [0.5, 0.5, 0.5, 0.5]
        actuals = [1.0, 0.0, 1.0, 0.0]
        auroc = compute_auroc(preds, actuals)
        assert auroc == 0.5

    def test_empty_inputs(self) -> None:
        assert compute_auroc([], []) == 0.5

    def test_all_same_class(self) -> None:
        preds = [0.8, 0.9]
        actuals = [1.0, 1.0]
        assert compute_auroc(preds, actuals) == 0.5


class TestEvaluateCalibration:
    def test_returns_report(self) -> None:
        preds = [0.8, 0.2, 0.9, 0.1]
        actuals = [1.0, 0.0, 1.0, 0.0]
        report = evaluate_calibration(preds, actuals)
        assert isinstance(report, CalibrationReport)
        assert report.n_samples == 4
        assert 0 <= report.ece <= 1
        assert 0 <= report.brier <= 1
        assert 0 <= report.auroc <= 1


# ---------------------------------------------------------------------------
# DINCO Calibration Tests
# ---------------------------------------------------------------------------


class TestDINCOCalibrate:
    def test_default_distractor(self) -> None:
        score = dinco_calibrate(0.8)
        assert 0 < score <= 1.0

    def test_with_distractor_scores(self) -> None:
        score = dinco_calibrate(0.8, distractor_scores=[0.3, 0.4, 0.5])
        assert 0 < score <= 1.0

    def test_low_score_near_distractor(self) -> None:
        score = dinco_calibrate(0.4, distractor_scores=[0.4, 0.4])
        assert abs(score - 0.5) < 0.01

    def test_high_score_above_distractor(self) -> None:
        high = dinco_calibrate(0.9, distractor_scores=[0.2])
        low = dinco_calibrate(0.3, distractor_scores=[0.2])
        assert high > low

    def test_empty_distractor_scores(self) -> None:
        score = dinco_calibrate(0.7, distractor_scores=[])
        assert score == 0.7

    def test_zero_temperature(self) -> None:
        score = dinco_calibrate(0.5, temperature=0.0)
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# Platt Scaling Tests
# ---------------------------------------------------------------------------


class TestPlattParams:
    def test_identity_transform(self) -> None:
        params = PlattParams(a=1.0, b=0.0)
        result = params.transform(0.0)
        assert result == 0.5

    def test_positive_slope(self) -> None:
        params = PlattParams(a=5.0, b=0.0)
        high = params.transform(1.0)
        low = params.transform(-1.0)
        assert high > low

    def test_overflow_protection(self) -> None:
        params = PlattParams(a=100.0, b=100.0)
        result = params.transform(100.0)
        assert result <= 1.0


class TestFitPlattScaling:
    def test_fit_basic(self) -> None:
        preds = [0.1, 0.3, 0.7, 0.9]
        actuals = [0.0, 0.0, 1.0, 1.0]
        params = fit_platt_scaling(preds, actuals, iterations=200)
        assert isinstance(params, PlattParams)
        assert params.a > 0  # positive correlation

    def test_fit_empty(self) -> None:
        params = fit_platt_scaling([], [])
        assert params.a == 1.0
        assert params.b == 0.0

    def test_fit_mismatched(self) -> None:
        params = fit_platt_scaling([0.5], [0.0, 1.0])
        assert params.a == 1.0


# ---------------------------------------------------------------------------
# Damped Fusion Tests
# ---------------------------------------------------------------------------


class TestDampedFusion:
    def test_uniform_signals(self) -> None:
        # 0.5^0.8 ≈ 0.574 (power damping raises sub-1.0 scores)
        result = damped_fusion([0.5, 0.5, 0.5])
        assert abs(result - 0.574) < 0.01

    def test_empty_signals(self) -> None:
        assert damped_fusion([]) == 0.0

    def test_single_signal(self) -> None:
        result = damped_fusion([0.8])
        assert result > 0

    def test_weighted(self) -> None:
        result = damped_fusion([0.9, 0.1], [0.9, 0.1])
        assert result > 0.5

    def test_no_damping(self) -> None:
        result = damped_fusion([0.6, 0.4], damping=1.0)
        assert abs(result - 0.5) < 0.01

    def test_weight_mismatch_uses_uniform(self) -> None:
        result = damped_fusion([0.5, 0.5], [0.3])
        assert result > 0


# ---------------------------------------------------------------------------
# Claim Complexity Tests
# ---------------------------------------------------------------------------


class TestEstimateClaimComplexity:
    def test_simple_claim(self) -> None:
        claim = _make_claim(statement="Simple fact.", n_evidence=1)
        complexity = estimate_claim_complexity(claim)
        assert complexity < 0.3

    def test_complex_claim(self) -> None:
        claim = _make_claim(
            statement=" ".join(["word"] * 60),
            n_evidence=5,
            evidence_class=EvidenceClass.CONFLICTING,
        )
        complexity = estimate_claim_complexity(claim)
        assert complexity > 0.5

    def test_no_evidence(self) -> None:
        claim = _make_claim(n_evidence=0)
        complexity = estimate_claim_complexity(claim)
        assert complexity >= 0


# ---------------------------------------------------------------------------
# NLI Triage Tests
# ---------------------------------------------------------------------------


class TestNLITriage:
    def test_high_confidence_skips(self) -> None:
        claim = _make_claim()
        decision = nli_triage(claim, l1_score=0.95)
        assert decision == GranularityDecision.SKIP

    def test_low_confidence_skips(self) -> None:
        claim = _make_claim(evidence_class=EvidenceClass.UNSUPPORTED, n_evidence=0)
        decision = nli_triage(claim, l1_score=0.05)
        assert decision == GranularityDecision.SKIP

    def test_moderate_simple_keeps(self) -> None:
        claim = _make_claim(statement="Simple fact.", n_evidence=1)
        decision = nli_triage(claim, l1_score=0.5)
        assert decision == GranularityDecision.KEEP

    def test_moderate_complex_decomposes(self) -> None:
        claim = _make_claim(
            statement=" ".join(["word"] * 60),
            n_evidence=5,
            evidence_class=EvidenceClass.CONFLICTING,
        )
        decision = nli_triage(claim, l1_score=0.5)
        assert decision == GranularityDecision.DECOMPOSE


# ---------------------------------------------------------------------------
# Layer Tests
# ---------------------------------------------------------------------------


class TestL1:
    def test_supported_claim_high_score(self) -> None:
        claim = _make_claim()
        config = ArchitectureConfig()
        l1 = _run_l1(claim, config)
        assert l1.fast_score > 0.3
        assert l1.evidence_signal > 0

    def test_unsupported_claim_low_score(self) -> None:
        claim = _make_claim(evidence_class=EvidenceClass.UNSUPPORTED, n_evidence=0)
        config = ArchitectureConfig()
        l1 = _run_l1(claim, config)
        assert l1.fast_score < 0.5

    def test_skip_very_high(self) -> None:
        claim = _make_claim(relevance=1.0, n_evidence=5)
        config = ArchitectureConfig(l1_skip_high=0.7)
        l1 = _run_l1(claim, config)
        if l1.fast_score >= 0.7:
            assert l1.skip_deeper


class TestL2:
    def test_skip_propagates(self) -> None:
        claim = _make_claim()
        l1 = L1Result(
            fast_score=0.95,
            hedging_signal=0.8,
            evidence_signal=0.9,
            citation_density=0.6,
            retrieval_quality=0.8,
            skip_deeper=True,
        )
        config = ArchitectureConfig()
        l2 = _run_l2(claim, l1, config)
        assert l2.decision == GranularityDecision.SKIP
        assert l2.adjusted_score == 0.95

    def test_keeps_moderate(self) -> None:
        claim = _make_claim(statement="Simple fact.", n_evidence=1)
        l1 = L1Result(
            fast_score=0.5,
            hedging_signal=0.5,
            evidence_signal=0.5,
            citation_density=0.2,
            retrieval_quality=0.5,
            skip_deeper=False,
        )
        config = ArchitectureConfig()
        l2 = _run_l2(claim, l1, config)
        assert l2.decision in (GranularityDecision.KEEP, GranularityDecision.DECOMPOSE)


class TestL3:
    def test_dinco_default(self) -> None:
        config = ArchitectureConfig()
        l3 = _run_l3(0.6, config)
        assert l3.method == CalibrationMethod.DINCO
        assert l3.calibrated_score > 0

    def test_platt_when_available(self) -> None:
        config = ArchitectureConfig(platt_params=PlattParams(a=2.0, b=-0.5))
        l3 = _run_l3(0.6, config)
        assert l3.method == CalibrationMethod.PLATT

    def test_calibration_delta(self) -> None:
        config = ArchitectureConfig()
        l3 = _run_l3(0.5, config)
        assert abs(l3.calibration_delta) < 1.0
        expected_delta = round(l3.calibrated_score - l3.raw_score, 4)
        assert l3.calibration_delta == expected_delta


class TestL4:
    def test_not_triggered_above_threshold(self) -> None:
        claim = _make_claim()
        config = ArchitectureConfig(l4_threshold=0.5)
        l4 = _run_l4(claim, 0.7, config)
        assert not l4.triggered

    def test_triggered_below_threshold(self) -> None:
        claim = _make_claim()
        config = ArchitectureConfig(l4_threshold=0.5)
        l4 = _run_l4(claim, 0.3, config)
        assert l4.triggered

    def test_no_llm_still_triggers(self) -> None:
        claim = _make_claim()
        config = ArchitectureConfig(l4_threshold=0.5)
        l4 = _run_l4(claim, 0.2, config, llm_provider=None)
        assert l4.triggered
        assert l4.samples_used == 0


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------


class TestScoreClaimLayered:
    def test_basic_scoring(self) -> None:
        claim = _make_claim()
        result = score_claim_layered(claim)
        assert isinstance(result, LayeredConfidenceResult)
        assert 0 <= result.final_score <= 1
        assert len(result.layers_executed) >= 3

    def test_high_confidence_skips_l4(self) -> None:
        claim = _make_claim(relevance=1.0, n_evidence=5)
        result = score_claim_layered(claim)
        assert ConfidenceLayer.L4_VERIFICATION.value not in result.layers_executed

    def test_to_dict(self) -> None:
        claim = _make_claim()
        result = score_claim_layered(claim)
        d = result.to_dict()
        assert "claim_id" in d
        assert "final_score" in d
        assert "l1" in d
        assert "l2" in d
        assert "l3" in d
        assert "l4" in d

    def test_custom_config(self) -> None:
        claim = _make_claim()
        config = ArchitectureConfig(
            l1_skip_high=0.99,
            l1_skip_low=0.01,
            l4_threshold=0.99,
        )
        result = score_claim_layered(claim, config=config)
        assert result.final_score > 0


class TestScoreBatchLayered:
    def test_batch_scoring(self) -> None:
        claims = [
            _make_claim(claim_id="CL-001"),
            _make_claim(
                claim_id="CL-002",
                evidence_class=EvidenceClass.UNSUPPORTED,
                n_evidence=0,
            ),
            _make_claim(
                claim_id="CL-003",
                evidence_class=EvidenceClass.CONFLICTING,
            ),
        ]
        results = score_batch_layered(claims)
        assert len(results) == 3
        assert all(isinstance(r, LayeredConfidenceResult) for r in results)

    def test_empty_batch(self) -> None:
        results = score_batch_layered([])
        assert results == []


class TestBatchCalibrationReport:
    def test_report_generation(self) -> None:
        claims = [_make_claim(claim_id=f"CL-{i:03d}") for i in range(5)]
        results = score_batch_layered(claims)
        report = batch_calibration_report(results)
        assert isinstance(report, CalibrationReport)
        assert report.n_samples == 5
        d = report.to_dict()
        assert "ece" in d
        assert "brier" in d
        assert "auroc" in d

    def test_with_ground_truth(self) -> None:
        claims = [_make_claim(claim_id=f"CL-{i:03d}") for i in range(4)]
        results = score_batch_layered(claims)
        report = batch_calibration_report(results, ground_truth=[1.0, 0.0, 1.0, 0.0])
        assert report.n_samples == 4


# ---------------------------------------------------------------------------
# Enum and Dataclass Tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_confidence_layer_values(self) -> None:
        assert ConfidenceLayer.L1_FAST_SIGNAL.value == "L1_fast_signal"
        assert ConfidenceLayer.L2_DECOMPOSITION.value == "L2_decomposition"
        assert ConfidenceLayer.L3_CALIBRATION.value == "L3_calibration"
        assert ConfidenceLayer.L4_VERIFICATION.value == "L4_verification"

    def test_granularity_decision_values(self) -> None:
        assert GranularityDecision.SKIP.value == "skip"
        assert GranularityDecision.KEEP.value == "keep"
        assert GranularityDecision.DECOMPOSE.value == "decompose"

    def test_calibration_method_values(self) -> None:
        assert CalibrationMethod.IDENTITY.value == "identity"
        assert CalibrationMethod.DINCO.value == "dinco"
        assert CalibrationMethod.PLATT.value == "platt"
