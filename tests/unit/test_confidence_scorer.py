"""Tests for confidence scoring module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research_pipeline.confidence.scorer import (
    ConfidenceSignals,
    _compute_consistency,
    compute_citation_density,
    compute_evidence_signal,
    compute_hedging_signal,
    compute_retrieval_quality,
    score_claim,
    score_decomposition,
)
from research_pipeline.models.claim import (
    AtomicClaim,
    ClaimDecomposition,
    ClaimEvidence,
    EvidenceClass,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evidence(relevance: float = 0.8, chunk_id: str = "c1") -> ClaimEvidence:
    return ClaimEvidence(chunk_id=chunk_id, relevance_score=relevance, quote="quote")


def _make_claim(
    claim_id: str = "CL-001",
    paper_id: str = "2401.00001",
    statement: str = "The model demonstrates improved accuracy.",
    evidence_class: EvidenceClass = EvidenceClass.SUPPORTED,
    evidence: list[ClaimEvidence] | None = None,
    confidence_score: float = 0.0,
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        paper_id=paper_id,
        source_type="finding",
        statement=statement,
        evidence_class=evidence_class,
        evidence=evidence or [],
        confidence_score=confidence_score,
    )


def _make_decomposition(
    claims: list[AtomicClaim] | None = None,
) -> ClaimDecomposition:
    if claims is None:
        claims = [_make_claim()]
    return ClaimDecomposition(
        paper_id="2401.00001",
        title="Test Paper",
        claims=claims,
        total_claims=len(claims),
        evidence_summary={"supported": len(claims)},
    )


# ---------------------------------------------------------------------------
# compute_evidence_signal
# ---------------------------------------------------------------------------


class TestComputeEvidenceSignal:
    def test_supported(self) -> None:
        assert compute_evidence_signal(EvidenceClass.SUPPORTED) == 0.9

    def test_partial(self) -> None:
        assert compute_evidence_signal(EvidenceClass.PARTIAL) == 0.6

    def test_conflicting(self) -> None:
        assert compute_evidence_signal(EvidenceClass.CONFLICTING) == 0.3

    def test_inconclusive(self) -> None:
        assert compute_evidence_signal(EvidenceClass.INCONCLUSIVE) == 0.2

    def test_unsupported(self) -> None:
        assert compute_evidence_signal(EvidenceClass.UNSUPPORTED) == 0.05


# ---------------------------------------------------------------------------
# compute_hedging_signal
# ---------------------------------------------------------------------------


class TestComputeHedgingSignal:
    def test_empty_text(self) -> None:
        assert compute_hedging_signal("") == 0.5

    def test_strong_hedge_words(self) -> None:
        text = "This might possibly suggest a trend."
        score = compute_hedging_signal(text)
        assert score < 0.3

    def test_certainty_words(self) -> None:
        text = "This clearly demonstrates the result and confirms the hypothesis."
        score = compute_hedging_signal(text)
        assert score > 0.7

    def test_mixed_hedging_and_certainty(self) -> None:
        text = (
            "The result clearly demonstrates improvement, "
            "but might suggest limitations."
        )
        score = compute_hedging_signal(text)
        assert 0.2 < score < 0.8

    def test_no_hedge_or_certainty(self) -> None:
        text = "The model was trained on ImageNet for 100 epochs."
        assert compute_hedging_signal(text) == 0.5

    def test_weak_hedge_words(self) -> None:
        text = "Generally, the method typically works in some cases."
        score = compute_hedging_signal(text)
        assert score < 0.3

    def test_all_strong_hedges(self) -> None:
        text = "It might could possibly perhaps be speculative."
        score = compute_hedging_signal(text)
        assert score == 0.0

    def test_all_certainty(self) -> None:
        text = "Clearly proven and conclusively established."
        score = compute_hedging_signal(text)
        assert score == 1.0


# ---------------------------------------------------------------------------
# compute_citation_density
# ---------------------------------------------------------------------------


class TestComputeCitationDensity:
    def test_no_evidence(self) -> None:
        claim = _make_claim(evidence=[])
        assert compute_citation_density(claim) == 0.0

    def test_one_evidence(self) -> None:
        claim = _make_claim(evidence=[_make_evidence()])
        assert compute_citation_density(claim) == pytest.approx(0.2, abs=0.01)

    def test_max_evidence(self) -> None:
        claim = _make_claim(
            evidence=[_make_evidence(chunk_id=f"c{i}") for i in range(5)]
        )
        assert compute_citation_density(claim) == 1.0

    def test_over_max_clamped(self) -> None:
        claim = _make_claim(
            evidence=[_make_evidence(chunk_id=f"c{i}") for i in range(10)]
        )
        assert compute_citation_density(claim) == 1.0

    def test_custom_max(self) -> None:
        claim = _make_claim(
            evidence=[_make_evidence(chunk_id=f"c{i}") for i in range(2)]
        )
        assert compute_citation_density(claim, max_evidence=4) == pytest.approx(
            0.5, abs=0.01
        )


# ---------------------------------------------------------------------------
# compute_retrieval_quality
# ---------------------------------------------------------------------------


class TestComputeRetrievalQuality:
    def test_no_evidence(self) -> None:
        claim = _make_claim(evidence=[])
        assert compute_retrieval_quality(claim) == 0.0

    def test_high_score(self) -> None:
        claim = _make_claim(
            evidence=[_make_evidence(0.3), _make_evidence(0.95, chunk_id="c2")]
        )
        assert compute_retrieval_quality(claim) == pytest.approx(0.95, abs=0.01)

    def test_all_low_scores(self) -> None:
        claim = _make_claim(
            evidence=[
                _make_evidence(0.1),
                _make_evidence(0.15, chunk_id="c2"),
            ]
        )
        assert compute_retrieval_quality(claim) == pytest.approx(0.15, abs=0.01)

    def test_clamped_to_one(self) -> None:
        claim = _make_claim(evidence=[_make_evidence(1.5)])
        assert compute_retrieval_quality(claim) == 1.0

    def test_negative_clamped_to_zero(self) -> None:
        claim = _make_claim(evidence=[_make_evidence(-0.1)])
        assert compute_retrieval_quality(claim) == 0.0


# ---------------------------------------------------------------------------
# ConfidenceSignals.aggregate
# ---------------------------------------------------------------------------


class TestConfidenceSignalsAggregate:
    def test_default_weights_no_llm(self) -> None:
        signals = ConfidenceSignals(
            evidence_signal=0.9,
            hedging_signal=0.8,
            citation_density=0.6,
            retrieval_quality=0.7,
        )
        # 0.35*0.9 + 0.20*0.8 + 0.20*0.6 + 0.25*0.7 = 0.315+0.16+0.12+0.175 = 0.77
        score = signals.aggregate()
        assert score == pytest.approx(0.77, abs=0.01)

    def test_default_weights_with_llm(self) -> None:
        signals = ConfidenceSignals(
            evidence_signal=0.9,
            hedging_signal=0.8,
            citation_density=0.6,
            retrieval_quality=0.7,
            consistency_signal=1.0,
        )
        # 0.25*0.9 + 0.15*0.8 + 0.15*0.6 + 0.15*0.7 + 0.30*1.0
        # = 0.225+0.12+0.09+0.105+0.30 = 0.84
        score = signals.aggregate()
        assert score == pytest.approx(0.84, abs=0.01)

    def test_custom_weights(self) -> None:
        signals = ConfidenceSignals(
            evidence_signal=1.0,
            hedging_signal=0.0,
            citation_density=0.0,
            retrieval_quality=0.0,
        )
        score = signals.aggregate(weights={"evidence": 1.0})
        assert score == pytest.approx(1.0, abs=0.001)

    def test_clamped_to_zero(self) -> None:
        signals = ConfidenceSignals(
            evidence_signal=0.0,
            hedging_signal=0.0,
            citation_density=0.0,
            retrieval_quality=0.0,
        )
        assert signals.aggregate() == 0.0

    def test_clamped_to_one(self) -> None:
        signals = ConfidenceSignals(
            evidence_signal=1.0,
            hedging_signal=1.0,
            citation_density=1.0,
            retrieval_quality=1.0,
        )
        assert signals.aggregate() == 1.0

    def test_consistency_none_excluded(self) -> None:
        signals = ConfidenceSignals(
            evidence_signal=0.5,
            hedging_signal=0.5,
            citation_density=0.5,
            retrieval_quality=0.5,
            consistency_signal=None,
        )
        score = signals.aggregate()
        # No consistency weight: 0.35*0.5 + 0.20*0.5 + 0.20*0.5 + 0.25*0.5 = 0.5
        assert score == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# score_claim
# ---------------------------------------------------------------------------


class TestScoreClaim:
    def test_without_llm(self) -> None:
        claim = _make_claim(
            statement="The model clearly demonstrates accuracy.",
            evidence_class=EvidenceClass.SUPPORTED,
            evidence=[_make_evidence(0.85)],
        )
        score, signals = score_claim(claim, llm_provider=None)
        assert 0.0 <= score <= 1.0
        assert signals.consistency_signal is None
        assert signals.evidence_signal == 0.9
        assert signals.retrieval_quality == pytest.approx(0.85, abs=0.01)

    def test_with_mocked_llm(self) -> None:
        claim = _make_claim(
            evidence_class=EvidenceClass.SUPPORTED,
            evidence=[_make_evidence(0.8)],
        )
        mock_llm = MagicMock()
        mock_llm.call.return_value = {"supported": True, "reasoning": "ok"}

        score, signals = score_claim(claim, llm_provider=mock_llm)
        assert signals.consistency_signal is not None
        assert signals.consistency_signal == 1.0
        assert 0.0 <= score <= 1.0

    def test_llm_failure_falls_back(self) -> None:
        claim = _make_claim(
            evidence_class=EvidenceClass.SUPPORTED,
            evidence=[_make_evidence(0.8)],
        )
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("LLM unavailable")

        score, signals = score_claim(claim, llm_provider=mock_llm)
        # All LLM samples fail → _compute_consistency returns 0.5 (neutral)
        assert signals.consistency_signal == 0.5
        assert 0.0 <= score <= 1.0

    def test_no_evidence_claim(self) -> None:
        claim = _make_claim(
            evidence_class=EvidenceClass.UNSUPPORTED,
            evidence=[],
        )
        score, signals = score_claim(claim, llm_provider=None)
        assert signals.citation_density == 0.0
        assert signals.retrieval_quality == 0.0
        assert signals.evidence_signal == 0.05


# ---------------------------------------------------------------------------
# score_decomposition
# ---------------------------------------------------------------------------


class TestScoreDecomposition:
    def test_all_claims_scored(self) -> None:
        claims = [
            _make_claim(claim_id="CL-001", evidence=[_make_evidence()]),
            _make_claim(claim_id="CL-002", evidence=[_make_evidence(0.5, "c2")]),
        ]
        decomp = _make_decomposition(claims)
        result = score_decomposition(decomp, llm_provider=None)
        assert len(result.claims) == 2
        for c in result.claims:
            assert c.confidence_score > 0.0

    def test_confidence_scores_updated(self) -> None:
        claim = _make_claim(
            confidence_score=0.0,
            evidence_class=EvidenceClass.SUPPORTED,
            evidence=[_make_evidence(0.9)],
        )
        decomp = _make_decomposition([claim])
        result = score_decomposition(decomp, llm_provider=None)
        assert result.claims[0].confidence_score > 0.0

    def test_original_unchanged(self) -> None:
        claim = _make_claim(confidence_score=0.0, evidence=[_make_evidence()])
        decomp = _make_decomposition([claim])
        _ = score_decomposition(decomp, llm_provider=None)
        # Original should be unchanged (model_copy)
        assert decomp.claims[0].confidence_score == 0.0

    def test_empty_claims(self) -> None:
        decomp = _make_decomposition([])
        result = score_decomposition(decomp, llm_provider=None)
        assert len(result.claims) == 0


# ---------------------------------------------------------------------------
# _compute_consistency
# ---------------------------------------------------------------------------


class TestComputeConsistency:
    def test_all_agree(self) -> None:
        claim = _make_claim()
        mock_llm = MagicMock()
        mock_llm.call.return_value = {"supported": True, "reasoning": "ok"}
        result = _compute_consistency(claim, mock_llm, samples=5)
        assert result == 1.0

    def test_all_disagree(self) -> None:
        claim = _make_claim()
        mock_llm = MagicMock()
        mock_llm.call.return_value = {"supported": False, "reasoning": "no"}
        result = _compute_consistency(claim, mock_llm, samples=5)
        assert result == 0.0

    def test_mixed_agreement(self) -> None:
        claim = _make_claim()
        mock_llm = MagicMock()
        responses = [
            {"supported": True, "reasoning": "ok"},
            {"supported": True, "reasoning": "ok"},
            {"supported": False, "reasoning": "no"},
            {"supported": True, "reasoning": "ok"},
            {"supported": False, "reasoning": "no"},
        ]
        mock_llm.call.side_effect = responses
        result = _compute_consistency(claim, mock_llm, samples=5)
        assert result == pytest.approx(0.6, abs=0.01)

    def test_no_valid_samples(self) -> None:
        claim = _make_claim()
        mock_llm = MagicMock()
        mock_llm.call.return_value = {"invalid": "response"}
        result = _compute_consistency(claim, mock_llm, samples=5)
        assert result == 0.5

    def test_llm_exceptions_handled(self) -> None:
        claim = _make_claim()
        mock_llm = MagicMock()
        mock_llm.call.side_effect = RuntimeError("boom")
        result = _compute_consistency(claim, mock_llm, samples=3)
        # All fail → no valid samples → 0.5
        assert result == 0.5

    def test_partial_exceptions(self) -> None:
        claim = _make_claim()
        mock_llm = MagicMock()
        mock_llm.call.side_effect = [
            {"supported": True, "reasoning": "ok"},
            RuntimeError("fail"),
            {"supported": False, "reasoning": "no"},
        ]
        result = _compute_consistency(claim, mock_llm, samples=3)
        # 1 agree out of 2 valid = 0.5
        assert result == 0.5
