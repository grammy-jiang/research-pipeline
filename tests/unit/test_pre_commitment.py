"""Tests for evaluation.pre_commitment — pre-commitment protocol."""

from __future__ import annotations

import pytest

from research_pipeline.evaluation.pre_commitment import (
    Disagreement,
    IndependentAssessment,
    PreCommitmentProtocol,
    PreCommitmentRound,
    ProtocolState,
    ReconciliationResult,
    ReconciliationStrategy,
    VerdictType,
)

# ---------------------------------------------------------------------------
# IndependentAssessment
# ---------------------------------------------------------------------------


class TestIndependentAssessment:
    def test_binary_creation(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict=True,
            verdict_type=VerdictType.BINARY,
            confidence=0.9,
            evidence="strong",
        )
        assert a.evaluator_id == "m1"
        assert a.verdict is True
        assert a.confidence == 0.9

    def test_scalar_creation(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict=0.75,
            verdict_type=VerdictType.SCALAR,
        )
        assert a.verdict == 0.75

    def test_label_creation(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict="relevant",
            verdict_type=VerdictType.LABEL,
        )
        assert a.verdict == "relevant"

    def test_confidence_clamped(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict=True,
            verdict_type=VerdictType.BINARY,
            confidence=1.5,
        )
        assert a.confidence == 1.0
        b = IndependentAssessment(
            evaluator_id="m2",
            verdict=True,
            verdict_type=VerdictType.BINARY,
            confidence=-0.5,
        )
        assert b.confidence == 0.0

    def test_integrity_check(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict=True,
            verdict_type=VerdictType.BINARY,
        )
        assert a.verify_integrity()

    def test_integrity_tampered(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict=True,
            verdict_type=VerdictType.BINARY,
        )
        a.evidence = "modified after lock"
        assert not a.verify_integrity()

    def test_to_dict(self) -> None:
        a = IndependentAssessment(
            evaluator_id="m1",
            verdict=True,
            verdict_type=VerdictType.BINARY,
            confidence=0.8,
            evidence="proof",
        )
        d = a.to_dict()
        assert d["evaluator_id"] == "m1"
        assert d["verdict"] is True
        assert d["verdict_type"] == "binary"
        assert d["confidence"] == 0.8
        assert "commitment_hash" in d


# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


class TestDisagreement:
    def test_to_dict(self) -> None:
        d = Disagreement(
            evaluator_a="m1",
            evaluator_b="m2",
            verdict_a=True,
            verdict_b=False,
            severity=0.3,
        )
        out = d.to_dict()
        assert out["evaluator_a"] == "m1"
        assert out["severity"] == 0.3


# ---------------------------------------------------------------------------
# ReconciliationResult
# ---------------------------------------------------------------------------


class TestReconciliationResult:
    def test_to_dict(self) -> None:
        r = ReconciliationResult(
            final_verdict=True,
            strategy=ReconciliationStrategy.MAJORITY,
            agreement_ratio=0.67,
            confidence=0.8,
        )
        d = r.to_dict()
        assert d["final_verdict"] is True
        assert d["strategy"] == "majority"
        assert d["disagreement_count"] == 0


# ---------------------------------------------------------------------------
# PreCommitmentRound — binary
# ---------------------------------------------------------------------------


class TestPreCommitmentRoundBinary:
    def test_basic_majority(self) -> None:
        rnd = PreCommitmentRound(item_id="p1")
        rnd.register("a")
        rnd.register("b")
        rnd.register("c")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=True, confidence=0.8)
        rnd.submit("c", verdict=False, confidence=0.6)
        result = rnd.reconcile()
        assert result.final_verdict is True
        assert result.agreement_ratio == pytest.approx(2 / 3)
        assert rnd.is_complete

    def test_majority_false_wins(self) -> None:
        rnd = PreCommitmentRound(item_id="p2")
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=False, confidence=0.9)
        rnd.submit("b", verdict=False, confidence=0.8)
        result = rnd.reconcile()
        assert result.final_verdict is False
        assert result.agreement_ratio == 1.0

    def test_weighted_strategy(self) -> None:
        rnd = PreCommitmentRound(
            item_id="p3",
            strategy=ReconciliationStrategy.WEIGHTED,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True, confidence=0.3)
        rnd.submit("b", verdict=False, confidence=0.9)
        result = rnd.reconcile()
        assert result.final_verdict is False

    def test_unanimous_all_agree(self) -> None:
        rnd = PreCommitmentRound(
            item_id="p4",
            strategy=ReconciliationStrategy.UNANIMOUS,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=True, confidence=0.8)
        result = rnd.reconcile()
        assert result.final_verdict is True
        assert result.agreement_ratio == 1.0

    def test_unanimous_disagree(self) -> None:
        rnd = PreCommitmentRound(
            item_id="p5",
            strategy=ReconciliationStrategy.UNANIMOUS,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=False, confidence=0.8)
        result = rnd.reconcile()
        assert result.final_verdict is False
        assert result.agreement_ratio == 0.0

    def test_disagreements_detected(self) -> None:
        rnd = PreCommitmentRound(item_id="p6")
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=False, confidence=0.7)
        result = rnd.reconcile()
        assert len(result.disagreements) == 1
        assert result.disagreements[0].evaluator_a == "a"

    def test_needs_human_review(self) -> None:
        rnd = PreCommitmentRound(
            item_id="p7",
            disagreement_threshold=0.8,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=False, confidence=0.8)
        result = rnd.reconcile()
        assert result.needs_human_review


# ---------------------------------------------------------------------------
# PreCommitmentRound — scalar
# ---------------------------------------------------------------------------


class TestPreCommitmentRoundScalar:
    def test_mean_scalar(self) -> None:
        rnd = PreCommitmentRound(item_id="s1")
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=0.8)
        rnd.submit("b", verdict=0.6)
        result = rnd.reconcile()
        assert result.final_verdict == pytest.approx(0.7)

    def test_weighted_scalar(self) -> None:
        rnd = PreCommitmentRound(
            item_id="s2",
            strategy=ReconciliationStrategy.WEIGHTED,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=0.8, confidence=0.9)
        rnd.submit("b", verdict=0.2, confidence=0.1)
        result = rnd.reconcile()
        assert result.final_verdict > 0.7

    def test_scalar_disagreement(self) -> None:
        rnd = PreCommitmentRound(
            item_id="s3",
            disagreement_threshold=0.3,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=0.9)
        rnd.submit("b", verdict=0.1)
        result = rnd.reconcile()
        assert len(result.disagreements) > 0


# ---------------------------------------------------------------------------
# PreCommitmentRound — label
# ---------------------------------------------------------------------------


class TestPreCommitmentRoundLabel:
    def test_majority_label(self) -> None:
        rnd = PreCommitmentRound(item_id="l1")
        rnd.register("a")
        rnd.register("b")
        rnd.register("c")
        rnd.submit("a", verdict="relevant")
        rnd.submit("b", verdict="relevant")
        rnd.submit("c", verdict="irrelevant")
        result = rnd.reconcile()
        assert result.final_verdict == "relevant"
        assert result.agreement_ratio == pytest.approx(2 / 3)

    def test_weighted_label(self) -> None:
        rnd = PreCommitmentRound(
            item_id="l2",
            strategy=ReconciliationStrategy.WEIGHTED,
        )
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict="low", confidence=0.3)
        rnd.submit("b", verdict="high", confidence=0.9)
        result = rnd.reconcile()
        assert result.final_verdict == "high"


# ---------------------------------------------------------------------------
# PreCommitmentRound — error handling
# ---------------------------------------------------------------------------


class TestPreCommitmentRoundErrors:
    def test_register_after_lock(self) -> None:
        rnd = PreCommitmentRound(item_id="e1")
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True)
        rnd.submit("b", verdict=True)
        assert rnd.state == ProtocolState.LOCKED
        with pytest.raises(RuntimeError, match="Cannot register"):
            rnd.register("c")

    def test_submit_unregistered(self) -> None:
        rnd = PreCommitmentRound(item_id="e2")
        rnd.register("a")
        with pytest.raises(ValueError, match="not registered"):
            rnd.submit("b", verdict=True)

    def test_double_submit(self) -> None:
        rnd = PreCommitmentRound(item_id="e3")
        rnd.register("a")
        rnd.submit("a", verdict=True)
        with pytest.raises(ValueError, match="already submitted"):
            rnd.submit("a", verdict=False)

    def test_submit_after_reconcile(self) -> None:
        rnd = PreCommitmentRound(item_id="e4", min_evaluators=1)
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True)
        rnd.reconcile()
        with pytest.raises(RuntimeError, match="already reconciled"):
            rnd.submit("b", verdict=False)

    def test_reconcile_too_few(self) -> None:
        rnd = PreCommitmentRound(item_id="e5", min_evaluators=3)
        rnd.register("a")
        rnd.register("b")
        rnd.register("c")
        rnd.submit("a", verdict=True)
        with pytest.raises(RuntimeError, match="Need at least 3"):
            rnd.reconcile()


# ---------------------------------------------------------------------------
# PreCommitmentRound — serialization
# ---------------------------------------------------------------------------


class TestPreCommitmentRoundSerialization:
    def test_to_dict(self) -> None:
        rnd = PreCommitmentRound(item_id="ser1")
        rnd.register("a")
        rnd.register("b")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=False, confidence=0.7)
        rnd.reconcile()
        d = rnd.to_dict()
        assert d["item_id"] == "ser1"
        assert d["state"] == "reconciled"
        assert "a" in d["assessments"]
        assert d["result"] is not None
        assert d["result"]["strategy"] == "majority"


# ---------------------------------------------------------------------------
# PreCommitmentProtocol
# ---------------------------------------------------------------------------


class TestPreCommitmentProtocol:
    def test_create_round(self) -> None:
        proto = PreCommitmentProtocol(evaluator_ids=["a", "b"])
        rnd = proto.create_round("paper-1")
        assert rnd.item_id == "paper-1"
        assert "a" in rnd.evaluators
        assert "b" in rnd.evaluators

    def test_duplicate_round_error(self) -> None:
        proto = PreCommitmentProtocol(evaluator_ids=["a"])
        proto.create_round("paper-1")
        with pytest.raises(ValueError, match="already exists"):
            proto.create_round("paper-1")

    def test_get_round(self) -> None:
        proto = PreCommitmentProtocol(evaluator_ids=["a"])
        proto.create_round("paper-1")
        assert proto.get_round("paper-1") is not None
        assert proto.get_round("paper-999") is None

    def test_completed_and_pending(self) -> None:
        proto = PreCommitmentProtocol(
            evaluator_ids=["a", "b"],
            min_evaluators=2,
        )
        r1 = proto.create_round("p1")
        r1.submit("a", verdict=True)
        r1.submit("b", verdict=True)
        r1.reconcile()
        proto.create_round("p2")
        assert proto.completed_count == 1
        assert proto.pending_count == 1

    def test_reconcile_all(self) -> None:
        proto = PreCommitmentProtocol(
            evaluator_ids=["a", "b"],
            min_evaluators=2,
        )
        r1 = proto.create_round("p1")
        r1.submit("a", verdict=True)
        r1.submit("b", verdict=False)
        r2 = proto.create_round("p2")
        r2.submit("a", verdict=0.8)
        r2.submit("b", verdict=0.6)
        results = proto.reconcile_all()
        assert "p1" in results
        assert "p2" in results

    def test_agreement_stats_empty(self) -> None:
        proto = PreCommitmentProtocol(evaluator_ids=["a"])
        stats = proto.agreement_stats()
        assert stats["completed"] == 0

    def test_agreement_stats(self) -> None:
        proto = PreCommitmentProtocol(
            evaluator_ids=["a", "b"],
            min_evaluators=2,
        )
        r1 = proto.create_round("p1")
        r1.submit("a", verdict=True)
        r1.submit("b", verdict=True)
        r1.reconcile()
        r2 = proto.create_round("p2")
        r2.submit("a", verdict=True)
        r2.submit("b", verdict=False)
        r2.reconcile()
        stats = proto.agreement_stats()
        assert stats["completed"] == 2
        assert stats["mean_agreement"] > 0

    def test_summary(self) -> None:
        proto = PreCommitmentProtocol(
            evaluator_ids=["a", "b"],
            strategy=ReconciliationStrategy.WEIGHTED,
        )
        rnd = proto.create_round("p1")
        rnd.submit("a", verdict=True, confidence=0.9)
        rnd.submit("b", verdict=True, confidence=0.8)
        rnd.reconcile()
        s = proto.summary()
        assert s["total_rounds"] == 1
        assert s["strategy"] == "weighted"
        assert "agreement" in s

    def test_custom_strategy(self) -> None:
        proto = PreCommitmentProtocol(
            evaluator_ids=["a", "b"],
            strategy=ReconciliationStrategy.UNANIMOUS,
            min_evaluators=2,
        )
        rnd = proto.create_round("p1")
        rnd.submit("a", verdict=True)
        rnd.submit("b", verdict=True)
        result = rnd.reconcile()
        assert result.final_verdict is True
        assert result.agreement_ratio == 1.0

    def test_high_disagreement_threshold(self) -> None:
        proto = PreCommitmentProtocol(
            evaluator_ids=["a", "b", "c"],
            disagreement_threshold=0.9,
            min_evaluators=3,
        )
        rnd = proto.create_round("p1")
        rnd.submit("a", verdict=True)
        rnd.submit("b", verdict=True)
        rnd.submit("c", verdict=False)
        result = rnd.reconcile()
        assert result.needs_human_review


# ---------------------------------------------------------------------------
# ProtocolState transitions
# ---------------------------------------------------------------------------


class TestProtocolStateTransitions:
    def test_open_to_locked(self) -> None:
        rnd = PreCommitmentRound(item_id="st1")
        assert rnd.state == ProtocolState.OPEN
        rnd.register("a")
        rnd.submit("a", verdict=True)
        assert rnd.state == ProtocolState.LOCKED

    def test_locked_to_reconciled(self) -> None:
        rnd = PreCommitmentRound(item_id="st2", min_evaluators=1)
        rnd.register("a")
        rnd.submit("a", verdict=True)
        rnd.reconcile()
        assert rnd.state == ProtocolState.RECONCILED

    def test_reconcile_idempotent(self) -> None:
        rnd = PreCommitmentRound(item_id="st3", min_evaluators=1)
        rnd.register("a")
        rnd.submit("a", verdict=True)
        r1 = rnd.reconcile()
        r2 = rnd.reconcile()
        assert r1 is r2
