"""Pre-commitment protocol for eliminating conformity bias.

Implements the Social Dynamics pattern from multi-agent research (arXiv
2604.06013): each evaluator forms an independent assessment *before*
seeing any other evaluator's output.  Only after all independent
assessments are locked does a reconciliation step merge them.

Protocol stages:
  1. **Register** evaluators (agent / model / persona identifiers).
  2. **Lock** independent assessments — each evaluator submits a
     verdict with evidence *before* any cross-evaluator information
     is revealed.
  3. **Reconcile** — majority vote, disagreement flagging, confidence
     weighting, and optional consensus-seeking.

Design goals:
  - Zero external dependencies (stdlib + dataclasses only).
  - Deterministic — same inputs always produce the same reconciled
    result.
  - Composable — works for relevance screening, quality scoring,
    synthesis claims, or any binary/scalar decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerdictType(StrEnum):
    """Supported verdict value types."""

    BINARY = "binary"  # True / False
    SCALAR = "scalar"  # float in [0, 1]
    LABEL = "label"  # arbitrary string label


class ReconciliationStrategy(StrEnum):
    """How to merge independent verdicts."""

    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"


class ProtocolState(StrEnum):
    """State machine for the pre-commitment round."""

    OPEN = "open"  # accepting registrations
    LOCKED = "locked"  # assessments submitted, no more changes
    RECONCILED = "reconciled"  # final result computed


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IndependentAssessment:
    """A single evaluator's locked assessment."""

    evaluator_id: str
    verdict: bool | float | str
    verdict_type: VerdictType
    confidence: float = 1.0  # [0, 1]
    evidence: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    _commitment_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not self._commitment_hash:
            self._commitment_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "evaluator_id": self.evaluator_id,
                "verdict": self._serialise_verdict(),
                "confidence": round(self.confidence, 6),
                "evidence": self.evidence,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _serialise_verdict(self) -> str:
        if self.verdict_type == VerdictType.BINARY:
            return str(bool(self.verdict))
        if self.verdict_type == VerdictType.SCALAR:
            return str(round(float(self.verdict), 6))
        return str(self.verdict)

    def verify_integrity(self) -> bool:
        """Check that the assessment has not been tampered with."""
        return self._commitment_hash == self._compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "verdict": self.verdict,
            "verdict_type": self.verdict_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "commitment_hash": self._commitment_hash,
        }


@dataclass
class Disagreement:
    """Describes a disagreement between two evaluators."""

    evaluator_a: str
    evaluator_b: str
    verdict_a: bool | float | str
    verdict_b: bool | float | str
    severity: float = 0.0  # [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_a": self.evaluator_a,
            "evaluator_b": self.evaluator_b,
            "verdict_a": self.verdict_a,
            "verdict_b": self.verdict_b,
            "severity": round(self.severity, 4),
        }


@dataclass
class ReconciliationResult:
    """Merged outcome after reconciliation."""

    final_verdict: bool | float | str
    strategy: ReconciliationStrategy
    agreement_ratio: float  # fraction of evaluators that agree with final
    disagreements: list[Disagreement] = field(default_factory=list)
    confidence: float = 0.0
    needs_human_review: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_verdict": self.final_verdict,
            "strategy": self.strategy.value,
            "agreement_ratio": round(self.agreement_ratio, 4),
            "confidence": round(self.confidence, 4),
            "needs_human_review": self.needs_human_review,
            "disagreement_count": len(self.disagreements),
            "disagreements": [d.to_dict() for d in self.disagreements],
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Pre-commitment round
# ---------------------------------------------------------------------------


class PreCommitmentRound:
    """Manages one pre-commitment round for a single decision.

    Usage::

        rnd = PreCommitmentRound(item_id="paper-42")
        rnd.register("model-a")
        rnd.register("model-b")
        rnd.submit("model-a", verdict=True, confidence=0.9, evidence="Strong method")
        rnd.submit("model-b", verdict=False, confidence=0.7, evidence="Weak dataset")
        result = rnd.reconcile()
    """

    def __init__(
        self,
        item_id: str = "",
        strategy: ReconciliationStrategy = ReconciliationStrategy.MAJORITY,
        min_evaluators: int = 2,
        disagreement_threshold: float = 0.5,
    ) -> None:
        self.item_id = item_id
        self.strategy = strategy
        self.min_evaluators = max(1, min_evaluators)
        self.disagreement_threshold = disagreement_threshold
        self._evaluators: set[str] = set()
        self._assessments: dict[str, IndependentAssessment] = {}
        self._state = ProtocolState.OPEN
        self._result: ReconciliationResult | None = None

    # -- properties --

    @property
    def state(self) -> ProtocolState:
        return self._state

    @property
    def evaluators(self) -> frozenset[str]:
        return frozenset(self._evaluators)

    @property
    def assessments(self) -> dict[str, IndependentAssessment]:
        return dict(self._assessments)

    @property
    def result(self) -> ReconciliationResult | None:
        return self._result

    @property
    def is_complete(self) -> bool:
        return self._state == ProtocolState.RECONCILED

    # -- registration --

    def register(self, evaluator_id: str) -> None:
        """Register an evaluator for this round."""
        if self._state != ProtocolState.OPEN:
            raise RuntimeError(f"Cannot register: round is {self._state.value}")
        self._evaluators.add(evaluator_id)
        logger.debug("Registered evaluator %s for item %s", evaluator_id, self.item_id)

    # -- submission --

    def submit(
        self,
        evaluator_id: str,
        verdict: bool | float | str,
        confidence: float = 1.0,
        evidence: str = "",
    ) -> IndependentAssessment:
        """Submit an independent assessment (locks on first submit per evaluator)."""
        if self._state == ProtocolState.RECONCILED:
            raise RuntimeError("Round already reconciled")
        if evaluator_id not in self._evaluators:
            raise ValueError(f"Evaluator {evaluator_id!r} not registered")
        if evaluator_id in self._assessments:
            raise ValueError(
                f"Evaluator {evaluator_id!r} already submitted "
                "(pre-commitment is immutable)"
            )

        vtype = self._infer_verdict_type(verdict)
        assessment = IndependentAssessment(
            evaluator_id=evaluator_id,
            verdict=verdict,
            verdict_type=vtype,
            confidence=confidence,
            evidence=evidence,
        )
        self._assessments[evaluator_id] = assessment

        if len(self._assessments) == len(self._evaluators):
            self._state = ProtocolState.LOCKED
            logger.debug("Round for %s is now LOCKED", self.item_id)

        return assessment

    # -- reconciliation --

    def reconcile(self) -> ReconciliationResult:
        """Reconcile all submitted assessments into a single verdict."""
        if self._state == ProtocolState.RECONCILED and self._result is not None:
            return self._result
        if len(self._assessments) < self.min_evaluators:
            raise RuntimeError(
                f"Need at least {self.min_evaluators} assessments, "
                f"got {len(self._assessments)}"
            )

        # Lock if not already
        self._state = ProtocolState.LOCKED

        assessments = list(self._assessments.values())
        vtype = assessments[0].verdict_type

        if vtype == VerdictType.BINARY:
            result = self._reconcile_binary(assessments)
        elif vtype == VerdictType.SCALAR:
            result = self._reconcile_scalar(assessments)
        else:
            result = self._reconcile_label(assessments)

        self._result = result
        self._state = ProtocolState.RECONCILED
        logger.info(
            "Reconciled %s: verdict=%s agreement=%.2f",
            self.item_id,
            result.final_verdict,
            result.agreement_ratio,
        )
        return result

    # -- private reconciliation methods --

    def _reconcile_binary(
        self, assessments: list[IndependentAssessment]
    ) -> ReconciliationResult:
        true_count = sum(1 for a in assessments if a.verdict)
        false_count = len(assessments) - true_count

        if self.strategy == ReconciliationStrategy.UNANIMOUS:
            final = true_count == len(assessments)
            agreement = 1.0 if (true_count == 0 or false_count == 0) else 0.0
        elif self.strategy == ReconciliationStrategy.WEIGHTED:
            weighted_true = sum(a.confidence for a in assessments if a.verdict)
            weighted_false = sum(a.confidence for a in assessments if not a.verdict)
            final = weighted_true >= weighted_false
            total_weight = weighted_true + weighted_false
            agreement = (
                max(weighted_true, weighted_false) / total_weight
                if total_weight > 0
                else 0.0
            )
        else:  # MAJORITY
            final = true_count > false_count
            agreement = max(true_count, false_count) / len(assessments)

        disagreements = self._find_binary_disagreements(assessments, final)
        conf = statistics.mean(a.confidence for a in assessments)

        return ReconciliationResult(
            final_verdict=final,
            strategy=self.strategy,
            agreement_ratio=agreement,
            disagreements=disagreements,
            confidence=conf,
            needs_human_review=agreement < self.disagreement_threshold,
            detail={
                "true_count": true_count,
                "false_count": false_count,
            },
        )

    def _reconcile_scalar(
        self, assessments: list[IndependentAssessment]
    ) -> ReconciliationResult:
        values = [float(a.verdict) for a in assessments]
        confidences = [a.confidence for a in assessments]

        if self.strategy == ReconciliationStrategy.WEIGHTED:
            total_w = sum(confidences)
            if total_w > 0:
                final = (
                    sum(v * c for v, c in zip(values, confidences, strict=False))
                    / total_w
                )
            else:
                final = statistics.mean(values)
        else:
            final = statistics.mean(values)

        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        agreement = max(0.0, 1.0 - stdev)

        disagreements = self._find_scalar_disagreements(assessments)
        conf = statistics.mean(confidences)

        return ReconciliationResult(
            final_verdict=round(final, 6),
            strategy=self.strategy,
            agreement_ratio=round(agreement, 6),
            disagreements=disagreements,
            confidence=conf,
            needs_human_review=agreement < self.disagreement_threshold,
            detail={
                "values": values,
                "stdev": round(stdev, 6),
            },
        )

    def _reconcile_label(
        self, assessments: list[IndependentAssessment]
    ) -> ReconciliationResult:
        labels = [str(a.verdict) for a in assessments]
        label_counts: dict[str, int] = {}
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        if self.strategy == ReconciliationStrategy.WEIGHTED:
            label_weights: dict[str, float] = {}
            for a in assessments:
                lbl = str(a.verdict)
                label_weights[lbl] = label_weights.get(lbl, 0.0) + a.confidence
            final = max(label_weights, key=label_weights.get)  # type: ignore[arg-type]
        elif self.strategy == ReconciliationStrategy.UNANIMOUS:
            if len(label_counts) == 1:
                final = labels[0]
            else:
                final = max(label_counts, key=label_counts.get)  # type: ignore[arg-type]
        else:
            final = max(label_counts, key=label_counts.get)  # type: ignore[arg-type]

        agreement = label_counts.get(final, 0) / len(assessments)
        disagreements = self._find_label_disagreements(assessments, final)
        conf = statistics.mean(a.confidence for a in assessments)

        return ReconciliationResult(
            final_verdict=final,
            strategy=self.strategy,
            agreement_ratio=agreement,
            disagreements=disagreements,
            confidence=conf,
            needs_human_review=agreement < self.disagreement_threshold,
            detail={"label_counts": label_counts},
        )

    # -- disagreement detection --

    def _find_binary_disagreements(
        self,
        assessments: list[IndependentAssessment],
        final: bool,
    ) -> list[Disagreement]:
        dissenters = [a for a in assessments if a.verdict != final]
        agreers = [a for a in assessments if a.verdict == final]
        result: list[Disagreement] = []
        for d in dissenters:
            for ag in agreers:
                result.append(
                    Disagreement(
                        evaluator_a=d.evaluator_id,
                        evaluator_b=ag.evaluator_id,
                        verdict_a=d.verdict,
                        verdict_b=ag.verdict,
                        severity=abs(d.confidence - ag.confidence),
                    )
                )
        return result

    def _find_scalar_disagreements(
        self, assessments: list[IndependentAssessment]
    ) -> list[Disagreement]:
        result: list[Disagreement] = []
        for i, a in enumerate(assessments):
            for b in assessments[i + 1 :]:
                diff = abs(float(a.verdict) - float(b.verdict))
                if diff > self.disagreement_threshold:
                    result.append(
                        Disagreement(
                            evaluator_a=a.evaluator_id,
                            evaluator_b=b.evaluator_id,
                            verdict_a=a.verdict,
                            verdict_b=b.verdict,
                            severity=diff,
                        )
                    )
        return result

    def _find_label_disagreements(
        self,
        assessments: list[IndependentAssessment],
        final: str,
    ) -> list[Disagreement]:
        dissenters = [a for a in assessments if str(a.verdict) != final]
        agreers = [a for a in assessments if str(a.verdict) == final]
        result: list[Disagreement] = []
        for d in dissenters:
            for ag in agreers:
                result.append(
                    Disagreement(
                        evaluator_a=d.evaluator_id,
                        evaluator_b=ag.evaluator_id,
                        verdict_a=d.verdict,
                        verdict_b=ag.verdict,
                        severity=1.0,
                    )
                )
        return result

    # -- helpers --

    @staticmethod
    def _infer_verdict_type(verdict: bool | float | str) -> VerdictType:
        if isinstance(verdict, bool):
            return VerdictType.BINARY
        if isinstance(verdict, int | float):
            return VerdictType.SCALAR
        return VerdictType.LABEL

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire round state."""
        return {
            "item_id": self.item_id,
            "state": self._state.value,
            "strategy": self.strategy.value,
            "min_evaluators": self.min_evaluators,
            "evaluators": sorted(self._evaluators),
            "assessments": {k: v.to_dict() for k, v in self._assessments.items()},
            "result": self._result.to_dict() if self._result else None,
        }


# ---------------------------------------------------------------------------
# Multi-round manager
# ---------------------------------------------------------------------------


class PreCommitmentProtocol:
    """Manage multiple pre-commitment rounds across items.

    Usage::

        proto = PreCommitmentProtocol(
            evaluator_ids=["gpt-4o", "claude-3", "local-llama"],
        )
        rnd = proto.create_round("paper-42")
        rnd.submit("gpt-4o", verdict=True, confidence=0.9)
        rnd.submit("claude-3", verdict=True, confidence=0.8)
        rnd.submit("local-llama", verdict=False, confidence=0.6)
        result = rnd.reconcile()

        report = proto.summary()
    """

    def __init__(
        self,
        evaluator_ids: list[str] | None = None,
        strategy: ReconciliationStrategy = ReconciliationStrategy.MAJORITY,
        min_evaluators: int = 2,
        disagreement_threshold: float = 0.5,
    ) -> None:
        self.evaluator_ids = list(evaluator_ids or [])
        self.strategy = strategy
        self.min_evaluators = min_evaluators
        self.disagreement_threshold = disagreement_threshold
        self._rounds: dict[str, PreCommitmentRound] = {}

    def create_round(self, item_id: str) -> PreCommitmentRound:
        """Create a new pre-commitment round for an item."""
        if item_id in self._rounds:
            raise ValueError(f"Round for {item_id!r} already exists")
        rnd = PreCommitmentRound(
            item_id=item_id,
            strategy=self.strategy,
            min_evaluators=self.min_evaluators,
            disagreement_threshold=self.disagreement_threshold,
        )
        for eid in self.evaluator_ids:
            rnd.register(eid)
        self._rounds[item_id] = rnd
        return rnd

    def get_round(self, item_id: str) -> PreCommitmentRound | None:
        """Retrieve an existing round."""
        return self._rounds.get(item_id)

    @property
    def rounds(self) -> dict[str, PreCommitmentRound]:
        return dict(self._rounds)

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self._rounds.values() if r.is_complete)

    @property
    def pending_count(self) -> int:
        return sum(1 for r in self._rounds.values() if not r.is_complete)

    def reconcile_all(self) -> dict[str, ReconciliationResult]:
        """Reconcile all rounds that have enough assessments."""
        results: dict[str, ReconciliationResult] = {}
        for item_id, rnd in self._rounds.items():
            if rnd.is_complete:
                if rnd.result is not None:
                    results[item_id] = rnd.result
                continue
            if len(rnd.assessments) >= rnd.min_evaluators:
                results[item_id] = rnd.reconcile()
        return results

    def agreement_stats(self) -> dict[str, Any]:
        """Compute aggregate agreement statistics across completed rounds."""
        completed = [
            r for r in self._rounds.values() if r.is_complete and r.result is not None
        ]
        if not completed:
            return {
                "completed": 0,
                "mean_agreement": 0.0,
                "min_agreement": 0.0,
                "max_agreement": 0.0,
                "human_review_count": 0,
                "human_review_fraction": 0.0,
            }

        agreements = [r.result.agreement_ratio for r in completed if r.result]
        review_count = sum(
            1 for r in completed if r.result and r.result.needs_human_review
        )

        return {
            "completed": len(completed),
            "mean_agreement": round(statistics.mean(agreements), 4),
            "min_agreement": round(min(agreements), 4),
            "max_agreement": round(max(agreements), 4),
            "human_review_count": review_count,
            "human_review_fraction": round(review_count / len(completed), 4),
        }

    def summary(self) -> dict[str, Any]:
        """Full protocol summary."""
        return {
            "total_rounds": len(self._rounds),
            "completed": self.completed_count,
            "pending": self.pending_count,
            "evaluator_ids": self.evaluator_ids,
            "strategy": self.strategy.value,
            "agreement": self.agreement_stats(),
            "rounds": {k: v.to_dict() for k, v in self._rounds.items()},
        }
