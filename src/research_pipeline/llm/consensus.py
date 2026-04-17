"""Multi-model consensus for critical decisions.

Uses ≥2 model families for critical pipeline decisions (screening
verdicts, quality ratings, synthesis claims). Catches single-model
hallucinations through disagreement detection.

Key patterns from deep research:
- Majority voting across model families
- Weighted voting (by historical accuracy)
- Disagreement flagging for human review
- Confidence-weighted aggregation
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AggregationStrategy(StrEnum):
    """How to aggregate multiple model responses."""

    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    MEDIAN = "median"


class DisagreementSeverity(StrEnum):
    """Severity of inter-model disagreement."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelResponse:
    """A single model's response to a prompt.

    Attributes:
        model_id: Identifier for the model (e.g. "gpt-4o", "claude-3").
        verdict: The model's decision — bool, float, or str.
        confidence: Model's self-reported confidence [0, 1].
        reasoning: Optional chain-of-thought excerpt.
        latency_ms: Response time in milliseconds.
        prompt_hash: SHA-256 of the prompt for audit.
    """

    model_id: str
    verdict: bool | float | str
    confidence: float = 1.0
    reasoning: str = ""
    latency_ms: float = 0.0
    prompt_hash: str = ""


@dataclass
class ConsensusResult:
    """Outcome of multi-model consensus.

    Attributes:
        final_verdict: The aggregated decision.
        strategy: Which aggregation strategy was used.
        agreement_ratio: Fraction of models that agree with the verdict.
        responses: All individual model responses.
        disagreement: Severity of disagreement.
        needs_human_review: Whether disagreement warrants human review.
        confidence: Aggregated confidence score.
        metadata: Additional audit data.
    """

    final_verdict: bool | float | str
    strategy: AggregationStrategy
    agreement_ratio: float
    responses: list[ModelResponse] = field(default_factory=list)
    disagreement: DisagreementSeverity = DisagreementSeverity.NONE
    needs_human_review: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_verdict": self.final_verdict,
            "strategy": self.strategy.value,
            "agreement_ratio": round(self.agreement_ratio, 4),
            "disagreement": self.disagreement.value,
            "needs_human_review": self.needs_human_review,
            "confidence": round(self.confidence, 4),
            "num_models": len(self.responses),
            "model_ids": [r.model_id for r in self.responses],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Disagreement analysis
# ---------------------------------------------------------------------------


def _classify_disagreement_binary(
    responses: list[ModelResponse],
) -> DisagreementSeverity:
    """Classify disagreement severity for binary verdicts."""
    if not responses:
        return DisagreementSeverity.NONE

    verdicts = [bool(r.verdict) for r in responses]
    true_count = sum(verdicts)
    total = len(verdicts)

    if true_count == 0 or true_count == total:
        return DisagreementSeverity.NONE
    ratio = min(true_count, total - true_count) / total
    if ratio <= 0.2:
        return DisagreementSeverity.LOW
    if ratio <= 0.4:
        return DisagreementSeverity.MODERATE
    return DisagreementSeverity.HIGH


def _classify_disagreement_numeric(
    responses: list[ModelResponse],
) -> DisagreementSeverity:
    """Classify disagreement severity for numeric verdicts."""
    if len(responses) < 2:
        return DisagreementSeverity.NONE

    values = [float(r.verdict) for r in responses]
    stdev = statistics.stdev(values)

    if stdev < 0.05:
        return DisagreementSeverity.NONE
    if stdev < 0.15:
        return DisagreementSeverity.LOW
    if stdev < 0.30:
        return DisagreementSeverity.MODERATE
    return DisagreementSeverity.HIGH


def _classify_disagreement_label(
    responses: list[ModelResponse],
) -> DisagreementSeverity:
    """Classify disagreement severity for label verdicts."""
    if not responses:
        return DisagreementSeverity.NONE

    labels = [str(r.verdict) for r in responses]
    unique = set(labels)

    if len(unique) == 1:
        return DisagreementSeverity.NONE
    ratio = len(unique) / len(labels)
    if ratio <= 0.4:
        return DisagreementSeverity.LOW
    if ratio <= 0.7:
        return DisagreementSeverity.MODERATE
    return DisagreementSeverity.HIGH


def classify_disagreement(
    responses: list[ModelResponse],
) -> DisagreementSeverity:
    """Classify the severity of inter-model disagreement.

    Automatically detects verdict type (bool, float, str) and applies
    the appropriate disagreement metric.

    Args:
        responses: List of model responses.

    Returns:
        DisagreementSeverity level.
    """
    if not responses:
        return DisagreementSeverity.NONE

    verdict_types = {type(r.verdict) for r in responses}
    if verdict_types == {bool}:
        return _classify_disagreement_binary(responses)
    if verdict_types <= {int, float}:
        return _classify_disagreement_numeric(responses)
    return _classify_disagreement_label(responses)


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def _majority_binary(responses: list[ModelResponse]) -> bool:
    """Majority vote for binary verdicts."""
    verdicts = [bool(r.verdict) for r in responses]
    true_count = sum(verdicts)
    return true_count > len(verdicts) / 2


def _weighted_binary(
    responses: list[ModelResponse],
    weights: dict[str, float] | None = None,
) -> bool:
    """Weighted vote for binary verdicts."""
    total_true = 0.0
    total_weight = 0.0
    for r in responses:
        w = (weights or {}).get(r.model_id, 1.0) * r.confidence
        if bool(r.verdict):
            total_true += w
        total_weight += w
    return total_true > total_weight / 2 if total_weight > 0 else False


def _median_numeric(responses: list[ModelResponse]) -> float:
    """Median aggregation for numeric verdicts."""
    values = [float(r.verdict) for r in responses]
    return statistics.median(values)


def _weighted_numeric(
    responses: list[ModelResponse],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted mean for numeric verdicts."""
    total = 0.0
    total_weight = 0.0
    for r in responses:
        w = (weights or {}).get(r.model_id, 1.0) * r.confidence
        total += float(r.verdict) * w
        total_weight += w
    return total / total_weight if total_weight > 0 else 0.0


def _majority_label(responses: list[ModelResponse]) -> str:
    """Majority vote for label verdicts."""
    counts: dict[str, int] = {}
    for r in responses:
        label = str(r.verdict)
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get) if counts else ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Prompt hashing
# ---------------------------------------------------------------------------


def hash_prompt(prompt: str) -> str:
    """SHA-256 hash of a prompt for audit trail."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Consensus engine
# ---------------------------------------------------------------------------


class ConsensusEngine:
    """Multi-model consensus engine.

    Collects responses from multiple models and aggregates them using
    the configured strategy.

    Args:
        strategy: Aggregation strategy.
        model_weights: Per-model reliability weights (model_id → weight).
        human_review_threshold: Disagreement level triggering human review.
        min_models: Minimum models required for consensus.
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.MAJORITY,
        model_weights: dict[str, float] | None = None,
        human_review_threshold: DisagreementSeverity = DisagreementSeverity.HIGH,
        min_models: int = 2,
    ) -> None:
        self._strategy = strategy
        self._weights = model_weights or {}
        self._review_threshold = human_review_threshold
        self._min_models = max(1, min_models)
        self._history: list[ConsensusResult] = []

    @property
    def strategy(self) -> AggregationStrategy:
        return self._strategy

    @property
    def min_models(self) -> int:
        return self._min_models

    @property
    def history(self) -> list[ConsensusResult]:
        return list(self._history)

    def evaluate(
        self,
        responses: list[ModelResponse],
    ) -> ConsensusResult:
        """Evaluate consensus across model responses.

        Args:
            responses: Model responses to aggregate.

        Returns:
            ConsensusResult with aggregated verdict and diagnostics.

        Raises:
            ValueError: If fewer than min_models responses provided.
        """
        if len(responses) < self._min_models:
            msg = (
                f"Need at least {self._min_models} model responses, "
                f"got {len(responses)}"
            )
            raise ValueError(msg)

        disagreement = classify_disagreement(responses)

        severity_order = list(DisagreementSeverity)
        needs_review = (
            severity_order.index(disagreement)
            >= severity_order.index(self._review_threshold)
        )

        verdict = self._aggregate(responses)
        agreement = self._compute_agreement(responses, verdict)
        confidence = self._compute_confidence(responses, agreement)

        result = ConsensusResult(
            final_verdict=verdict,
            strategy=self._strategy,
            agreement_ratio=agreement,
            responses=list(responses),
            disagreement=disagreement,
            needs_human_review=needs_review,
            confidence=confidence,
            metadata={
                "model_weights": dict(self._weights),
                "review_threshold": self._review_threshold.value,
            },
        )
        self._history.append(result)

        logger.info(
            "Consensus: verdict=%s, agreement=%.2f, disagreement=%s, "
            "review=%s, models=%d",
            verdict,
            agreement,
            disagreement.value,
            needs_review,
            len(responses),
        )
        return result

    def _aggregate(
        self,
        responses: list[ModelResponse],
    ) -> bool | float | str:
        """Aggregate verdicts based on strategy."""
        verdict_types = {type(r.verdict) for r in responses}
        is_binary = verdict_types == {bool}
        is_numeric = verdict_types <= {int, float}

        if self._strategy == AggregationStrategy.UNANIMOUS:
            return self._unanimous(responses)

        if self._strategy == AggregationStrategy.MEDIAN and is_numeric:
            return _median_numeric(responses)

        if self._strategy == AggregationStrategy.WEIGHTED:
            if is_binary:
                return _weighted_binary(responses, self._weights)
            if is_numeric:
                return _weighted_numeric(responses, self._weights)

        # Default: majority
        if is_binary:
            return _majority_binary(responses)
        if is_numeric:
            return _median_numeric(responses)
        return _majority_label(responses)

    def _unanimous(
        self,
        responses: list[ModelResponse],
    ) -> bool | float | str:
        """Unanimous consensus — all must agree exactly."""
        verdicts = [r.verdict for r in responses]
        if len(set(str(v) for v in verdicts)) == 1:
            return verdicts[0]
        # No unanimity — fall back to majority
        logger.warning("No unanimity among %d models, falling back to majority", len(responses))
        verdict_types = {type(r.verdict) for r in responses}
        if verdict_types == {bool}:
            return _majority_binary(responses)
        if verdict_types <= {int, float}:
            return _median_numeric(responses)
        return _majority_label(responses)

    def _compute_agreement(
        self,
        responses: list[ModelResponse],
        verdict: bool | float | str,
    ) -> float:
        """Compute fraction of models agreeing with the final verdict."""
        if not responses:
            return 0.0

        agrees = 0
        for r in responses:
            if isinstance(verdict, bool) and isinstance(r.verdict, bool):
                if r.verdict == verdict:
                    agrees += 1
            elif isinstance(verdict, float | int) and isinstance(r.verdict, float | int):
                if abs(float(r.verdict) - float(verdict)) < 0.1:
                    agrees += 1
            elif str(r.verdict) == str(verdict):
                agrees += 1

        return agrees / len(responses)

    def _compute_confidence(
        self,
        responses: list[ModelResponse],
        agreement: float,
    ) -> float:
        """Compute aggregated confidence."""
        if not responses:
            return 0.0
        avg_conf = statistics.mean(r.confidence for r in responses)
        return avg_conf * agreement

    def summary(self) -> dict[str, Any]:
        """Get engine summary with history stats."""
        if not self._history:
            return {
                "strategy": self._strategy.value,
                "total_evaluations": 0,
            }
        agreements = [r.agreement_ratio for r in self._history]
        reviews = sum(1 for r in self._history if r.needs_human_review)
        return {
            "strategy": self._strategy.value,
            "total_evaluations": len(self._history),
            "mean_agreement": round(statistics.mean(agreements), 4),
            "human_reviews_triggered": reviews,
            "disagreement_distribution": self._disagreement_distribution(),
        }

    def _disagreement_distribution(self) -> dict[str, int]:
        """Count disagreement levels across history."""
        dist: dict[str, int] = {}
        for r in self._history:
            key = r.disagreement.value
            dist[key] = dist.get(key, 0) + 1
        return dist
