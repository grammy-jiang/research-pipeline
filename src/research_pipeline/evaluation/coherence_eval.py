"""7-Dimension coherence evaluation framework.

Multi-session coherence evaluation across seven quality dimensions
inspired by research on LLM consistency, knowledge maintenance, and
memory-system resilience:

1. **Factual consistency** — do assertions remain self-consistent?
2. **Temporal ordering** — are events/findings in correct chronological order?
3. **Knowledge update fidelity** — are corrections properly integrated?
4. **Cross-session reasoning** — can conclusions span session boundaries?
5. **Contradiction detection** — are contradictions flagged and resolved?
6. **Coherence degradation curve** — how does quality degrade over scale?
7. **Memory pressure resilience** — does quality hold under memory limits?

Each dimension produces a 0–1 score.  The composite ``CoherenceReport``
aggregates them with configurable weights.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CoherenceDimension(Enum):
    """The seven evaluation dimensions."""

    FACTUAL_CONSISTENCY = "factual_consistency"
    TEMPORAL_ORDERING = "temporal_ordering"
    KNOWLEDGE_UPDATE_FIDELITY = "knowledge_update_fidelity"
    CROSS_SESSION_REASONING = "cross_session_reasoning"
    CONTRADICTION_DETECTION = "contradiction_detection"
    COHERENCE_DEGRADATION = "coherence_degradation"
    MEMORY_PRESSURE_RESILIENCE = "memory_pressure_resilience"


class Severity(Enum):
    """Issue severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[CoherenceDimension, float] = {
    CoherenceDimension.FACTUAL_CONSISTENCY: 0.20,
    CoherenceDimension.TEMPORAL_ORDERING: 0.10,
    CoherenceDimension.KNOWLEDGE_UPDATE_FIDELITY: 0.15,
    CoherenceDimension.CROSS_SESSION_REASONING: 0.15,
    CoherenceDimension.CONTRADICTION_DETECTION: 0.15,
    CoherenceDimension.COHERENCE_DEGRADATION: 0.10,
    CoherenceDimension.MEMORY_PRESSURE_RESILIENCE: 0.15,
}


@dataclass(frozen=True)
class CoherenceIssue:
    """A single coherence issue found during evaluation."""

    dimension: CoherenceDimension
    severity: Severity
    description: str
    evidence: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single coherence dimension."""

    dimension: CoherenceDimension
    score: float  # 0.0 – 1.0
    issues: tuple[CoherenceIssue, ...] = ()
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 4),
            "num_issues": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "details": self.details,
        }


@dataclass
class CoherenceReport:
    """Aggregated coherence evaluation report."""

    dimension_scores: dict[CoherenceDimension, DimensionScore] = field(
        default_factory=dict
    )
    weights: dict[CoherenceDimension, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )

    @property
    def composite_score(self) -> float:
        """Weighted average of dimension scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for dim, w in self.weights.items():
            if dim in self.dimension_scores:
                weighted_sum += w * self.dimension_scores[dim].score
                total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @property
    def all_issues(self) -> list[CoherenceIssue]:
        """Flatten all issues across dimensions."""
        issues: list[CoherenceIssue] = []
        for ds in self.dimension_scores.values():
            issues.extend(ds.issues)
        return issues

    @property
    def critical_issues(self) -> list[CoherenceIssue]:
        return [i for i in self.all_issues if i.severity == Severity.CRITICAL]

    def to_dict(self) -> dict:
        return {
            "composite_score": round(self.composite_score, 4),
            "dimensions": {
                dim.value: ds.to_dict() for dim, ds in self.dimension_scores.items()
            },
            "total_issues": len(self.all_issues),
            "critical_issues": len(self.critical_issues),
        }


# ---------------------------------------------------------------------------
# Individual dimension evaluators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assertion:
    """A factual assertion extracted from a session."""

    text: str
    session_id: str = ""
    timestamp: float = 0.0
    source: str = ""


def evaluate_factual_consistency(
    assertions: Sequence[Assertion],
    contradiction_pairs: Sequence[tuple[int, int]] | None = None,
) -> DimensionScore:
    """Evaluate factual consistency across assertions.

    Parameters
    ----------
    assertions:
        All factual assertions extracted from sessions.
    contradiction_pairs:
        Indices of contradicting assertion pairs (pre-computed or detected).
    """
    if not assertions:
        return DimensionScore(
            dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
            score=1.0,
            details="No assertions to evaluate.",
        )

    pairs = contradiction_pairs or []
    n = len(assertions)
    max_pairs = n * (n - 1) / 2 if n > 1 else 1
    contradiction_rate = len(pairs) / max_pairs if max_pairs > 0 else 0.0
    score = max(0.0, 1.0 - contradiction_rate * 10)

    issues: list[CoherenceIssue] = []
    for i, j in pairs:
        sev = Severity.HIGH if score < 0.5 else Severity.MEDIUM
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
                severity=sev,
                description=f"Contradiction between assertion {i} and {j}",
                evidence=(
                    f"A[{i}]: {assertions[i].text[:80]}"
                    f" vs A[{j}]: {assertions[j].text[:80]}"
                ),
            )
        )
    return DimensionScore(
        dimension=CoherenceDimension.FACTUAL_CONSISTENCY,
        score=score,
        issues=tuple(issues),
        details=f"{len(pairs)} contradictions in {n} assertions.",
    )


def evaluate_temporal_ordering(
    events: Sequence[tuple[str, float]],
) -> DimensionScore:
    """Evaluate whether events are reported in correct temporal order.

    Parameters
    ----------
    events:
        Sequence of (event_description, timestamp) tuples in the order
        they appear in the output.
    """
    if len(events) < 2:
        return DimensionScore(
            dimension=CoherenceDimension.TEMPORAL_ORDERING,
            score=1.0,
            details="Fewer than 2 events; ordering trivially correct.",
        )

    inversions = 0
    total_pairs = 0
    issues: list[CoherenceIssue] = []
    for i in range(len(events) - 1):
        for j in range(i + 1, len(events)):
            total_pairs += 1
            if events[i][1] > events[j][1]:
                inversions += 1
                issues.append(
                    CoherenceIssue(
                        dimension=CoherenceDimension.TEMPORAL_ORDERING,
                        severity=Severity.MEDIUM,
                        description=(
                            f"Temporal inversion: '{events[i][0][:40]}' "
                            f"(t={events[i][1]}) before "
                            f"'{events[j][0][:40]}' (t={events[j][1]})"
                        ),
                    )
                )
    score = 1.0 - (inversions / total_pairs) if total_pairs > 0 else 1.0
    return DimensionScore(
        dimension=CoherenceDimension.TEMPORAL_ORDERING,
        score=max(0.0, score),
        issues=tuple(issues),
        details=f"{inversions} inversions in {total_pairs} pairs.",
    )


def evaluate_knowledge_update_fidelity(
    updates: Sequence[dict[str, Any]],
) -> DimensionScore:
    """Evaluate whether knowledge updates are correctly integrated.

    Parameters
    ----------
    updates:
        Each dict has keys:
        - ``old_claim``: the original claim text
        - ``new_claim``: the corrected claim text
        - ``integrated``: bool — was the correction reflected in output?
    """
    if not updates:
        return DimensionScore(
            dimension=CoherenceDimension.KNOWLEDGE_UPDATE_FIDELITY,
            score=1.0,
            details="No knowledge updates to evaluate.",
        )

    integrated = sum(1 for u in updates if u.get("integrated", False))
    score = integrated / len(updates)
    issues: list[CoherenceIssue] = []
    for u in updates:
        if not u.get("integrated", False):
            issues.append(
                CoherenceIssue(
                    dimension=CoherenceDimension.KNOWLEDGE_UPDATE_FIDELITY,
                    severity=Severity.HIGH,
                    description="Knowledge update not integrated",
                    evidence=(
                        f"Old: {str(u.get('old_claim', ''))[:60]}"
                        f" → New: {str(u.get('new_claim', ''))[:60]}"
                    ),
                )
            )
    return DimensionScore(
        dimension=CoherenceDimension.KNOWLEDGE_UPDATE_FIDELITY,
        score=score,
        issues=tuple(issues),
        details=f"{integrated}/{len(updates)} updates integrated.",
    )


def evaluate_cross_session_reasoning(
    session_conclusions: Sequence[dict[str, Any]],
) -> DimensionScore:
    """Evaluate reasoning consistency across session boundaries.

    Parameters
    ----------
    session_conclusions:
        Each dict has keys:
        - ``session_id``: str
        - ``conclusion``: str
        - ``consistent_with_prior``: bool
    """
    if not session_conclusions:
        return DimensionScore(
            dimension=CoherenceDimension.CROSS_SESSION_REASONING,
            score=1.0,
            details="No cross-session conclusions to evaluate.",
        )

    consistent = sum(
        1 for c in session_conclusions if c.get("consistent_with_prior", True)
    )
    score = consistent / len(session_conclusions)
    issues: list[CoherenceIssue] = []
    for c in session_conclusions:
        if not c.get("consistent_with_prior", True):
            issues.append(
                CoherenceIssue(
                    dimension=CoherenceDimension.CROSS_SESSION_REASONING,
                    severity=Severity.HIGH,
                    description="Cross-session reasoning inconsistency",
                    evidence=(
                        f"Session {c.get('session_id', '?')}: "
                        f"{str(c.get('conclusion', ''))[:80]}"
                    ),
                    session_id=str(c.get("session_id", "")),
                )
            )
    return DimensionScore(
        dimension=CoherenceDimension.CROSS_SESSION_REASONING,
        score=score,
        issues=tuple(issues),
        details=f"{consistent}/{len(session_conclusions)} conclusions consistent.",
    )


def evaluate_contradiction_detection(
    known_contradictions: int,
    detected_contradictions: int,
    false_positives: int = 0,
) -> DimensionScore:
    """Evaluate the system's ability to detect contradictions.

    Parameters
    ----------
    known_contradictions:
        Ground-truth number of contradictions.
    detected_contradictions:
        Number correctly detected by the system.
    false_positives:
        Spurious contradiction flags.
    """
    if known_contradictions == 0 and false_positives == 0:
        return DimensionScore(
            dimension=CoherenceDimension.CONTRADICTION_DETECTION,
            score=1.0,
            details="No contradictions to detect.",
        )

    recall = (
        detected_contradictions / known_contradictions
        if known_contradictions > 0
        else 1.0
    )
    total_detections = detected_contradictions + false_positives
    precision = (
        detected_contradictions / total_detections if total_detections > 0 else 0.0
    )
    score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    issues: list[CoherenceIssue] = []
    missed = known_contradictions - detected_contradictions
    if missed > 0:
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.CONTRADICTION_DETECTION,
                severity=Severity.HIGH,
                description=f"{missed} contradictions missed",
            )
        )
    if false_positives > 0:
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.CONTRADICTION_DETECTION,
                severity=Severity.MEDIUM,
                description=f"{false_positives} false positive contradiction flags",
            )
        )

    return DimensionScore(
        dimension=CoherenceDimension.CONTRADICTION_DETECTION,
        score=score,
        issues=tuple(issues),
        details=f"P={precision:.2f} R={recall:.2f} F1={score:.2f}",
    )


def evaluate_coherence_degradation(
    scores_over_scale: Sequence[float],
) -> DimensionScore:
    """Evaluate how coherence degrades as corpus/context grows.

    Parameters
    ----------
    scores_over_scale:
        Quality scores measured at increasing scale points
        (e.g., after 10, 50, 100, 500 documents).
    """
    if len(scores_over_scale) < 2:
        return DimensionScore(
            dimension=CoherenceDimension.COHERENCE_DEGRADATION,
            score=1.0,
            details="Insufficient data points for degradation curve.",
        )

    scores = list(scores_over_scale)
    first = scores[0]
    last = scores[-1]
    drop = first - last if first > 0 else 0.0
    relative_drop = drop / first if first > 0 else 0.0

    # Penalise large drops: 10% drop → score 0.9, 50% → 0.5
    score = max(0.0, 1.0 - relative_drop)

    issues: list[CoherenceIssue] = []
    if relative_drop > 0.3:
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.COHERENCE_DEGRADATION,
                severity=Severity.HIGH,
                description=f"Coherence dropped {relative_drop:.0%} over scale",
            )
        )
    elif relative_drop > 0.1:
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.COHERENCE_DEGRADATION,
                severity=Severity.MEDIUM,
                description=f"Coherence dropped {relative_drop:.0%} over scale",
            )
        )

    return DimensionScore(
        dimension=CoherenceDimension.COHERENCE_DEGRADATION,
        score=score,
        issues=tuple(issues),
        details=f"From {first:.3f} to {last:.3f} ({relative_drop:.1%} drop).",
    )


def evaluate_memory_pressure_resilience(
    baseline_score: float,
    constrained_score: float,
    memory_limit_fraction: float = 0.5,
) -> DimensionScore:
    """Evaluate quality retention under memory pressure.

    Parameters
    ----------
    baseline_score:
        Quality score with full memory budget.
    constrained_score:
        Quality score with reduced memory (e.g., 50% of baseline).
    memory_limit_fraction:
        Fraction of baseline memory used for constrained test.
    """
    if baseline_score <= 0:
        return DimensionScore(
            dimension=CoherenceDimension.MEMORY_PRESSURE_RESILIENCE,
            score=0.0,
            details="Baseline score is zero; cannot evaluate resilience.",
        )

    retention = constrained_score / baseline_score
    score = min(1.0, max(0.0, retention))

    issues: list[CoherenceIssue] = []
    if retention < 0.7:
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.MEMORY_PRESSURE_RESILIENCE,
                severity=Severity.HIGH,
                description=(
                    f"Quality dropped to {retention:.0%} at "
                    f"{memory_limit_fraction:.0%} memory"
                ),
            )
        )
    elif retention < 0.9:
        issues.append(
            CoherenceIssue(
                dimension=CoherenceDimension.MEMORY_PRESSURE_RESILIENCE,
                severity=Severity.MEDIUM,
                description=(
                    f"Quality dropped to {retention:.0%} at "
                    f"{memory_limit_fraction:.0%} memory"
                ),
            )
        )

    return DimensionScore(
        dimension=CoherenceDimension.MEMORY_PRESSURE_RESILIENCE,
        score=score,
        issues=tuple(issues),
        details=(
            f"Baseline {baseline_score:.3f} → "
            f"Constrained {constrained_score:.3f} "
            f"({retention:.1%} retention at {memory_limit_fraction:.0%} memory)."
        ),
    )


# ---------------------------------------------------------------------------
# Coherence evaluator
# ---------------------------------------------------------------------------


class CoherenceEvaluator:
    """Orchestrate 7-dimension coherence evaluation.

    Parameters
    ----------
    weights:
        Custom dimension weights (default: ``DEFAULT_WEIGHTS``).
    """

    def __init__(
        self,
        weights: dict[CoherenceDimension, float] | None = None,
    ) -> None:
        self._weights = weights or dict(DEFAULT_WEIGHTS)

    def evaluate(
        self,
        *,
        assertions: Sequence[Assertion] | None = None,
        contradiction_pairs: Sequence[tuple[int, int]] | None = None,
        events: Sequence[tuple[str, float]] | None = None,
        knowledge_updates: Sequence[dict[str, Any]] | None = None,
        session_conclusions: Sequence[dict[str, Any]] | None = None,
        known_contradictions: int = 0,
        detected_contradictions: int = 0,
        false_positive_contradictions: int = 0,
        scores_over_scale: Sequence[float] | None = None,
        baseline_score: float = 0.0,
        constrained_score: float = 0.0,
        memory_limit_fraction: float = 0.5,
    ) -> CoherenceReport:
        """Run all seven dimension evaluations and return a report."""
        report = CoherenceReport(weights=self._weights)

        report.dimension_scores[CoherenceDimension.FACTUAL_CONSISTENCY] = (
            evaluate_factual_consistency(assertions or [], contradiction_pairs)
        )
        report.dimension_scores[CoherenceDimension.TEMPORAL_ORDERING] = (
            evaluate_temporal_ordering(events or [])
        )
        report.dimension_scores[CoherenceDimension.KNOWLEDGE_UPDATE_FIDELITY] = (
            evaluate_knowledge_update_fidelity(knowledge_updates or [])
        )
        report.dimension_scores[CoherenceDimension.CROSS_SESSION_REASONING] = (
            evaluate_cross_session_reasoning(session_conclusions or [])
        )
        report.dimension_scores[CoherenceDimension.CONTRADICTION_DETECTION] = (
            evaluate_contradiction_detection(
                known_contradictions,
                detected_contradictions,
                false_positive_contradictions,
            )
        )
        report.dimension_scores[CoherenceDimension.COHERENCE_DEGRADATION] = (
            evaluate_coherence_degradation(scores_over_scale or [])
        )
        report.dimension_scores[CoherenceDimension.MEMORY_PRESSURE_RESILIENCE] = (
            evaluate_memory_pressure_resilience(
                baseline_score, constrained_score, memory_limit_fraction
            )
        )

        return report
