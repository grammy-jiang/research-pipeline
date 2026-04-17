"""Retention regularization for cross-session memory stability.

Detects and penalises abrupt semantic shifts between pipeline sessions
to prevent **memory drift** — the gradual degradation of stored knowledge
quality when new sessions overwrite or contradict established facts.

Implements the MLMF retention-regularisation pattern (FMR 6.9% → 5.1%):

1. **Semantic similarity** between consecutive session summaries
   (cosine distance on token-frequency vectors).
2. **Drift scoring** with configurable thresholds per stage.
3. **Stabilisation recommendations** — when drift exceeds the threshold,
   the module suggests consolidation, rollback, or review.
4. **Drift history** — tracks drift scores over time to detect trends.

References:
    MLMF retention regularization (Deep Research Report §R3).
    Memory Survey: retention regularization FMR 6.9% → 5.1%.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenisation (lightweight, no external deps)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Split *text* into lowercase word tokens."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compute normalised term-frequency vector."""
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {t: c / total for t, c in counts.items()}


def cosine_similarity(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
) -> float:
    """Compute cosine similarity between two sparse TF vectors.

    Returns:
        Value in [0.0, 1.0].  0 = orthogonal, 1 = identical.
    """
    if not vec_a or not vec_b:
        return 0.0

    keys = set(vec_a) | set(vec_b)
    dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def text_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using TF vectors.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    return cosine_similarity(
        _term_freq(_tokenize(text_a)),
        _term_freq(_tokenize(text_b)),
    )


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class DriftSeverity(StrEnum):
    """Categorises the severity of semantic drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StabilisationAction(StrEnum):
    """Recommended actions when drift is detected."""

    NONE = "none"
    REVIEW = "review"
    CONSOLIDATE = "consolidate"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable thresholds for drift severity classification.

    Similarity is in [0, 1]; *lower* similarity means *more* drift.

    Args:
        low: Below this similarity → LOW drift.
        medium: Below this → MEDIUM.
        high: Below this → HIGH.
        critical: Below this → CRITICAL.
    """

    low: float = 0.80
    medium: float = 0.60
    high: float = 0.40
    critical: float = 0.20


DEFAULT_THRESHOLDS = DriftThresholds()


@dataclass
class DriftScore:
    """Result of a single drift measurement."""

    similarity: float
    severity: DriftSeverity
    recommendation: StabilisationAction
    session_a: str
    session_b: str
    stage: str = ""

    @property
    def drift_magnitude(self) -> float:
        """Complementary value: 1 − similarity (higher = more drift)."""
        return 1.0 - self.similarity

    def to_dict(self) -> dict[str, object]:
        """Serialise to plain dict."""
        return {
            "similarity": round(self.similarity, 4),
            "drift_magnitude": round(self.drift_magnitude, 4),
            "severity": self.severity.value,
            "recommendation": self.recommendation.value,
            "session_a": self.session_a,
            "session_b": self.session_b,
            "stage": self.stage,
        }


@dataclass
class DriftTrend:
    """Trend analysis across multiple consecutive drift measurements."""

    scores: list[float]
    direction: str  # "stable", "improving", "degrading"
    average_similarity: float
    min_similarity: float
    max_similarity: float

    def to_dict(self) -> dict[str, object]:
        """Serialise to plain dict."""
        return {
            "direction": self.direction,
            "average_similarity": round(self.average_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "measurement_count": len(self.scores),
        }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def classify_drift(
    similarity: float,
    thresholds: DriftThresholds = DEFAULT_THRESHOLDS,
) -> DriftSeverity:
    """Map a similarity score to a :class:`DriftSeverity`.

    Args:
        similarity: Cosine similarity in [0, 1].
        thresholds: Configurable thresholds.

    Returns:
        The appropriate severity level.
    """
    if similarity >= thresholds.low:
        return DriftSeverity.NONE
    if similarity >= thresholds.medium:
        return DriftSeverity.LOW
    if similarity >= thresholds.high:
        return DriftSeverity.MEDIUM
    if similarity >= thresholds.critical:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


def recommend_action(severity: DriftSeverity) -> StabilisationAction:
    """Map severity to a recommended stabilisation action.

    Args:
        severity: Drift severity level.

    Returns:
        Recommended action.
    """
    mapping = {
        DriftSeverity.NONE: StabilisationAction.NONE,
        DriftSeverity.LOW: StabilisationAction.NONE,
        DriftSeverity.MEDIUM: StabilisationAction.REVIEW,
        DriftSeverity.HIGH: StabilisationAction.CONSOLIDATE,
        DriftSeverity.CRITICAL: StabilisationAction.ROLLBACK,
    }
    return mapping[severity]


def measure_drift(
    text_a: str,
    text_b: str,
    *,
    session_a: str = "previous",
    session_b: str = "current",
    stage: str = "",
    thresholds: DriftThresholds = DEFAULT_THRESHOLDS,
) -> DriftScore:
    """Measure semantic drift between two texts.

    Args:
        text_a: Text from the earlier session.
        text_b: Text from the later session.
        session_a: Identifier for the earlier session.
        session_b: Identifier for the later session.
        stage: Pipeline stage name (for context).
        thresholds: Drift severity thresholds.

    Returns:
        :class:`DriftScore` with similarity, severity, and recommendation.
    """
    sim = text_similarity(text_a, text_b)
    severity = classify_drift(sim, thresholds)
    action = recommend_action(severity)
    return DriftScore(
        similarity=sim,
        severity=severity,
        recommendation=action,
        session_a=session_a,
        session_b=session_b,
        stage=stage,
    )


class RetentionRegularizer:
    """Tracks and regularises semantic drift across sessions.

    Maintains a history of drift measurements and computes trends.

    Args:
        thresholds: Drift severity thresholds.
        history_limit: Maximum number of drift scores to retain.
    """

    def __init__(
        self,
        thresholds: DriftThresholds = DEFAULT_THRESHOLDS,
        *,
        history_limit: int = 100,
    ) -> None:
        self._thresholds = thresholds
        self._history: list[DriftScore] = []
        self._history_limit = history_limit

    @property
    def history(self) -> list[DriftScore]:
        """All recorded drift scores in order."""
        return list(self._history)

    @property
    def thresholds(self) -> DriftThresholds:
        """Current thresholds."""
        return self._thresholds

    def record(
        self,
        text_a: str,
        text_b: str,
        *,
        session_a: str = "previous",
        session_b: str = "current",
        stage: str = "",
    ) -> DriftScore:
        """Measure drift and append to history.

        Args:
            text_a: Earlier session text.
            text_b: Later session text.
            session_a: Identifier for earlier session.
            session_b: Identifier for later session.
            stage: Pipeline stage context.

        Returns:
            The computed :class:`DriftScore`.
        """
        score = measure_drift(
            text_a,
            text_b,
            session_a=session_a,
            session_b=session_b,
            stage=stage,
            thresholds=self._thresholds,
        )
        self._history.append(score)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

        if score.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            logger.warning(
                "Semantic drift %s between %s → %s (sim=%.3f, stage=%s)",
                score.severity.value,
                session_a,
                session_b,
                score.similarity,
                stage,
            )

        return score

    def compute_trend(self, window: int = 0) -> DriftTrend | None:
        """Analyse the drift trend over recent history.

        Args:
            window: Number of recent scores to analyse.
                0 means all history.

        Returns:
            :class:`DriftTrend` or ``None`` if no history.
        """
        if not self._history:
            return None

        scores = self._history[-window:] if window > 0 else self._history
        sims = [s.similarity for s in scores]
        avg = sum(sims) / len(sims)

        if len(sims) < 2:
            direction = "stable"
        else:
            # Simple linear trend: compare first half vs second half
            mid = len(sims) // 2
            first_avg = sum(sims[:mid]) / mid if mid > 0 else avg
            second_avg = sum(sims[mid:]) / (len(sims) - mid) if mid < len(sims) else avg
            delta = second_avg - first_avg
            if delta > 0.05:
                direction = "improving"
            elif delta < -0.05:
                direction = "degrading"
            else:
                direction = "stable"

        return DriftTrend(
            scores=sims,
            direction=direction,
            average_similarity=avg,
            min_similarity=min(sims),
            max_similarity=max(sims),
        )

    def get_stage_drift(self, stage: str) -> list[DriftScore]:
        """Filter history by pipeline stage.

        Args:
            stage: Stage name to filter.

        Returns:
            Scores for that stage.
        """
        return [s for s in self._history if s.stage == stage]

    def needs_stabilisation(self) -> bool:
        """Whether any recent drift score recommends non-NONE action.

        Looks at the last 3 scores (or all if fewer).
        """
        recent = self._history[-3:] if len(self._history) >= 3 else self._history
        return any(s.recommendation != StabilisationAction.NONE for s in recent)

    def compute_regularization_penalty(
        self,
        text_a: str,
        text_b: str,
        *,
        alpha: float = 1.0,
    ) -> float:
        """Compute a regularisation penalty for a proposed memory update.

        The penalty is ``alpha * (1 - similarity)`` — use this to weight
        new entries lower when they diverge sharply from existing memory.

        Args:
            text_a: Existing memory content.
            text_b: Proposed new content.
            alpha: Scaling factor (higher = stronger regularisation).

        Returns:
            Penalty in [0, alpha].  0 = no drift.
        """
        sim = text_similarity(text_a, text_b)
        return alpha * (1.0 - sim)

    def regularize_score(
        self,
        base_score: float,
        text_a: str,
        text_b: str,
        *,
        alpha: float = 0.3,
    ) -> float:
        """Apply retention regularisation to a relevance score.

        Reduces the score proportionally to the semantic drift, so that
        sharply divergent updates are weighted down.

        Args:
            base_score: Original relevance/quality score.
            text_a: Existing memory content.
            text_b: Proposed new content.
            alpha: Regularisation strength (0 = none, 1 = full).

        Returns:
            Adjusted score: ``base_score * (1 - alpha * drift)``.
        """
        penalty = self.compute_regularization_penalty(text_a, text_b, alpha=alpha)
        return max(0.0, base_score * (1.0 - penalty))

    def summary(self) -> dict[str, object]:
        """Summary statistics for the regulariser state."""
        trend = self.compute_trend()
        return {
            "total_measurements": len(self._history),
            "needs_stabilisation": self.needs_stabilisation(),
            "trend": trend.to_dict() if trend else None,
            "thresholds": {
                "low": self._thresholds.low,
                "medium": self._thresholds.medium,
                "high": self._thresholds.high,
                "critical": self._thresholds.critical,
            },
        }
