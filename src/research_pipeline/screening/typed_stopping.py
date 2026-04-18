"""Query-typed retrieval stopping profiles.

Extends the adaptive_stopping module with per-query-type stopping
profiles that optimize cost vs. completeness trade-offs.  Based on
the HingeMem (WWW '26) pattern achieving 68% lower retrieval cost.

Key insight: different query *intents* need different stopping aggressiveness.

- **Recall queries** (surveys, literature reviews): permissive — keep
  fetching until marginal gain drops to near-zero.
- **Precision queries** (specific facts, definitions): aggressive — stop
  as soon as top results are confident.
- **Judgment queries** (comparisons, evaluations): moderate — stop when
  top-1 is stable but verify with a few more batches.
- **Exploratory queries**: very permissive — encourage breadth.

Each profile defines thresholds, patience, and cost multipliers so
the pipeline can self-select the right stopping behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended query types
# ---------------------------------------------------------------------------


class ExtendedQueryType(StrEnum):
    """Fine-grained query intent classification."""

    RECALL = "recall"  # broad recall (surveys, reviews)
    PRECISION = "precision"  # targeted precision (specific facts)
    JUDGMENT = "judgment"  # evaluative (comparisons)
    EXPLORATORY = "exploratory"  # open-ended discovery
    VERIFICATION = "verification"  # fact-checking a claim


# ---------------------------------------------------------------------------
# Stopping profile
# ---------------------------------------------------------------------------


@dataclass
class StoppingProfile:
    """Per-query-type stopping configuration.

    Attributes:
        query_type: Which query type this profile applies to.
        knee_threshold: Marginal gain ratio below which to stop (recall).
        saturation_threshold: Fraction of top-k above quality bar (precision).
        stability_window: Consecutive stable batches before stopping (judgment).
        min_batches: Minimum batches before any stopping check.
        max_batches: Hard budget limit.
        patience: Extra batches to wait after initial stop signal.
        cost_weight: Relative cost sensitivity (higher = stop sooner).
        quality_floor: Minimum score to consider a result "good".
        description: Human-readable description.
    """

    query_type: ExtendedQueryType
    knee_threshold: float = 0.05
    saturation_threshold: float = 0.80
    stability_window: int = 3
    min_batches: int = 2
    max_batches: int = 20
    patience: int = 1
    cost_weight: float = 1.0
    quality_floor: float = 0.5
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "knee_threshold": self.knee_threshold,
            "saturation_threshold": self.saturation_threshold,
            "stability_window": self.stability_window,
            "min_batches": self.min_batches,
            "max_batches": self.max_batches,
            "patience": self.patience,
            "cost_weight": round(self.cost_weight, 4),
            "quality_floor": self.quality_floor,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Default profiles
# ---------------------------------------------------------------------------

RECALL_PROFILE = StoppingProfile(
    query_type=ExtendedQueryType.RECALL,
    knee_threshold=0.02,
    saturation_threshold=0.60,
    stability_window=5,
    min_batches=3,
    max_batches=30,
    patience=2,
    cost_weight=0.3,
    quality_floor=0.3,
    description="Permissive: maximise completeness, tolerate low precision",
)

PRECISION_PROFILE = StoppingProfile(
    query_type=ExtendedQueryType.PRECISION,
    knee_threshold=0.15,
    saturation_threshold=0.90,
    stability_window=2,
    min_batches=1,
    max_batches=10,
    patience=0,
    cost_weight=2.0,
    quality_floor=0.7,
    description="Aggressive: stop early once top results are confident",
)

JUDGMENT_PROFILE = StoppingProfile(
    query_type=ExtendedQueryType.JUDGMENT,
    knee_threshold=0.08,
    saturation_threshold=0.80,
    stability_window=3,
    min_batches=2,
    max_batches=15,
    patience=1,
    cost_weight=1.0,
    quality_floor=0.5,
    description="Moderate: verify top-1 stability with a few extra batches",
)

EXPLORATORY_PROFILE = StoppingProfile(
    query_type=ExtendedQueryType.EXPLORATORY,
    knee_threshold=0.01,
    saturation_threshold=0.50,
    stability_window=4,
    min_batches=3,
    max_batches=25,
    patience=3,
    cost_weight=0.2,
    quality_floor=0.2,
    description="Very permissive: encourage breadth and serendipity",
)

VERIFICATION_PROFILE = StoppingProfile(
    query_type=ExtendedQueryType.VERIFICATION,
    knee_threshold=0.20,
    saturation_threshold=0.95,
    stability_window=2,
    min_batches=1,
    max_batches=8,
    patience=0,
    cost_weight=3.0,
    quality_floor=0.8,
    description="Most aggressive: find the answer or confirm absence quickly",
)

DEFAULT_PROFILES: dict[ExtendedQueryType, StoppingProfile] = {
    ExtendedQueryType.RECALL: RECALL_PROFILE,
    ExtendedQueryType.PRECISION: PRECISION_PROFILE,
    ExtendedQueryType.JUDGMENT: JUDGMENT_PROFILE,
    ExtendedQueryType.EXPLORATORY: EXPLORATORY_PROFILE,
    ExtendedQueryType.VERIFICATION: VERIFICATION_PROFILE,
}


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

# Keyword patterns for heuristic classification
_RECALL_KEYWORDS = frozenset(
    [
        "survey",
        "review",
        "comprehensive",
        "all",
        "every",
        "literature",
        "overview",
        "systematic",
        "broad",
        "exhaustive",
        "landscape",
    ]
)
_PRECISION_KEYWORDS = frozenset(
    [
        "specific",
        "exact",
        "define",
        "what is",
        "how does",
        "method for",
        "technique",
        "algorithm",
        "formula",
        "equation",
    ]
)
_JUDGMENT_KEYWORDS = frozenset(
    [
        "compare",
        "versus",
        "better",
        "which",
        "evaluate",
        "assess",
        "pros and cons",
        "trade-off",
        "advantage",
        "benchmark",
    ]
)
_EXPLORATORY_KEYWORDS = frozenset(
    [
        "explore",
        "discover",
        "emerging",
        "novel",
        "new",
        "trend",
        "future",
        "potential",
        "opportunity",
        "frontier",
    ]
)
_VERIFICATION_KEYWORDS = frozenset(
    [
        "verify",
        "confirm",
        "check",
        "validate",
        "true",
        "false",
        "correct",
        "accurate",
        "fact",
        "claim",
    ]
)

_KEYWORD_MAP: list[tuple[frozenset[str], ExtendedQueryType]] = [
    (_VERIFICATION_KEYWORDS, ExtendedQueryType.VERIFICATION),
    (_PRECISION_KEYWORDS, ExtendedQueryType.PRECISION),
    (_JUDGMENT_KEYWORDS, ExtendedQueryType.JUDGMENT),
    (_RECALL_KEYWORDS, ExtendedQueryType.RECALL),
    (_EXPLORATORY_KEYWORDS, ExtendedQueryType.EXPLORATORY),
]


def classify_query_type(query: str) -> ExtendedQueryType:
    """Classify a query string into an extended query type.

    Uses keyword matching with priority ordering (verification first,
    exploratory last).  Falls back to RECALL for ambiguous queries.

    Args:
        query: The search query string.

    Returns:
        ExtendedQueryType classification.
    """
    if not query:
        return ExtendedQueryType.RECALL

    lower = query.lower()
    scores: dict[ExtendedQueryType, int] = dict.fromkeys(ExtendedQueryType, 0)

    for keywords, qtype in _KEYWORD_MAP:
        for kw in keywords:
            if kw in lower:
                scores[qtype] += 1

    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_type] == 0:
        return ExtendedQueryType.RECALL

    logger.debug(
        "Query classified as %s (score=%d): %s",
        best_type.value,
        scores[best_type],
        query,
    )
    return best_type


def get_profile(
    query_type: ExtendedQueryType,
    custom_profiles: dict[ExtendedQueryType, StoppingProfile] | None = None,
) -> StoppingProfile:
    """Get the stopping profile for a query type.

    Args:
        query_type: The classified query type.
        custom_profiles: Optional overrides for default profiles.

    Returns:
        The matching StoppingProfile.
    """
    profiles = custom_profiles or DEFAULT_PROFILES
    return profiles.get(query_type, RECALL_PROFILE)


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


@dataclass
class CostEstimate:
    """Estimated retrieval cost for a query type."""

    query_type: ExtendedQueryType
    estimated_batches: int
    cost_multiplier: float
    relative_cost: float  # vs recall baseline
    estimated_savings: float  # fraction of cost saved vs naive

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "estimated_batches": self.estimated_batches,
            "cost_multiplier": round(self.cost_multiplier, 4),
            "relative_cost": round(self.relative_cost, 4),
            "estimated_savings": round(self.estimated_savings, 4),
        }


def estimate_cost(
    query: str,
    custom_profiles: dict[ExtendedQueryType, StoppingProfile] | None = None,
) -> CostEstimate:
    """Estimate retrieval cost for a query.

    Compares the expected batches for the detected query type against
    the recall (baseline) profile to compute relative savings.

    Args:
        query: The search query.
        custom_profiles: Optional profile overrides.

    Returns:
        CostEstimate with savings projection.
    """
    qtype = classify_query_type(query)
    profile = get_profile(qtype, custom_profiles)
    baseline = get_profile(ExtendedQueryType.RECALL, custom_profiles)

    # Estimated batches is midpoint between min and max, weighted by cost_weight
    est = profile.min_batches + (
        (profile.max_batches - profile.min_batches) / (1 + profile.cost_weight)
    )
    baseline_est = baseline.min_batches + (
        (baseline.max_batches - baseline.min_batches) / (1 + baseline.cost_weight)
    )

    relative = est / baseline_est if baseline_est > 0 else 1.0
    savings = max(0.0, 1.0 - relative)

    return CostEstimate(
        query_type=qtype,
        estimated_batches=round(est),
        cost_multiplier=profile.cost_weight,
        relative_cost=relative,
        estimated_savings=savings,
    )


# ---------------------------------------------------------------------------
# Typed stopping evaluator
# ---------------------------------------------------------------------------


@dataclass
class TypedStoppingResult:
    """Result of a typed stopping evaluation."""

    should_stop: bool
    query_type: ExtendedQueryType
    profile: StoppingProfile
    reason: str = ""
    batches_processed: int = 0
    cost_so_far: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "query_type": self.query_type.value,
            "reason": self.reason,
            "batches_processed": self.batches_processed,
            "cost_so_far": round(self.cost_so_far, 4),
            "profile": self.profile.to_dict(),
        }


class TypedStoppingEvaluator:
    """Evaluate stopping criteria using query-typed profiles.

    Usage::

        evaluator = TypedStoppingEvaluator(query="survey of knowledge graphs")
        evaluator.add_batch([0.9, 0.8, 0.7, 0.6])
        result = evaluator.evaluate()
        if result.should_stop:
            print(f"Stop: {result.reason}")
    """

    def __init__(
        self,
        query: str = "",
        query_type: ExtendedQueryType | None = None,
        custom_profiles: dict[ExtendedQueryType, StoppingProfile] | None = None,
    ) -> None:
        self._query = query
        self._query_type = query_type or classify_query_type(query)
        self._profile = get_profile(self._query_type, custom_profiles)
        self._batches: list[list[float]] = []
        self._top1_history: list[float] = []

    @property
    def query_type(self) -> ExtendedQueryType:
        return self._query_type

    @property
    def profile(self) -> StoppingProfile:
        return self._profile

    @property
    def batches_processed(self) -> int:
        return len(self._batches)

    def add_batch(self, scores: list[float]) -> None:
        """Add a batch of relevance scores."""
        self._batches.append(list(scores))
        if scores:
            self._top1_history.append(max(scores))

    def evaluate(self) -> TypedStoppingResult:
        """Evaluate whether retrieval should stop.

        Checks in order:
        1. Budget exhausted (max_batches).
        2. Below min_batches → continue.
        3. Saturation check (precision-style).
        4. Knee detection (recall-style).
        5. Top-1 stability (judgment-style).
        """
        prof = self._profile
        n = len(self._batches)

        # Budget check
        if n >= prof.max_batches:
            return TypedStoppingResult(
                should_stop=True,
                query_type=self._query_type,
                profile=prof,
                reason="budget_exhausted",
                batches_processed=n,
                cost_so_far=n * prof.cost_weight,
            )

        # Min batches check
        if n < prof.min_batches:
            return TypedStoppingResult(
                should_stop=False,
                query_type=self._query_type,
                profile=prof,
                reason="below_min_batches",
                batches_processed=n,
                cost_so_far=n * prof.cost_weight,
            )

        # Saturation check: fraction of all scores above quality floor
        all_scores = [s for batch in self._batches for s in batch]
        if all_scores:
            above_floor = sum(1 for s in all_scores if s >= prof.quality_floor)
            sat_ratio = above_floor / len(all_scores)
            if sat_ratio >= prof.saturation_threshold:
                return TypedStoppingResult(
                    should_stop=True,
                    query_type=self._query_type,
                    profile=prof,
                    reason=(
                        f"saturation_reached"
                        f" ({sat_ratio:.2f} >= {prof.saturation_threshold})"
                    ),
                    batches_processed=n,
                    cost_so_far=n * prof.cost_weight,
                )

        # Knee detection: check last batch marginal gain
        if len(self._batches) >= 2:
            prev_mean = (
                sum(self._batches[-2]) / len(self._batches[-2])
                if self._batches[-2]
                else 0.0
            )
            curr_mean = (
                sum(self._batches[-1]) / len(self._batches[-1])
                if self._batches[-1]
                else 0.0
            )
            gain = abs(curr_mean - prev_mean)
            if gain < prof.knee_threshold:
                return TypedStoppingResult(
                    should_stop=True,
                    query_type=self._query_type,
                    profile=prof,
                    reason=f"knee_detected (gain={gain:.4f} < {prof.knee_threshold})",
                    batches_processed=n,
                    cost_so_far=n * prof.cost_weight,
                )

        # Top-1 stability: check if top score is stable
        if len(self._top1_history) >= prof.stability_window:
            window = self._top1_history[-prof.stability_window :]
            if max(window) - min(window) < prof.knee_threshold:
                win_range = max(window) - min(window)
                return TypedStoppingResult(
                    should_stop=True,
                    query_type=self._query_type,
                    profile=prof,
                    reason=f"top1_stable (window_range={win_range:.4f})",
                    batches_processed=n,
                    cost_so_far=n * prof.cost_weight,
                )

        return TypedStoppingResult(
            should_stop=False,
            query_type=self._query_type,
            profile=prof,
            reason="continue",
            batches_processed=n,
            cost_so_far=n * prof.cost_weight,
        )

    def summary(self) -> dict[str, Any]:
        """Get evaluator summary."""
        return {
            "query": self._query,
            "query_type": self._query_type.value,
            "batches_processed": len(self._batches),
            "total_scores": sum(len(b) for b in self._batches),
            "profile": self._profile.to_dict(),
            "current_evaluation": self.evaluate().to_dict(),
        }
