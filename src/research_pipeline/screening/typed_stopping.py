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

Keywords: Q2D, Query-to-Document augmentation.
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
    AUTO = "auto"  # auto-detect sentinel (shared with adaptive_stopping, #111)


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
        "best",
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

    best_score = max(scores.values())
    if best_score == 0:
        return ExtendedQueryType.RECALL

    # Break ties by _KEYWORD_MAP priority (verification first, exploratory last)
    # as documented — NOT by enum/dict iteration order, which put RECALL first
    # and silently won every tie (#111).
    best_type = next(
        qtype for _keywords, qtype in _KEYWORD_MAP if scores[qtype] == best_score
    )

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
    reclassified: bool = False
    reclassified_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "query_type": self.query_type.value,
            "reason": self.reason,
            "batches_processed": self.batches_processed,
            "cost_so_far": round(self.cost_so_far, 4),
            "profile": self.profile.to_dict(),
            "reclassified": self.reclassified,
            "reclassified_from": self.reclassified_from,
        }


# Minimal stopword set for the re-classification enrichment signal — just enough
# to keep generic filler out of the high-frequency terms fed back to the
# classifier. Deliberately small: the classifier still sees the original query.
_ENRICHMENT_STOPWORDS: frozenset[str] = frozenset(
    {
        "with",
        "from",
        "this",
        "that",
        "using",
        "based",
        "into",
        "their",
        "these",
        "those",
        "while",
        "about",
        "across",
        "between",
        "model",
        "models",
        "method",
        "methods",
        "approach",
        "approaches",
        "paper",
        "study",
        "results",
    }
)


def _enrichment_terms(texts: list[str], k: int = 12) -> list[str]:
    """Top-*k* high-frequency content terms from accumulated batch texts.

    Enriches the re-classification signal (#111): the original query is
    concatenated with these terms so the classifier can revise its type once
    retrieved titles/abstracts reveal the true intent. Deterministic — ties break
    alphabetically — so the same texts always yield the same terms.

    Args:
        texts: Accumulated per-batch document texts (e.g. titles + abstracts).
        k: Maximum number of terms to return.

    Returns:
        Up to *k* lower-cased terms, most frequent first.
    """
    counts: dict[str, int] = {}
    for text in texts:
        for raw in text.lower().split():
            word = raw.strip(".,:;!?()[]{}\"'`")
            if len(word) >= 4 and word not in _ENRICHMENT_STOPWORDS:
                counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:k]]


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
        reclassify_after_n: int = 0,
    ) -> None:
        self._query = query
        self._custom_profiles = custom_profiles
        self._query_type = query_type or classify_query_type(query)
        self._profile = get_profile(self._query_type, custom_profiles)
        self._batches: list[list[float]] = []
        self._top1_history: list[float] = []
        # Opt-in re-classification (#111): once ``reclassify_after_n`` batches have
        # arrived, revise the query type from accumulated retrieved text so an
        # initial topic-only mis-read no longer permanently fixes the stopping
        # regime. 0 disables it (default) → identical to the pre-#111 behaviour.
        self._reclassify_after_n = reclassify_after_n
        self._texts: list[str] = []
        self._reclassified_from: ExtendedQueryType | None = None

    @property
    def query_type(self) -> ExtendedQueryType:
        return self._query_type

    @property
    def profile(self) -> StoppingProfile:
        return self._profile

    @property
    def batches_processed(self) -> int:
        return len(self._batches)

    @property
    def reclassified(self) -> bool:
        """Whether accumulated evidence has revised the initial query type."""
        return self._reclassified_from is not None

    def add_batch(self, scores: list[float], texts: list[str] | None = None) -> None:
        """Add a batch of relevance scores and (optionally) their texts.

        Args:
            scores: Relevance scores for this batch.
            texts: Document texts (e.g. titles + abstracts) for this batch, used
                only to enrich the opt-in re-classification signal. Ignored when
                ``reclassify_after_n`` is 0.
        """
        self._batches.append(list(scores))
        if scores:
            self._top1_history.append(max(scores))
        if texts:
            self._texts.extend(texts)
        self._maybe_reclassify()

    def _maybe_reclassify(self) -> None:
        """Revise the query type from accumulated evidence (opt-in, #111).

        Once at least ``reclassify_after_n`` batches have arrived, re-run the one
        canonical classifier over the query enriched with the highest-frequency
        terms from the retrieved texts. If the type changed, swap in the new
        profile and remember the original. No-op when disabled, when too few
        batches have arrived, or when no texts were supplied.
        """
        if self._reclassify_after_n <= 0:
            return
        if len(self._batches) < self._reclassify_after_n:
            return
        if not self._texts:
            return
        enriched = self._query + " " + " ".join(_enrichment_terms(self._texts))
        new_type = classify_query_type(enriched)
        if new_type in (self._query_type, ExtendedQueryType.AUTO):
            return
        if self._reclassified_from is None:
            self._reclassified_from = self._query_type
        self._query_type = new_type
        self._profile = get_profile(new_type, self._custom_profiles)

    def evaluate(self) -> TypedStoppingResult:
        """Evaluate whether retrieval should stop.

        Thin wrapper over :meth:`_evaluate_core` that stamps the re-classification
        outcome (#111) onto the result, so callers see whether accumulated
        evidence revised the query type — and from which original type.
        """
        result = self._evaluate_core()
        result.reclassified = self._reclassified_from is not None
        result.reclassified_from = (
            self._reclassified_from.value
            if self._reclassified_from is not None
            else None
        )
        return result

    def _evaluate_core(self) -> TypedStoppingResult:
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

        # Knee detection: require the marginal gain to stay below threshold for
        # TWO consecutive batch pairs (≥3 batches) before stopping, so a single
        # small-sample dip does not trigger premature stopping (#123). Computed
        # from the batch history, so evaluate() stays side-effect-free.
        if len(self._batches) >= 3:

            def _mean(batch: list[float]) -> float:
                return sum(batch) / len(batch) if batch else 0.0

            recent_gains = [
                abs(_mean(self._batches[-2]) - _mean(self._batches[-3])),
                abs(_mean(self._batches[-1]) - _mean(self._batches[-2])),
            ]
            if all(gain < prof.knee_threshold for gain in recent_gains):
                return TypedStoppingResult(
                    should_stop=True,
                    query_type=self._query_type,
                    profile=prof,
                    reason=(
                        f"knee_detected (2 consecutive gains < {prof.knee_threshold})"
                    ),
                    batches_processed=n,
                    cost_so_far=n * prof.cost_weight,
                )

        # Top-1 stability: require at least three observations (mirroring the
        # knee floor above) so the aggressive profiles — PRECISION and
        # VERIFICATION set stability_window=2 — cannot stop on a two-sample
        # window, which a single small-sample coincidence could satisfy (#123).
        stability_window = max(prof.stability_window, 3)
        if len(self._top1_history) >= stability_window:
            window = self._top1_history[-stability_window:]
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
