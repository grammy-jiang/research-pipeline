"""Plan-revision evaluator (A3-3 from the research report).

The THINK → EXECUTE → REFLECT (TER) loop in :mod:`pipeline.ter_loop`
iteratively revises queries / plans based on discovered gaps. The research
report identifies a missing component: an *evaluator* that scores each
revision so the loop can terminate when revisions plateau (rather than
burning budget on low-signal iterations).

This module implements that evaluator:

- :class:`PlanRevision` records one revision step (before/after query).
- :func:`score_revision` assigns a :class:`PlanRevisionScore` capturing
  coverage growth, term preservation, semantic drift, and a composite
  quality score.
- :class:`PlanRevisionTracker` holds a sequence of revisions and exposes a
  ``should_stop`` decision once the composite scores plateau.

No LLM call is required — the evaluator uses deterministic string metrics so
it can run inside the TER loop cheaply and be unit-tested without mocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanRevisionScore:
    """Score for a single revision step.

    Attributes:
        coverage_delta: New tokens introduced by the revision, as a
            fraction of the union of both queries. Higher → more
            exploration.
        preservation: Fraction of the original query's tokens preserved.
            Low preservation indicates topic drift.
        drift: Jaccard distance between original and revised token sets
            (``1 - jaccard_similarity``). Complementary to preservation.
        composite: Combined score in ``[0, 1]``; ~``0.6 * coverage_delta
            + 0.4 * preservation`` — rewards exploration that retains the
            core topic.
    """

    coverage_delta: float
    preservation: float
    drift: float
    composite: float


@dataclass
class PlanRevision:
    """One revision step within the TER loop."""

    iteration: int
    original_query: str
    revised_query: str
    score: PlanRevisionScore | None = None


def _tokens(query: str) -> set[str]:
    return {t for t in query.lower().split() if len(t) > 2}


def score_revision(original: str, revised: str) -> PlanRevisionScore:
    """Score a single revision.

    Empty revised query is treated as a pathological no-op with composite 0.
    Identical queries get preservation=1, coverage_delta=0, composite=0.4.
    """
    orig = _tokens(original)
    new = _tokens(revised)
    if not new:
        return PlanRevisionScore(0.0, 0.0, 1.0, 0.0)

    union = orig | new
    added = new - orig
    kept = orig & new

    coverage_delta = len(added) / len(union) if union else 0.0
    preservation = len(kept) / len(orig) if orig else 1.0
    jaccard = len(orig & new) / len(orig | new) if orig or new else 1.0
    drift = 1.0 - jaccard
    composite = round(0.6 * coverage_delta + 0.4 * preservation, 6)
    return PlanRevisionScore(
        coverage_delta=round(coverage_delta, 6),
        preservation=round(preservation, 6),
        drift=round(drift, 6),
        composite=composite,
    )


@dataclass
class PlanRevisionTracker:
    """Tracks a trajectory of plan revisions for convergence detection."""

    plateau_tolerance: float = 0.02
    plateau_window: int = 2
    min_iterations: int = 2
    revisions: list[PlanRevision] = field(default_factory=list)

    def record(self, original: str, revised: str) -> PlanRevision:
        """Score and append one revision; returns the stored object."""
        iteration = len(self.revisions) + 1
        score = score_revision(original, revised)
        rev = PlanRevision(
            iteration=iteration,
            original_query=original,
            revised_query=revised,
            score=score,
        )
        self.revisions.append(rev)
        logger.debug(
            "Plan revision %d: composite=%.3f (cov=%.3f, pres=%.3f)",
            iteration,
            score.composite,
            score.coverage_delta,
            score.preservation,
        )
        return rev

    def composite_series(self) -> list[float]:
        return [r.score.composite for r in self.revisions if r.score is not None]

    def should_stop(self) -> bool:
        """Plateau test: last ``plateau_window+1`` scores within tolerance."""
        series = self.composite_series()
        if len(series) < max(self.min_iterations, self.plateau_window + 1):
            return False
        window = series[-(self.plateau_window + 1) :]
        spread = max(window) - min(window)
        return spread <= self.plateau_tolerance

    def best(self) -> PlanRevision | None:
        """Return the revision with the highest composite score, if any."""
        scored = [r for r in self.revisions if r.score is not None]
        if not scored:
            return None
        return max(scored, key=lambda r: r.score.composite)  # type: ignore[union-attr]
