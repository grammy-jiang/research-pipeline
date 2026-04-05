"""Composite quality scoring.

Aggregates citation impact, venue reputation, author credibility,
and recency into a single quality score per paper.
"""

import logging
import math
from datetime import UTC, datetime

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.quality import QualityScore
from research_pipeline.quality.author_metrics import author_credibility
from research_pipeline.quality.citation_metrics import citation_impact
from research_pipeline.quality.venue_scoring import venue_score

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "citation_weight": 0.35,
    "venue_weight": 0.25,
    "author_weight": 0.25,
    "recency_weight": 0.15,
}


def _recency_score(
    published: datetime | None = None,
    year: int | None = None,
    half_life_days: float = 365.0,
) -> float:
    """Compute recency bonus with exponential decay.

    Args:
        published: Publication datetime.
        year: Publication year (fallback).
        half_life_days: Days for the bonus to halve.

    Returns:
        Score in [0, 1].
    """
    now = datetime.now(UTC)
    if published:
        age_days = max((now - published).total_seconds() / 86400, 0)
    elif year:
        age_days = max((now.year - year) * 365.25, 0)
    else:
        return 0.5  # Unknown age gets neutral score

    return math.exp(-0.693 * age_days / half_life_days)


def compute_quality_score(
    candidate: CandidateRecord,
    max_h_index: int | None = None,
    weights: dict[str, float] | None = None,
    venue_data_path: str = "",
) -> QualityScore:
    """Compute composite quality score for a paper.

    Args:
        candidate: Paper candidate with metadata.
        max_h_index: Max h-index among paper's authors.
        weights: Component weights. Uses defaults if None.
        venue_data_path: Path to venue rankings data.

    Returns:
        QualityScore with full breakdown.
    """
    w = weights or DEFAULT_WEIGHTS

    cit_impact = citation_impact(candidate.citation_count)
    ven_score = venue_score(candidate.venue, data_path=venue_data_path)
    auth_cred = author_credibility(max_h_index)
    recency = _recency_score(
        published=candidate.published,
        year=candidate.year,
    )

    composite = (
        w["citation_weight"] * cit_impact
        + w["venue_weight"] * ven_score
        + w["author_weight"] * auth_cred
        + w["recency_weight"] * recency
    )
    composite = max(0.0, min(1.0, composite))

    return QualityScore(
        paper_id=candidate.arxiv_id,
        citation_impact=round(cit_impact, 4),
        venue_score=round(ven_score, 4),
        author_credibility=round(auth_cred, 4),
        composite_score=round(composite, 4),
        details={
            "recency_bonus": round(recency, 4),
            "max_h_index": max_h_index,
            "citation_count": candidate.citation_count,
            "venue": candidate.venue,
        },
    )
