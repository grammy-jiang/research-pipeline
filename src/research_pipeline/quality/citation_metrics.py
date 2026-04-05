"""Citation-based quality metrics.

Normalizes raw citation counts to a [0, 1] impact score using
log-scale normalization.  Computes citation velocity (citations per
year since publication).
"""

import logging
import math
from datetime import UTC, datetime

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)


def citation_impact(citation_count: int | None, scale: int = 1000) -> float:
    """Compute log-normalized citation impact score.

    Args:
        citation_count: Raw citation count (None treated as 0).
        scale: Reference citation count for normalization.

    Returns:
        Score in [0, 1].
    """
    count = citation_count or 0
    return min(1.0, math.log(1 + count) / math.log(1 + scale))


def citation_velocity(
    citation_count: int | None,
    published: datetime | None = None,
    year: int | None = None,
) -> float:
    """Compute citation velocity: citations per year since publication.

    Args:
        citation_count: Raw citation count.
        published: Publication datetime.
        year: Publication year (fallback if published is None).

    Returns:
        Citations per year (0.0 if publication date unknown).
    """
    count = citation_count or 0
    if count == 0:
        return 0.0

    now = datetime.now(UTC)
    if published:
        age_years = max((now - published).total_seconds() / (365.25 * 86400), 0.25)
    elif year:
        age_years = max(now.year - year + 0.5, 0.25)
    else:
        return 0.0

    return count / age_years


def compute_citation_metrics(
    candidate: CandidateRecord,
) -> dict[str, float]:
    """Compute all citation metrics for a candidate.

    Args:
        candidate: Paper candidate with optional citation fields.

    Returns:
        Dict with 'citation_impact', 'citation_velocity',
        'influential_ratio' keys.
    """
    impact = citation_impact(candidate.citation_count)
    velocity = citation_velocity(
        candidate.citation_count,
        published=candidate.published,
        year=candidate.year,
    )

    influential = candidate.influential_citation_count or 0
    total = candidate.citation_count or 0
    influential_ratio = influential / total if total > 0 else 0.0

    return {
        "citation_impact": round(impact, 4),
        "citation_velocity": round(velocity, 2),
        "influential_ratio": round(influential_ratio, 4),
    }
