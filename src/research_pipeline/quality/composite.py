"""Composite quality scoring.

Aggregates citation impact, venue reputation, author credibility,
recency, and reproducibility into a single quality score per paper.
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
    "citation_weight": 0.30,
    "venue_weight": 0.20,
    "author_weight": 0.20,
    "recency_weight": 0.15,
    "reproducibility_weight": 0.15,
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


def reproducibility_score(
    candidate: CandidateRecord,
    code_url: str | None = None,
    has_data: bool = False,
) -> float:
    """Compute reproducibility score from available signals.

    Signals:
    - Code availability (GitHub link, code_url field): +0.4
    - Data availability (explicit dataset mention): +0.3
    - Methodology detail (abstract length as proxy): +0.15
    - Community engagement (citation count as proxy): +0.15

    Args:
        candidate: Paper candidate with metadata.
        code_url: URL to the code repository, if known.
        has_data: Whether a dataset is known to be available.

    Returns:
        Score in [0, 1].
    """
    score = 0.0

    # Code availability signal
    if code_url:
        score += 0.4
    else:
        abstract_lower = candidate.abstract.lower()
        code_keywords = [
            "github.com",
            "gitlab.com",
            "our code",
            "source code",
            "open-source",
            "open source",
            "publicly available",
            "code is available",
        ]
        if any(kw in abstract_lower for kw in code_keywords):
            score += 0.3

    # Data availability signal
    if has_data:
        score += 0.3
    else:
        abstract_lower = candidate.abstract.lower()
        data_keywords = [
            "dataset",
            "benchmark",
            "our data",
            "publicly available data",
            "data is available",
        ]
        if any(kw in abstract_lower for kw in data_keywords):
            score += 0.2

    # Methodology detail proxy (longer abstracts tend to be more detailed)
    abstract_words = len(candidate.abstract.split())
    if abstract_words > 200:
        score += 0.15
    elif abstract_words > 100:
        score += 0.10

    # Community engagement proxy
    if candidate.citation_count is not None and candidate.citation_count > 10:
        score += 0.15
    elif candidate.citation_count is not None and candidate.citation_count > 0:
        score += 0.05

    return min(1.0, score)


def compute_quality_score(
    candidate: CandidateRecord,
    max_h_index: int | None = None,
    weights: dict[str, float] | None = None,
    venue_data_path: str = "",
    code_url: str | None = None,
    has_data: bool = False,
    safety_flag: str | None = None,
) -> QualityScore:
    """Compute composite quality score for a paper.

    Args:
        candidate: Paper candidate with metadata.
        max_h_index: Max h-index among paper's authors.
        weights: Component weights. Uses defaults if None.
        venue_data_path: Path to venue rankings data.
        code_url: URL to code repository for reproducibility scoring.
        has_data: Whether paper has known dataset availability.
        safety_flag: Safety flag ('retracted', 'fabricated', or None if clean).

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
    repro = reproducibility_score(candidate, code_url=code_url, has_data=has_data)

    composite = (
        w.get("citation_weight", 0.0) * cit_impact
        + w.get("venue_weight", 0.0) * ven_score
        + w.get("author_weight", 0.0) * auth_cred
        + w.get("recency_weight", 0.0) * recency
        + w.get("reproducibility_weight", 0.0) * repro
    )
    composite = max(0.0, min(1.0, composite))

    if safety_flag is not None:
        logger.warning(
            "Paper %s flagged as '%s' — composite score zeroed.",
            candidate.arxiv_id,
            safety_flag,
        )
        composite = 0.0

    return QualityScore(
        paper_id=candidate.arxiv_id,
        citation_impact=round(cit_impact, 4),
        venue_score=round(ven_score, 4),
        author_credibility=round(auth_cred, 4),
        reproducibility=round(repro, 4),
        composite_score=round(composite, 4),
        safety_flag=safety_flag,
        details={
            "recency_bonus": round(recency, 4),
            "reproducibility_score": round(repro, 4),
            "max_h_index": max_h_index,
            "citation_count": candidate.citation_count,
            "venue": candidate.venue,
            "code_url": code_url,
            "has_data": has_data,
            "safety_flag": safety_flag,
        },
    )
