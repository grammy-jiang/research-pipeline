"""Confidence-gated retrieval depth classification.

Classifies papers into retrieval depth tiers based on screening scores,
enabling adaptive citation expansion — high-confidence papers skip
expansion, low-confidence papers trigger deeper search.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from research_pipeline.models.screening import RelevanceDecision

logger = logging.getLogger(__name__)


class DepthTier(BaseModel):
    """Retrieval depth classification for a single paper."""

    arxiv_id: str = Field(description="Paper identifier.")
    final_score: float = Field(description="Screening score that determined the tier.")
    tier: Literal["high", "medium", "low"] = Field(description="Retrieval depth tier.")
    expand_depth: int = Field(
        description="Recommended citation expansion depth "
        "(0 = skip, 1 = normal, 2+ = deep)."
    )


def classify_retrieval_depth(
    decisions: list[RelevanceDecision],
    high_threshold: float = 0.85,
    low_threshold: float = 0.50,
    high_depth: int = 0,
    medium_depth: int = 1,
    low_depth: int = 2,
) -> list[DepthTier]:
    """Classify papers into retrieval depth tiers.

    Papers with high confidence (>= high_threshold) skip citation expansion.
    Papers with low confidence (< low_threshold) trigger deeper expansion.
    This implements the confidence-gated retrieval pattern from deep research
    (FD2: >= 0.85 skip, < 0.50 deep).

    Args:
        decisions: Screening decisions with final_score.
        high_threshold: Score at or above which papers are 'high' tier.
        low_threshold: Score below which papers are 'low' tier.
        high_depth: Expansion depth for high-confidence papers.
        medium_depth: Expansion depth for medium-confidence papers.
        low_depth: Expansion depth for low-confidence papers.

    Returns:
        List of DepthTier classifications, one per decision.
    """
    tiers: list[DepthTier] = []
    counts = {"high": 0, "medium": 0, "low": 0}

    for decision in decisions:
        score = decision.final_score
        if score >= high_threshold:
            tier_name: Literal["high", "medium", "low"] = "high"
            depth = high_depth
        elif score >= low_threshold:
            tier_name = "medium"
            depth = medium_depth
        else:
            tier_name = "low"
            depth = low_depth

        counts[tier_name] += 1
        tiers.append(
            DepthTier(
                arxiv_id=decision.paper.arxiv_id,
                final_score=round(score, 4),
                tier=tier_name,
                expand_depth=depth,
            )
        )

    logger.info(
        "Depth classification: %d high (skip), %d medium (normal), %d low (deep)",
        counts["high"],
        counts["medium"],
        counts["low"],
    )
    return tiers


def papers_needing_expansion(
    tiers: list[DepthTier],
    min_depth: int = 1,
) -> list[tuple[str, int]]:
    """Filter to papers that need citation expansion.

    Args:
        tiers: Depth tier classifications.
        min_depth: Minimum expand_depth to include.

    Returns:
        List of (arxiv_id, expand_depth) for papers needing expansion.
    """
    return [(t.arxiv_id, t.expand_depth) for t in tiers if t.expand_depth >= min_depth]
