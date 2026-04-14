"""Quality score model for multi-dimensional paper evaluation.

Stores per-paper quality metrics computed from external APIs (Semantic Scholar,
CORE rankings, etc.) during the Tier 1 abstract-level filtering stage.
"""

from pydantic import BaseModel, Field


class QualityScore(BaseModel):
    """Multi-dimensional quality assessment for a single paper.

    All component scores are normalized to [0, 1].  The ``composite_score``
    is a weighted blend of the components.  The ``details`` dict preserves
    raw metrics for transparency so the agent (or human) can inspect why a
    paper received a given score.
    """

    paper_id: str = Field(description="Paper identifier (arxiv_id, DOI, or S2 ID).")
    citation_impact: float = Field(
        description="Normalized citation impact score (0–1)."
    )
    venue_tier: str | None = Field(
        default=None,
        description="CORE venue tier: A*, A, B, C, or None if unknown.",
    )
    venue_score: float = Field(description="Normalized venue reputation score (0–1).")
    author_credibility: float = Field(
        description="Normalized author credibility score (0–1, based on h-index)."
    )
    reproducibility: float = Field(
        default=0.0,
        description="Reproducibility signal (0–1, reserved for future use).",
    )
    composite_score: float = Field(
        description="Weighted composite quality score (0–1)."
    )
    safety_flag: str | None = Field(
        default=None,
        description="Safety flag: 'retracted', 'fabricated', or None if clean.",
    )
    details: dict[str, object] = Field(
        default_factory=dict,
        description="Raw metrics for transparency (e.g. citation_count, h_index).",
    )
