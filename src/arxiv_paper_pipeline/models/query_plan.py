"""Query plan models for topic normalization and arXiv query generation."""

from pydantic import BaseModel, Field


class SparsityThresholds(BaseModel):
    """Thresholds for detecting sparse result sets and triggering fallback."""

    min_candidates: int = Field(
        default=40,
        description="Minimum candidates before triggering fallback window.",
    )
    min_highscore: int = Field(
        default=10,
        description="Minimum high-score candidates before fallback.",
    )
    min_downloads: int = Field(
        default=5,
        description="Minimum downloadable candidates before fallback.",
    )


class QueryPlan(BaseModel):
    """Structured query plan derived from a user topic."""

    topic_raw: str = Field(description="Original user-provided topic string.")
    topic_normalized: str = Field(
        description="Normalized topic after expansion/cleanup."
    )
    must_terms: list[str] = Field(
        default_factory=list,
        description="Terms that MUST appear in results.",
    )
    nice_terms: list[str] = Field(
        default_factory=list,
        description="Terms that SHOULD appear (boost, not filter).",
    )
    negative_terms: list[str] = Field(
        default_factory=list,
        description="Terms to exclude via ANDNOT.",
    )
    candidate_categories: list[str] = Field(
        default_factory=list,
        description='arXiv category codes, e.g. ["cs.IR", "cs.CL"].',
    )
    query_variants: list[str] = Field(
        default_factory=list,
        description="3-6 arXiv query strings to execute.",
    )
    primary_months: int = Field(
        default=6,
        description="Primary search window in months.",
    )
    fallback_months: int = Field(
        default=12,
        description="Extended window if primary is too sparse.",
    )
    sparsity_thresholds: SparsityThresholds = Field(
        default_factory=SparsityThresholds,
    )
