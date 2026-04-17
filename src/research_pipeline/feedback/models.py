"""Pydantic models for user feedback on screened papers."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class FeedbackDecision(StrEnum):
    """User decision on a screened paper."""

    ACCEPT = "accept"
    REJECT = "reject"


class FeedbackRecord(BaseModel):
    """A single user feedback entry for a screened paper."""

    paper_id: str = Field(..., description="Paper identifier (arXiv ID or DOI).")
    run_id: str = Field(..., description="Run ID where the paper was screened.")
    decision: FeedbackDecision = Field(..., description="User accept/reject decision.")
    reason: str = Field(default="", description="Optional reason for the decision.")
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp when the feedback was recorded.",
    )
    cheap_score: float = Field(
        default=0.0,
        description="The cheap score the paper had when screened.",
    )


class WeightAdjustment(BaseModel):
    """Result of ELO-style weight adjustment from accumulated feedback."""

    bm25_must_title: float = Field(default=0.20)
    bm25_nice_title: float = Field(default=0.10)
    bm25_must_abstract: float = Field(default=0.25)
    bm25_nice_abstract: float = Field(default=0.10)
    cat_match: float = Field(default=0.15)
    negative_penalty: float = Field(default=0.10)
    recency_bonus: float = Field(default=0.10)
    feedback_count: int = Field(
        default=0, description="Number of feedback records used."
    )
    learning_rate: float = Field(default=0.05, description="ELO-style K-factor used.")

    def to_weight_dict(self) -> dict[str, float]:
        """Convert to the weight dict expected by score_candidates."""
        return {
            "bm25_must_title": self.bm25_must_title,
            "bm25_nice_title": self.bm25_nice_title,
            "bm25_must_abstract": self.bm25_must_abstract,
            "bm25_nice_abstract": self.bm25_nice_abstract,
            "cat_match": self.cat_match,
            "negative_penalty": self.negative_penalty,
            "recency_bonus": self.recency_bonus,
        }
