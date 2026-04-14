"""Claim decomposition and evidence taxonomy models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceClass(str, Enum):
    """Evidence support classification for a claim."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONFLICTING = "conflicting"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


class ClaimEvidence(BaseModel):
    """Evidence for a single atomic claim."""

    chunk_id: str = Field(description="Source chunk ID.")
    relevance_score: float = Field(description="BM25/retrieval score (0-1 normalized).")
    quote: str = Field(default="", description="Brief supporting quote from chunk.")


class AtomicClaim(BaseModel):
    """An atomic, self-contained claim decomposed from a paper summary."""

    claim_id: str = Field(description="Unique claim identifier (e.g. CL-001).")
    paper_id: str = Field(description="arXiv ID of the source paper.")
    source_type: str = Field(
        description="Origin: 'finding', 'limitation', 'methodology', 'objective'."
    )
    statement: str = Field(description="The atomic claim text.")
    evidence_class: EvidenceClass = Field(default=EvidenceClass.UNSUPPORTED)
    evidence: list[ClaimEvidence] = Field(default_factory=list)
    confidence_score: float = Field(
        default=0.0, description="Confidence in classification (0-1)."
    )


class ClaimDecomposition(BaseModel):
    """Complete claim decomposition result for a paper."""

    paper_id: str = Field(description="arXiv ID.")
    title: str = Field(description="Paper title.")
    claims: list[AtomicClaim] = Field(default_factory=list)
    total_claims: int = Field(default=0)
    evidence_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count per evidence class (e.g. {'supported': 5, 'partial': 2}).",
    )
