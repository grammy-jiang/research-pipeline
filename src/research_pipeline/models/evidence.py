"""Evidence-only aggregation models.

Models for merging multiple agent/synthesis outputs while retaining
only factual assertions backed by explicit evidence citations.
Rhetoric, hedging, and unsupported confidence claims are stripped.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RhetoricType(str, Enum):
    """Classification of rhetoric patterns detected in text."""

    HEDGING = "hedging"
    CONFIDENCE_CLAIM = "confidence_claim"
    SUBJECTIVE = "subjective"
    FILLER = "filler"
    UNSUPPORTED_CAUSAL = "unsupported_causal"


class RhetoricSpan(BaseModel):
    """A detected rhetoric span in text."""

    start: int = Field(description="Start character offset.")
    end: int = Field(description="End character offset.")
    text: str = Field(description="The rhetoric text.")
    rhetoric_type: RhetoricType = Field(description="Type of rhetoric.")


class EvidencePointer(BaseModel):
    """A structured pointer to a piece of evidence."""

    paper_id: str = Field(description="Source paper arXiv ID.")
    chunk_id: str = Field(default="", description="Source chunk ID if available.")
    section: str = Field(default="", description="Section name if identified.")
    quote: str = Field(default="", description="Brief supporting quote.")
    page: int = Field(default=0, description="Page number if known.")


class EvidenceStatement(BaseModel):
    """A factual assertion backed by evidence pointers."""

    statement_id: str = Field(description="Unique ID (e.g., ES-001).")
    text: str = Field(description="The factual assertion text.")
    pointers: list[EvidencePointer] = Field(
        default_factory=list,
        description="Evidence backing this statement.",
    )
    source_type: str = Field(
        default="finding",
        description="Origin: finding, methodology, limitation, objective.",
    )
    agreement_count: int = Field(
        default=1,
        description="Number of independent sources supporting this statement.",
    )


class AggregationStats(BaseModel):
    """Statistics from an evidence aggregation run."""

    input_statements: int = Field(default=0, description="Total input statements.")
    rhetoric_stripped: int = Field(default=0, description="Rhetoric spans removed.")
    evidence_matched: int = Field(default=0, description="Statements with evidence.")
    evidence_unmatched: int = Field(
        default=0, description="Statements lacking evidence (dropped)."
    )
    merged_duplicates: int = Field(
        default=0, description="Duplicate statements merged."
    )
    output_statements: int = Field(
        default=0, description="Final output statement count."
    )
    avg_pointers_per_statement: float = Field(
        default=0.0, description="Average evidence pointers per output statement."
    )


class EvidenceAggregation(BaseModel):
    """Result of evidence-only aggregation across multiple sources."""

    topic: str = Field(description="Research topic.")
    statements: list[EvidenceStatement] = Field(
        default_factory=list,
        description="Evidence-backed factual assertions.",
    )
    dropped: list[str] = Field(
        default_factory=list,
        description="Statements dropped due to insufficient evidence.",
    )
    stats: AggregationStats = Field(
        default_factory=AggregationStats,
        description="Aggregation statistics.",
    )
