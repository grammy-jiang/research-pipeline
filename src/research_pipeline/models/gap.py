"""Gap and evidence-map models for structured research output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GapRecord(BaseModel):
    """A research gap identified during synthesis.

    Gaps are classified as ACADEMIC (requiring more papers) or
    ENGINEERING (fillable with implementation knowledge).
    """

    gap_id: str = Field(description="Unique gap identifier (e.g. GAP-001).")
    gap_type: Literal["ACADEMIC", "ENGINEERING"] = Field(
        description="Whether the gap requires more papers or engineering judgment."
    )
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Impact severity on the research goal."
    )
    description: str = Field(description="What is missing or incomplete.")
    impact: str = Field(
        default="", description="Why this gap matters for the research goal."
    )
    related_papers: list[str] = Field(
        default_factory=list,
        description="arXiv IDs of papers that touch on but don't resolve this gap.",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Suggested search queries to fill this gap (ACADEMIC gaps only).",
    )
    resolution: str = Field(
        default="",
        description="Suggested resolution approach (ENGINEERING gaps only).",
    )
    resolved: bool = Field(
        default=False, description="Whether this gap has been resolved."
    )
    resolved_in_iteration: int | None = Field(
        default=None,
        description="Iteration number in which this gap was resolved.",
    )


class EvidenceMapEntry(BaseModel):
    """A single cell in the evidence map: whether a paper covers an aspect."""

    paper_id: str = Field(description="arXiv ID or source-prefixed ID.")
    aspect: str = Field(description="Research question aspect being assessed.")
    covered: bool = Field(
        default=False, description="Whether the paper addresses this aspect."
    )
    section_ref: str = Field(
        default="",
        description="Section reference if covered (e.g. '§3', '§4.2').",
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="MEDIUM",
        description="Confidence in the coverage assessment.",
    )


class EvidenceMap(BaseModel):
    """Research-question-aspect × paper coverage matrix.

    Maps which papers provide evidence for which aspects of the
    research question, enabling gap identification and audit.
    """

    research_question: str = Field(description="The research question being mapped.")
    aspects: list[str] = Field(
        description="Research question aspects (rows of the matrix)."
    )
    papers: list[str] = Field(description="Paper IDs (columns of the matrix).")
    entries: list[EvidenceMapEntry] = Field(
        default_factory=list,
        description="Coverage entries (the matrix cells).",
    )

    def coverage_for_aspect(self, aspect: str) -> list[EvidenceMapEntry]:
        """Return all entries for a given aspect.

        Args:
            aspect: The research question aspect to filter by.

        Returns:
            List of entries matching the aspect.
        """
        return [e for e in self.entries if e.aspect == aspect]

    def coverage_for_paper(self, paper_id: str) -> list[EvidenceMapEntry]:
        """Return all entries for a given paper.

        Args:
            paper_id: The paper ID to filter by.

        Returns:
            List of entries matching the paper.
        """
        return [e for e in self.entries if e.paper_id == paper_id]

    def uncovered_aspects(self) -> list[str]:
        """Return aspects with no covering paper.

        Returns:
            List of aspects not covered by any paper.
        """
        covered_aspects = {e.aspect for e in self.entries if e.covered}
        return [a for a in self.aspects if a not in covered_aspects]
