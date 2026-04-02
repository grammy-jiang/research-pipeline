"""Summary models for per-paper summaries and cross-paper synthesis."""

from pydantic import BaseModel, Field


class SummaryEvidence(BaseModel):
    """Evidence reference backing a summary claim."""

    chunk_id: str = Field(description="Source chunk ID.")
    line_range: str = Field(default="", description="Line range in source.")
    quote: str = Field(default="", description="Brief supporting quote.")


class PaperSummary(BaseModel):
    """Evidence-driven summary of a single paper."""

    arxiv_id: str = Field(description="Base arXiv ID.")
    version: str = Field(description="Version string.")
    title: str = Field(description="Paper title.")
    objective: str = Field(description="Main objective of the paper.")
    methodology: str = Field(description="Key methodology or approach.")
    findings: list[str] = Field(
        default_factory=list,
        description="Key findings.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Stated or identified limitations.",
    )
    evidence: list[SummaryEvidence] = Field(
        default_factory=list,
        description="Evidence references for claims.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Items marked uncertain (no evidence found).",
    )


class SynthesisAgreement(BaseModel):
    """A point of agreement across multiple papers."""

    claim: str = Field(description="The agreed-upon claim.")
    supporting_papers: list[str] = Field(description="arXiv IDs that agree.")
    evidence: list[SummaryEvidence] = Field(default_factory=list)


class SynthesisDisagreement(BaseModel):
    """A point of disagreement across papers."""

    topic: str = Field(description="The contested topic.")
    positions: dict[str, str] = Field(description="Mapping of arXiv ID to position.")
    evidence: list[SummaryEvidence] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    """Cross-paper synthesis report."""

    topic: str = Field(description="Research topic.")
    paper_count: int = Field(description="Number of papers synthesized.")
    agreements: list[SynthesisAgreement] = Field(default_factory=list)
    disagreements: list[SynthesisDisagreement] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions identified across papers.",
    )
    paper_summaries: list[PaperSummary] = Field(
        default_factory=list,
        description="Individual paper summaries included in synthesis.",
    )
