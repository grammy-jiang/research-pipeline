"""Summary models for per-paper extraction and cross-paper synthesis."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceLevel(StrEnum):
    """Confidence label used for extracted and synthesized statements."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class StatementType(StrEnum):
    """Source/interpretation type for an extracted statement."""

    AUTHOR_CLAIM = "author_claim"
    EMPIRICAL_RESULT = "empirical_result"
    INTERPRETATION = "interpretation"
    MODEL_INFERENCE = "model_inference"


class EvidenceSnippet(BaseModel):
    """Evidence span from a source paper."""

    evidence_id: str = Field(description="Stable evidence identifier.")
    paper_id: str = Field(description="Source paper ID.")
    chunk_id: str = Field(default="", description="Source chunk ID.")
    line_range: str = Field(default="", description="Line range in source.")
    section: str = Field(default="", description="Source section path.")
    page: int | None = Field(default=None, description="Page number if available.")
    quote: str = Field(default="", description="Brief supporting quote.")
    supports: list[str] = Field(
        default_factory=list,
        description="Statement IDs supported by this evidence.",
    )
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)


class ExtractedStatement(BaseModel):
    """A typed statement extracted from one paper."""

    statement_id: str = Field(description="Stable statement identifier.")
    statement: str = Field(description="Atomic statement text.")
    category: str = Field(
        description="Extraction category, e.g. contribution, method, result."
    )
    statement_type: StatementType = Field(default=StatementType.AUTHOR_CLAIM)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = Field(default="", description="Caveats or normalization notes.")


class ExtractionMetadata(BaseModel):
    """Metadata about how a paper extraction was produced."""

    mode: str = Field(default="structured", description="structured or fallback mode.")
    model: str = Field(default="", description="LLM model name if available.")
    prompt_version: str = Field(default="paper_extraction_v1")
    generated_at: str = Field(default="", description="ISO timestamp if recorded.")


class ExtractionQuality(BaseModel):
    """Deterministic quality checks for a per-paper extraction."""

    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    specificity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_statement_count: int = Field(default=0, ge=0)
    missing_critical_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PaperExtractionRecord(BaseModel):
    """Schema-first, evidence-backed extraction for one paper."""

    paper_id: str = Field(description="Base arXiv/source paper ID.")
    version: str = Field(default="", description="Version string if available.")
    title: str = Field(description="Paper title.")
    authors: list[str] = Field(default_factory=list)
    venue: str = Field(default="")
    year: str = Field(default="")
    doi: str = Field(default="")
    url: str = Field(default="")
    context: list[ExtractedStatement] = Field(default_factory=list)
    problem: list[ExtractedStatement] = Field(default_factory=list)
    contributions: list[ExtractedStatement] = Field(default_factory=list)
    methods: list[ExtractedStatement] = Field(default_factory=list)
    datasets: list[ExtractedStatement] = Field(default_factory=list)
    evaluation: list[ExtractedStatement] = Field(default_factory=list)
    results: list[ExtractedStatement] = Field(default_factory=list)
    operational_characteristics: list[ExtractedStatement] = Field(default_factory=list)
    scale_assumptions: list[ExtractedStatement] = Field(default_factory=list)
    hardware_requirements: list[ExtractedStatement] = Field(default_factory=list)
    software_tools: list[ExtractedStatement] = Field(default_factory=list)
    cost_drivers: list[ExtractedStatement] = Field(default_factory=list)
    security_privacy: list[ExtractedStatement] = Field(default_factory=list)
    reliability: list[ExtractedStatement] = Field(default_factory=list)
    observability_needs: list[ExtractedStatement] = Field(default_factory=list)
    assumptions: list[ExtractedStatement] = Field(default_factory=list)
    limitations: list[ExtractedStatement] = Field(default_factory=list)
    future_work: list[ExtractedStatement] = Field(default_factory=list)
    reusable_mechanisms: list[ExtractedStatement] = Field(default_factory=list)
    generality: list[ExtractedStatement] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    extraction_metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    quality: ExtractionQuality = Field(default_factory=ExtractionQuality)


class SynthesisFinding(BaseModel):
    """Evidence-backed finding produced by cross-paper synthesis."""

    finding_id: str = Field(description="Stable finding identifier.")
    finding: str = Field(description="Synthesized finding.")
    finding_type: str = Field(description="agreement, pattern, implication, etc.")
    supporting_papers: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.LOW)
    limitations: list[str] = Field(default_factory=list)
    interpretation_notes: str = Field(default="")


class ContradictionRecord(BaseModel):
    """Contradiction or unresolved tension across papers."""

    contradiction_id: str = Field(description="Stable contradiction identifier.")
    topic: str = Field(description="Contested topic.")
    positions: dict[str, str] = Field(default_factory=dict)
    source_papers: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    severity: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    resolution_status: str = Field(default="unresolved")


class AssumptionRecord(BaseModel):
    """Assumption consolidated across Step 1 records."""

    assumption_id: str = Field(description="Stable assumption identifier.")
    assumption: str = Field(description="Assumption text.")
    source_papers: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    scope: str = Field(default="")
    risk_if_false: str = Field(default="")


class ReusableMechanism(BaseModel):
    """Reusable method, component, or pattern found across papers."""

    mechanism_id: str = Field(description="Stable mechanism identifier.")
    name: str = Field(description="Short mechanism name.")
    description: str = Field(description="Mechanism description.")
    source_papers: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    generality: str = Field(default="")
    known_constraints: list[str] = Field(default_factory=list)


class SynthesisQuality(BaseModel):
    """Quality checks for a cross-paper synthesis."""

    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    traceability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    neutrality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class CrossPaperSynthesisRecord(BaseModel):
    """Design-neutral cross-paper synthesis built from Step 1 extractions."""

    topic: str = Field(description="Research topic.")
    corpus: list[dict[str, str]] = Field(default_factory=list)
    methodology: list[str] = Field(default_factory=list)
    taxonomy: list[SynthesisFinding] = Field(default_factory=list)
    evidence_matrix: list[dict[str, str]] = Field(default_factory=list)
    recurring_patterns: list[SynthesisFinding] = Field(default_factory=list)
    assumption_map: list[AssumptionRecord] = Field(default_factory=list)
    contradiction_map: list[ContradictionRecord] = Field(default_factory=list)
    evidence_strength_map: list[SynthesisFinding] = Field(default_factory=list)
    operational_implications: list[SynthesisFinding] = Field(default_factory=list)
    production_readiness: list[SynthesisFinding] = Field(default_factory=list)
    reusable_mechanism_inventory: list[ReusableMechanism] = Field(default_factory=list)
    design_implications: list[SynthesisFinding] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    risk_register: list[SynthesisFinding] = Field(default_factory=list)
    traceability_appendix: list[dict[str, str]] = Field(default_factory=list)
    quality: SynthesisQuality = Field(default_factory=SynthesisQuality)


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
