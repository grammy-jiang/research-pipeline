"""Pydantic schemas for LLM I/O validation."""

from pydantic import BaseModel, Field


class RelevanceJudgmentInput(BaseModel):
    """Input schema for the LLM relevance judge."""

    topic: str
    must_terms: list[str]
    paper_title: str
    paper_abstract: str
    paper_categories: list[str]


class RelevanceJudgmentOutput(BaseModel):
    """Output schema for the LLM relevance judge."""

    llm_score: float = Field(ge=0.0, le=1.0)
    label: str
    rationale: list[str]
    evidence_quotes: list[dict[str, str]]
    uncertainties: list[str]
    needs_fulltext_validation: list[str]


class SummarizationInput(BaseModel):
    """Input schema for LLM summarization."""

    topic: str
    paper_title: str
    chunks: list[dict[str, str]]


class SummarizationOutput(BaseModel):
    """Output schema for LLM summarization."""

    objective: str
    methodology: str
    findings: list[str]
    limitations: list[str]
    evidence: list[dict[str, str]]
    uncertainties: list[str]


class PaperExtractionSkimOutput(BaseModel):
    """Output schema for the Step 1 skim pass."""

    candidate_claims: list[str] = Field(default_factory=list)
    candidate_methods: list[str] = Field(default_factory=list)
    candidate_results: list[str] = Field(default_factory=list)
    candidate_assumptions: list[str] = Field(default_factory=list)
    candidate_limitations: list[str] = Field(default_factory=list)
    candidate_reusable_mechanisms: list[str] = Field(default_factory=list)


class PaperExtractionStatementOutput(BaseModel):
    """One statement in a Step 1 extraction response."""

    statement: str
    statement_type: str
    confidence: str
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class PaperExtractionRecordOutput(BaseModel):
    """Output schema for full Step 1 paper extraction."""

    context: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    problem: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    contributions: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    methods: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    datasets: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    evaluation: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    results: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    operational_characteristics: list[PaperExtractionStatementOutput] = Field(
        default_factory=list
    )
    scale_assumptions: list[PaperExtractionStatementOutput] = Field(
        default_factory=list
    )
    hardware_requirements: list[PaperExtractionStatementOutput] = Field(
        default_factory=list
    )
    software_tools: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    cost_drivers: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    security_privacy: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    reliability: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    observability_needs: list[PaperExtractionStatementOutput] = Field(
        default_factory=list
    )
    assumptions: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    limitations: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    future_work: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    reusable_mechanisms: list[PaperExtractionStatementOutput] = Field(
        default_factory=list
    )
    generality: list[PaperExtractionStatementOutput] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class PaperEvidenceAnchorOutput(BaseModel):
    """Output schema for an evidence anchoring pass."""

    statement_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: str = "MEDIUM"


class PaperExtractionQualityReviewOutput(BaseModel):
    """Output schema for an LLM-assisted Step 1 quality review."""

    warnings: list[str] = Field(default_factory=list)
    unsupported_statement_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    """Output schema for LLM cross-paper synthesis."""

    agreements: list[dict[str, object]]
    disagreements: list[dict[str, object]]
    open_questions: list[str]


class CrossPaperSynthesisOutput(BaseModel):
    """Output schema for Step 2 structured synthesis sub-passes."""

    findings: list[dict[str, object]] = Field(default_factory=list)
    assumptions: list[dict[str, object]] = Field(default_factory=list)
    contradictions: list[dict[str, object]] = Field(default_factory=list)
    risks: list[dict[str, object]] = Field(default_factory=list)
    traceability: list[dict[str, object]] = Field(default_factory=list)


class SynthesisNeutralityReviewOutput(BaseModel):
    """Output schema for Step 2 neutrality review."""

    neutral: bool = True
    prescriptive_phrases: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
