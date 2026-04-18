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


class SynthesisOutput(BaseModel):
    """Output schema for LLM cross-paper synthesis."""

    agreements: list[dict[str, object]]
    disagreements: list[dict[str, object]]
    open_questions: list[str]
