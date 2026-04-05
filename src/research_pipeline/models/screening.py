"""Screening models for relevance scoring and shortlist decisions."""

from typing import Literal

from pydantic import BaseModel, Field

from research_pipeline.models.candidate import CandidateRecord


class CheapScoreBreakdown(BaseModel):
    """Heuristic scoring breakdown for a single candidate."""

    bm25_title: float = Field(description="BM25 score on title.")
    bm25_abstract: float = Field(description="BM25 score on abstract.")
    cat_match: float = Field(description="Category match bonus.")
    negative_penalty: float = Field(description="Penalty for negative term matches.")
    recency_bonus: float = Field(description="Bonus for recent papers.")
    semantic_score: float | None = Field(
        default=None,
        description="SPECTER2 semantic similarity score (0-1). None when disabled.",
    )
    cheap_score: float = Field(description="Aggregated heuristic score.")


class EvidenceQuote(BaseModel):
    """An evidence quote extracted by the LLM judge."""

    text: str = Field(description="The quoted text.")
    source: Literal["title", "abstract", "category", "date"] = Field(
        description="Which field the quote came from."
    )


class LLMJudgment(BaseModel):
    """LLM-based relevance judgment for a single candidate."""

    llm_score: float = Field(description="LLM relevance score (0-1).")
    label: Literal["high", "medium", "low", "off_topic"] = Field(
        description="Relevance label."
    )
    rationale: list[str] = Field(
        default_factory=list,
        description="Reasoning steps for the decision.",
    )
    evidence_quotes: list[EvidenceQuote] = Field(
        default_factory=list,
        description="Supporting evidence from the paper metadata.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Aspects the LLM is uncertain about.",
    )
    needs_fulltext_validation: list[str] = Field(
        default_factory=list,
        description="Claims that need full-text validation.",
    )


class RelevanceDecision(BaseModel):
    """Final relevance decision combining heuristic and LLM scores."""

    paper: CandidateRecord
    cheap: CheapScoreBreakdown
    llm: LLMJudgment | None = None
    final_score: float = Field(description="Combined final relevance score.")
    download: bool = Field(description="Whether to download this paper.")
    download_reason: Literal["score_threshold", "topn_fill", "manual_override"] = Field(
        description="Why this paper was selected for download."
    )
