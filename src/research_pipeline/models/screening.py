"""Screening models for relevance scoring and shortlist decisions."""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)


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


_VALID_DOWNLOAD_REASONS: frozenset[str] = frozenset(
    {"score_threshold", "topn_fill", "manual_override"}
)


def _default_cheap() -> dict[str, Any]:
    """Return a zeroed-out CheapScoreBreakdown dict."""
    return {
        "bm25_title": 0.0,
        "bm25_abstract": 0.0,
        "cat_match": 0.0,
        "negative_penalty": 0.0,
        "recency_bonus": 0.0,
        "cheap_score": 0.0,
    }


def parse_shortlist_lenient(data: dict[str, Any]) -> RelevanceDecision:
    """Parse a shortlist entry leniently, filling defaults for missing fields.

    This allows manually-edited or sub-agent-curated shortlists to omit
    boilerplate fields like ``cheap`` (full breakdown) or ``download_reason``
    (enum literal) without causing ``ValidationError``.

    Args:
        data: Raw dict from shortlist.json.

    Returns:
        A validated ``RelevanceDecision`` instance.
    """
    patched = dict(data)

    # cheap: accept missing, float, or full dict
    cheap_val = patched.get("cheap")
    if cheap_val is None:
        patched["cheap"] = _default_cheap()
    elif isinstance(cheap_val, int | float):
        defaults = _default_cheap()
        defaults["cheap_score"] = float(cheap_val)
        patched["cheap"] = defaults

    # download_reason: default to manual_override if missing or invalid
    reason = patched.get("download_reason")
    if reason is None or reason not in _VALID_DOWNLOAD_REASONS:
        if reason is not None:
            logger.debug(
                "Coercing unrecognized download_reason %r to 'manual_override'",
                reason,
            )
        patched["download_reason"] = "manual_override"

    # final_score: default to 0.0 if missing
    if "final_score" not in patched:
        patched["final_score"] = 0.0

    # download: default to True if missing
    if "download" not in patched:
        patched["download"] = True

    return RelevanceDecision.model_validate(patched)
