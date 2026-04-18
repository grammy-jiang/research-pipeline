"""LLM-based relevance judge for second-pass screening.

This module provides schema-constrained LLM judgment for candidate relevance.
When LLM is disabled, it returns None (heuristic-only mode).
"""

import logging

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.screening import LLMJudgment

logger = logging.getLogger(__name__)

_VALID_LABELS: frozenset[str] = frozenset({"high", "medium", "low", "off_topic"})

_PROMPT_TEMPLATE = """\
You are an academic paper relevance judge. Evaluate whether the following paper \
is relevant to the research topic.

## Research Topic
{topic}

## Required Terms
{must_terms}

## Paper Metadata
- **Title**: {title}
- **Abstract**: {abstract}
- **Categories**: {categories}

## Instructions
Evaluate the paper's relevance to the research topic and required terms. \
Return your judgment as JSON with the following fields:

- "llm_score": a float between 0.0 and 1.0 (0 = completely irrelevant, \
1 = perfectly relevant)
- "label": one of "high", "medium", "low", or "off_topic"
- "rationale": a list of reasoning steps (strings) explaining your decision
- "evidence_quotes": a list of objects with "text" and "source" keys, where \
"source" is one of "title", "abstract", "category", "date"
- "uncertainties": a list of strings describing aspects you are uncertain about
- "needs_fulltext_validation": a list of strings describing claims that need \
full-text validation

Respond ONLY with valid JSON.
"""


def _build_prompt(
    candidate: CandidateRecord,
    topic: str,
    must_terms: list[str],
) -> str:
    """Build the relevance judgment prompt for a candidate.

    Args:
        candidate: The candidate paper to judge.
        topic: The research topic.
        must_terms: Required terms for context.

    Returns:
        Formatted prompt string.
    """
    return _PROMPT_TEMPLATE.format(
        topic=topic,
        must_terms=", ".join(must_terms) if must_terms else "(none)",
        title=candidate.title,
        abstract=candidate.abstract,
        categories=", ".join(candidate.categories),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the given range."""
    return max(lo, min(hi, value))


def _parse_response(raw: dict[str, object]) -> LLMJudgment | None:
    """Parse an LLM response dict into an LLMJudgment.

    Applies clamping and label validation before constructing the model.

    Args:
        raw: Raw response dictionary from the LLM provider.

    Returns:
        Parsed LLMJudgment, or None if parsing fails.
    """
    try:
        score = float(raw.get("llm_score", 0.0))  # type: ignore[arg-type]
        score = _clamp(score, 0.0, 1.0)

        label = str(raw.get("label", "medium"))
        if label not in _VALID_LABELS:
            logger.warning("Invalid label %r, defaulting to 'medium'", label)
            label = "medium"

        rationale = raw.get("rationale", [])
        if not isinstance(rationale, list):
            rationale = []

        evidence_quotes = raw.get("evidence_quotes", [])
        if not isinstance(evidence_quotes, list):
            evidence_quotes = []

        uncertainties = raw.get("uncertainties", [])
        if not isinstance(uncertainties, list):
            uncertainties = []

        needs_fulltext = raw.get("needs_fulltext_validation", [])
        if not isinstance(needs_fulltext, list):
            needs_fulltext = []

        return LLMJudgment(
            llm_score=score,
            label=label,  # type: ignore[arg-type]
            rationale=[str(r) for r in rationale],
            evidence_quotes=evidence_quotes,
            uncertainties=[str(u) for u in uncertainties],
            needs_fulltext_validation=[str(n) for n in needs_fulltext],
        )
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("Failed to parse LLM response: %s", exc)
        return None


def judge_candidate(
    candidate: CandidateRecord,
    topic: str,
    must_terms: list[str],
    llm_provider: LLMProvider | None = None,
) -> LLMJudgment | None:
    """Judge a single candidate's relevance using an LLM.

    Args:
        candidate: The candidate paper to judge.
        topic: The research topic.
        must_terms: Required terms for context.
        llm_provider: LLM provider instance. If None, returns None.

    Returns:
        LLMJudgment if LLM is available and response parses, None otherwise.
    """
    if llm_provider is None:
        logger.debug("LLM disabled, skipping judgment for %s", candidate.arxiv_id)
        return None

    prompt = _build_prompt(candidate, topic, must_terms)

    try:
        raw = llm_provider.call(
            prompt=prompt,
            schema_id="relevance_judgment",
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning(
            "LLM call failed for %s: %s",
            candidate.arxiv_id,
            exc,
        )
        return None

    if not isinstance(raw, dict):
        logger.warning(
            "LLM returned non-dict response for %s: %s",
            candidate.arxiv_id,
            type(raw).__name__,
        )
        return None

    judgment = _parse_response(raw)
    if judgment is None:
        logger.warning(
            "Failed to parse LLM judgment for %s",
            candidate.arxiv_id,
        )
    return judgment


def judge_batch(
    candidates: list[CandidateRecord],
    topic: str,
    must_terms: list[str],
    llm_provider: LLMProvider | None = None,
) -> list[LLMJudgment | None]:
    """Judge a batch of candidates using an LLM.

    Args:
        candidates: List of candidate papers to judge.
        topic: The research topic.
        must_terms: Required terms for context.
        llm_provider: LLM provider instance. If None, returns all Nones.

    Returns:
        List of LLMJudgment or None, one per candidate.
    """
    return [
        judge_candidate(candidate, topic, must_terms, llm_provider)
        for candidate in candidates
    ]
