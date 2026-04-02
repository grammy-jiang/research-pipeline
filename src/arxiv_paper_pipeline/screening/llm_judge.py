"""LLM-based relevance judge for second-pass screening.

This module provides schema-constrained LLM judgment for candidate relevance.
When LLM is disabled, it returns None (heuristic-only mode).
"""

import logging

from arxiv_paper_pipeline.models.candidate import CandidateRecord
from arxiv_paper_pipeline.models.screening import LLMJudgment

logger = logging.getLogger(__name__)


def judge_candidate(
    candidate: CandidateRecord,
    topic: str,
    must_terms: list[str],
    llm_call_fn: object | None = None,
) -> LLMJudgment | None:
    """Judge a single candidate's relevance using an LLM.

    Args:
        candidate: The candidate paper to judge.
        topic: The research topic.
        must_terms: Required terms for context.
        llm_call_fn: Callable that performs the LLM call. If None, returns None.

    Returns:
        LLMJudgment if LLM is available, None otherwise.
    """
    if llm_call_fn is None:
        logger.debug("LLM disabled, skipping judgment for %s", candidate.arxiv_id)
        return None

    # LLM integration point — to be implemented when LLM boundary is connected
    logger.warning("LLM judge called but not yet integrated for %s", candidate.arxiv_id)
    return None
