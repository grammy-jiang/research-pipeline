"""Length normalization for LLM responses.

Prevents verbosity bias by enforcing configurable token budgets on
agent responses.  Supports soft budgets (log a warning) and hard
budgets (truncate to limit), plus per-task budget profiles.

References:
    Deep-research report Theme 7 (Output Quality Control) — verbosity
    bias as a dominant failure mode in long-form synthesis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")

# ── Token estimation (shared with memory/segmentation.py) ────────────

DEFAULT_WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    """Estimate token count using the ≈0.75 words-per-token heuristic."""
    word_count = len(_WORD_RE.findall(text))
    return int(word_count / DEFAULT_WORDS_PER_TOKEN) + 1


# ── Budget enforcement mode ──────────────────────────────────────────


class BudgetMode(str, Enum):
    """How to handle responses that exceed the token budget."""

    SOFT = "soft"
    HARD = "hard"


# ── Budget profiles ──────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenBudget:
    """Token budget specification for a task type.

    Attributes:
        max_tokens: Maximum estimated tokens allowed.
        mode: Soft (warn only) or hard (truncate).
        task_type: Optional task identifier for logging.
    """

    max_tokens: int
    mode: BudgetMode = BudgetMode.SOFT
    task_type: str = "default"


# Common profiles matching pipeline stages
BUDGET_PROFILES: dict[str, TokenBudget] = {
    "screening": TokenBudget(
        max_tokens=300, mode=BudgetMode.SOFT, task_type="screening"
    ),
    "extraction": TokenBudget(
        max_tokens=600, mode=BudgetMode.SOFT, task_type="extraction"
    ),
    "summary": TokenBudget(max_tokens=1200, mode=BudgetMode.SOFT, task_type="summary"),
    "synthesis": TokenBudget(
        max_tokens=2400, mode=BudgetMode.SOFT, task_type="synthesis"
    ),
    "analysis": TokenBudget(
        max_tokens=1800, mode=BudgetMode.SOFT, task_type="analysis"
    ),
    "plan": TokenBudget(max_tokens=500, mode=BudgetMode.SOFT, task_type="plan"),
}


# ── Normalization result ─────────────────────────────────────────────


@dataclass
class NormalizationResult:
    """Result of applying length normalization.

    Attributes:
        text: The (possibly truncated) output text.
        original_tokens: Estimated tokens in the original text.
        final_tokens: Estimated tokens after normalization.
        truncated: Whether the text was truncated.
        budget: The budget that was applied.
    """

    text: str
    original_tokens: int
    final_tokens: int
    truncated: bool
    budget: TokenBudget


# ── Core normalization function ──────────────────────────────────────


def normalize_length(
    text: str,
    budget: TokenBudget | None = None,
    max_tokens: int | None = None,
    mode: BudgetMode = BudgetMode.SOFT,
) -> NormalizationResult:
    """Apply length normalization to an LLM response.

    Args:
        text: The LLM response text.
        budget: A ``TokenBudget`` object.  If provided, *max_tokens*
            and *mode* are ignored.
        max_tokens: Maximum token budget (used when *budget* is None).
            Defaults to 1200.
        mode: Budget mode (used when *budget* is None).

    Returns:
        A ``NormalizationResult`` with the normalized text and metadata.
    """
    if budget is None:
        budget = TokenBudget(
            max_tokens=max_tokens or 1200,
            mode=mode,
            task_type="custom",
        )

    original_tokens = estimate_tokens(text)

    if original_tokens <= budget.max_tokens:
        return NormalizationResult(
            text=text,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            truncated=False,
            budget=budget,
        )

    # Over budget
    if budget.mode == BudgetMode.SOFT:
        logger.warning(
            "Response exceeds %s budget: %d tokens > %d max (task=%s)",
            budget.mode.value,
            original_tokens,
            budget.max_tokens,
            budget.task_type,
        )
        return NormalizationResult(
            text=text,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            truncated=False,
            budget=budget,
        )

    # Hard mode: truncate at sentence boundary
    truncated_text = _truncate_at_sentence(text, budget.max_tokens)
    final_tokens = estimate_tokens(truncated_text)

    logger.info(
        "Truncated response from %d to %d tokens (budget=%d, task=%s)",
        original_tokens,
        final_tokens,
        budget.max_tokens,
        budget.task_type,
    )

    return NormalizationResult(
        text=truncated_text,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        truncated=True,
        budget=budget,
    )


def _truncate_at_sentence(text: str, max_tokens: int) -> str:
    """Truncate text at the nearest sentence boundary within budget."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text

    result_parts: list[str] = []
    running_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if running_tokens + sent_tokens > max_tokens and result_parts:
            break
        result_parts.append(sent)
        running_tokens += sent_tokens

    if not result_parts:
        # Even the first sentence is too long — do word-level truncation
        return _truncate_at_word(text, max_tokens)

    return " ".join(result_parts)


def _truncate_at_word(text: str, max_tokens: int) -> str:
    """Word-level truncation as a fallback."""
    words = text.split()
    # Each word ≈ 1/0.75 ≈ 1.33 tokens; so max_words ≈ max_tokens * 0.75
    max_words = int(max_tokens * DEFAULT_WORDS_PER_TOKEN)
    if max_words < 1:
        max_words = 1
    return " ".join(words[:max_words])


# ── Batch normalization ──────────────────────────────────────────────


def normalize_batch(
    texts: list[str],
    budget: TokenBudget | None = None,
    max_tokens: int | None = None,
    mode: BudgetMode = BudgetMode.SOFT,
) -> list[NormalizationResult]:
    """Apply length normalization to a batch of texts.

    Args:
        texts: List of LLM response texts.
        budget: Shared budget for all texts.
        max_tokens: Shared max token limit.
        mode: Shared budget mode.

    Returns:
        List of ``NormalizationResult`` objects.
    """
    return [
        normalize_length(text, budget=budget, max_tokens=max_tokens, mode=mode)
        for text in texts
    ]


# ── Statistics helper ────────────────────────────────────────────────


@dataclass
class BatchStats:
    """Aggregate statistics from a batch of normalization results."""

    total_texts: int
    truncated_count: int
    avg_original_tokens: float
    avg_final_tokens: float
    max_original_tokens: int
    over_budget_ratio: float
    details: list[dict[str, Any]] = field(default_factory=list)


def compute_batch_stats(results: list[NormalizationResult]) -> BatchStats:
    """Compute aggregate statistics for a batch of normalization results."""
    if not results:
        return BatchStats(
            total_texts=0,
            truncated_count=0,
            avg_original_tokens=0.0,
            avg_final_tokens=0.0,
            max_original_tokens=0,
            over_budget_ratio=0.0,
        )

    total = len(results)
    truncated = sum(1 for r in results if r.truncated)
    over_budget = sum(1 for r in results if r.original_tokens > r.budget.max_tokens)
    originals = [r.original_tokens for r in results]
    finals = [r.final_tokens for r in results]

    return BatchStats(
        total_texts=total,
        truncated_count=truncated,
        avg_original_tokens=sum(originals) / total,
        avg_final_tokens=sum(finals) / total,
        max_original_tokens=max(originals),
        over_budget_ratio=over_budget / total,
        details=[
            {
                "original_tokens": r.original_tokens,
                "final_tokens": r.final_tokens,
                "truncated": r.truncated,
                "task_type": r.budget.task_type,
            }
            for r in results
        ],
    )
