"""Pydantic models for bidirectional citation snowball expansion.

Supports budget-aware stopping, round-by-round tracking, and
marginal-relevance decay detection.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class StopReason(str, Enum):
    """Why snowball expansion terminated."""

    MAX_ROUNDS = "max_rounds"
    MAX_PAPERS = "max_papers"
    RELEVANCE_DECAY = "relevance_decay"
    DIVERSITY_SATURATION = "diversity_saturation"
    NO_NEW_PAPERS = "no_new_papers"
    USER_ABORT = "user_abort"


class SnowballBudget(BaseModel):
    """Budget constraints for snowball expansion.

    Args:
        max_rounds: Maximum expansion rounds (default 5).
        max_total_papers: Hard cap on total discovered papers.
        relevance_decay_threshold: Fraction of new papers that must
            score above the shortlist median (0.0-1.0). When below
            this for ``decay_patience`` consecutive rounds, stop.
        decay_patience: Consecutive low-relevance rounds before stopping.
        diversity_window: Stop if unique-category count hasn't grown
            for this many consecutive rounds.
        limit_per_paper: Max papers fetched per seed per direction.
        direction: "citations", "references", or "both".
        reference_boost: Multiplier for backward (reference) limit
            when direction is "both".
    """

    max_rounds: int = Field(default=5, ge=1, le=20)
    max_total_papers: int = Field(default=200, ge=1, le=5000)
    relevance_decay_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    decay_patience: int = Field(default=2, ge=1, le=10)
    diversity_window: int = Field(default=3, ge=1, le=10)
    limit_per_paper: int = Field(default=20, ge=1, le=500)
    direction: str = Field(default="both")
    reference_boost: float = Field(default=1.5, ge=0.1, le=10.0)


class SnowballRound(BaseModel):
    """Statistics for a single snowball expansion round.

    Args:
        round_number: 1-based round index.
        seeds_count: Papers used as seeds this round.
        fetched_count: Raw papers fetched from API.
        new_count: Papers not seen before (after dedup).
        relevant_count: New papers scoring above median.
        relevance_fraction: relevant_count / new_count.
        unique_categories: Cumulative unique categories so far.
        new_categories: Categories first seen this round.
    """

    round_number: int = Field(ge=1)
    seeds_count: int = Field(ge=0)
    fetched_count: int = Field(ge=0)
    new_count: int = Field(ge=0)
    relevant_count: int = Field(ge=0)
    relevance_fraction: float = Field(ge=0.0, le=1.0)
    unique_categories: int = Field(ge=0)
    new_categories: int = Field(ge=0)


class SnowballResult(BaseModel):
    """Complete result of a snowball expansion run.

    Args:
        seed_ids: Original seed paper IDs.
        query_terms: Terms used for relevance scoring.
        budget: Budget configuration used.
        rounds: Per-round statistics.
        total_discovered: Total unique papers found.
        stop_reason: Why expansion terminated.
        api_calls: Total API calls made.
    """

    seed_ids: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    budget: SnowballBudget = Field(default_factory=SnowballBudget)
    rounds: list[SnowballRound] = Field(default_factory=list)
    total_discovered: int = Field(default=0, ge=0)
    stop_reason: StopReason = Field(default=StopReason.MAX_ROUNDS)
    api_calls: int = Field(default=0, ge=0)
