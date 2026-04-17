"""Monitoring and doom-loop detection for iterative workflows.

Implements two harness engineering patterns:
1. Doom-loop detection (OpenDev): MD5 fingerprint + Jaccard similarity
   to detect when iterative synthesis is producing repetitive output.
2. Iteration drift monitoring (Agent Drift): tracking per-iteration
   metrics to detect search space exhaustion.

Principle: "Detect repetition before it wastes resources."
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from enum import StrEnum

logger = logging.getLogger(__name__)


class StopReason(StrEnum):
    """Explicit stop conditions for iterative workflows.

    Each is logged with reason in the ExecutionRecord (provenance).
    """

    IMPLEMENTATION_READY = "implementation_ready"
    MAX_ITERATIONS = "max_iterations"
    DOOM_LOOP_EXACT = "doom_loop_exact_match"
    DOOM_LOOP_SIMILAR = "doom_loop_high_similarity"
    NO_NEW_GAPS = "no_new_academic_gaps"
    USER_DECLINED = "user_declined"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_NEW_CANDIDATES = "no_new_unique_candidates"


def content_fingerprint(content: str) -> str:
    """Compute MD5 fingerprint for doom-loop detection."""
    normalized = content.strip().lower()
    return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()


def _extract_terms(text: str) -> Counter:
    """Extract normalized word terms from text for Jaccard comparison."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return Counter(words)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts using term sets.

    Lightweight (no LLM needed) — uses term overlap only.
    Returns a value in [0.0, 1.0].
    """
    if not text_a or not text_b:
        return 0.0

    terms_a = set(_extract_terms(text_a))
    terms_b = set(_extract_terms(text_b))

    if not terms_a and not terms_b:
        return 1.0
    if not terms_a or not terms_b:
        return 0.0

    intersection = terms_a & terms_b
    union = terms_a | terms_b
    return len(intersection) / len(union)


def check_doom_loop(
    previous_content: str | None,
    new_content: str,
    similarity_threshold: float = 0.85,
) -> tuple[bool, float, str]:
    """Check whether new content indicates a doom loop.

    Returns (is_loop, similarity_score, reason).

    Two-level detection:
    1. Exact MD5 match → definite doom loop
    2. Jaccard similarity > threshold → likely doom loop
    """
    if previous_content is None:
        return False, 0.0, "no_previous_content"

    # Level 1: Exact fingerprint match
    if content_fingerprint(previous_content) == content_fingerprint(new_content):
        return True, 1.0, StopReason.DOOM_LOOP_EXACT

    # Level 2: Jaccard similarity
    similarity = jaccard_similarity(previous_content, new_content)
    if similarity >= similarity_threshold:
        return True, similarity, StopReason.DOOM_LOOP_SIMILAR

    return False, similarity, "below_threshold"


class IterationMetrics:
    """Per-iteration metrics for drift monitoring.

    Tracks papers found, analyzed, and gaps across iterations to detect
    search space exhaustion (Agent Drift pattern).
    """

    def __init__(self) -> None:
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        """All recorded iteration metrics."""
        return list(self._history)

    def record(
        self,
        iteration: int,
        papers_found: int,
        papers_analyzed: int,
        gaps_remaining: int,
        new_unique_papers: int = 0,
    ) -> None:
        """Record metrics for an iteration."""
        self._history.append(
            {
                "iteration": iteration,
                "papers_found": papers_found,
                "papers_analyzed": papers_analyzed,
                "gaps_remaining": gaps_remaining,
                "new_unique_papers": new_unique_papers,
            }
        )

    def is_search_exhausted(self) -> bool:
        """Detect if the search space is likely exhausted.

        True when papers_found is monotonically decreasing over 2+ iterations
        AND gaps_remaining is stable (not decreasing).
        """
        if len(self._history) < 2:
            return False

        recent = self._history[-2:]
        papers_decreasing = recent[1]["papers_found"] <= recent[0]["papers_found"]
        gaps_stable = recent[1]["gaps_remaining"] >= recent[0]["gaps_remaining"]

        return papers_decreasing and gaps_stable

    def should_stop(self) -> StopReason | None:
        """Check all metric-based stop conditions.

        Returns a StopReason if stopping is recommended, None otherwise.
        """
        if not self._history:
            return None

        latest = self._history[-1]

        if latest.get("new_unique_papers", 0) == 0 and len(self._history) > 1:
            return StopReason.NO_NEW_CANDIDATES

        return None

    def summary(self) -> str:
        """Human-readable summary of iteration progression."""
        if not self._history:
            return "No iterations recorded."

        lines = []
        for m in self._history:
            lines.append(
                f"  Iter {m['iteration']}: "
                f"{m['papers_found']} found, "
                f"{m['papers_analyzed']} analyzed, "
                f"{m['gaps_remaining']} gaps, "
                f"{m.get('new_unique_papers', '?')} new"
            )
        return "Iteration history:\n" + "\n".join(lines)
