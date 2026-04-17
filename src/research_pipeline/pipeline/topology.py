"""Adaptive pipeline topology: stage selection based on research profile."""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class PipelineProfile(StrEnum):
    """Pipeline execution profiles."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


# Stage definitions per profile
PROFILE_STAGES: dict[PipelineProfile, list[str]] = {
    PipelineProfile.QUICK: ["plan", "search", "screen", "summarize"],
    PipelineProfile.STANDARD: [
        "plan",
        "search",
        "screen",
        "download",
        "convert",
        "extract",
        "summarize",
    ],
    PipelineProfile.DEEP: [
        "plan",
        "search",
        "screen",
        "download",
        "convert",
        "extract",
        "summarize",
        "expand",
        "quality",
        "analyze_claims",
        "score_claims",
    ],
}

# Stages that can be skipped in quick mode
SKIPPABLE_STAGES = frozenset(
    {
        "download",
        "convert",
        "extract",
        "expand",
        "quality",
        "analyze_claims",
        "score_claims",
    }
)


def get_stages(profile: PipelineProfile) -> list[str]:
    """Get the ordered list of stages for a profile.

    Args:
        profile: Pipeline execution profile.

    Returns:
        Ordered list of stage names.
    """
    return list(PROFILE_STAGES[profile])


def should_run_stage(profile: PipelineProfile, stage: str) -> bool:
    """Check if a stage should run for the given profile.

    Args:
        profile: Pipeline execution profile.
        stage: Stage name to check.

    Returns:
        True if stage should be executed.
    """
    return stage in PROFILE_STAGES[profile]


def classify_query_complexity(topic: str) -> PipelineProfile:
    """Auto-classify query complexity to suggest a pipeline profile.

    Uses heuristics:
    - Short queries (≤3 words, no special terms) → QUICK
    - Medium queries (standard academic topics) → STANDARD
    - Complex queries (multiple concepts, comparison, "comprehensive") → DEEP

    Args:
        topic: The research topic string.

    Returns:
        Suggested pipeline profile.
    """
    words = topic.lower().split()
    word_count = len(words)

    # Indicators of deep research need
    deep_indicators = [
        "comprehensive",
        "systematic review",
        "survey",
        "comparison",
        "versus",
        "vs.",
        "trade-off",
        "state of the art",
        "state-of-the-art",
        "taxonomy",
        "landscape",
        "evolution",
        "history",
    ]
    topic_lower = topic.lower()
    has_deep_indicator = any(ind in topic_lower for ind in deep_indicators)

    # Indicators of quick lookup
    quick_indicators = [
        "what is",
        "define",
        "overview",
        "introduction",
        "basics",
    ]
    has_quick_indicator = any(ind in topic_lower for ind in quick_indicators)

    if has_quick_indicator or word_count <= 3:
        return PipelineProfile.QUICK

    if has_deep_indicator or word_count >= 10:
        return PipelineProfile.DEEP

    return PipelineProfile.STANDARD


def profile_summary(profile: PipelineProfile) -> str:
    """Get a human-readable summary of what a profile does.

    Args:
        profile: Pipeline profile.

    Returns:
        Description string.
    """
    summaries = {
        PipelineProfile.QUICK: (
            "Quick overview: searches and screens papers, produces synthesis "
            "from abstracts only (no PDF download/conversion)."
        ),
        PipelineProfile.STANDARD: (
            "Standard pipeline: full 7-stage analysis with PDF download, "
            "conversion to markdown, extraction, and evidence-based "
            "summarization."
        ),
        PipelineProfile.DEEP: (
            "Deep research: standard pipeline plus citation graph expansion, "
            "quality scoring, claim decomposition, and confidence scoring."
        ),
    }
    return summaries[profile]
