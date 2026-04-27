"""Context engineering for token budget management and paper compaction.

Implements the Tokalator pattern: context is a finite, expensive resource
with O(T²) cost growth without management.

Paper content compaction follows the OpenDev 5-stage Adaptive Context
Compaction (ACC) pattern, adapted for academic papers:
1. Full paper — under budget, use as-is
2. Section extraction — keep key sections, drop references/appendices
3. Aggressive pruning — abstract + methodology + results only
4. Abstract-only — last resort for extremely long papers
5. Skip — budget exhausted

Each compaction level is recorded in the ExecutionRecord for provenance
(non-identifiability: we must record WHY content was compacted).
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

logger = logging.getLogger(__name__)

# Approximate tokens-per-character ratio for English text
CHARS_PER_TOKEN = 4


class CompactionLevel(StrEnum):
    """Paper content compaction levels (OpenDev ACC, adapted)."""

    FULL = "full"
    SECTIONS = "sections"
    AGGRESSIVE = "aggressive"
    ABSTRACT_ONLY = "abstract_only"
    SKIPPED = "skipped"


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length.

    Uses the standard ~4 chars/token heuristic for English text.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def extract_sections(
    markdown: str,
    sections: tuple[str, ...] = (
        "abstract",
        "introduction",
        "method",
        "methodology",
        "approach",
        "experiment",
        "result",
        "discussion",
        "conclusion",
    ),
) -> str:
    """Extract key sections from a markdown paper.

    Keeps sections whose headings match the given keywords.
    Drops references, appendices, acknowledgments, and other sections.
    """
    lines = markdown.split("\n")
    output_lines: list[str] = []
    in_section = False
    current_heading_level = 0

    for line in lines:
        heading_match = re.match(r"^(#{1,3})\s+(.+)", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip().lower()

            # Check if this heading matches a desired section
            matches_desired = any(s in title for s in sections)
            matches_skip = any(
                s in title
                for s in ("reference", "bibliography", "appendix", "acknowledgment")
            )

            if matches_desired:
                in_section = True
                current_heading_level = level
                output_lines.append(line)
            elif matches_skip or (
                level <= current_heading_level and not matches_desired
            ):
                in_section = False
            elif in_section and level > current_heading_level:
                # Sub-heading within a desired section
                output_lines.append(line)
            else:
                in_section = False
        elif in_section:
            output_lines.append(line)

    result = "\n".join(output_lines).strip()
    return result if result else markdown[:5000]


def extract_aggressive(markdown: str) -> str:
    """Aggressive pruning: abstract + methodology + results only."""
    return extract_sections(
        markdown,
        sections=("abstract", "method", "methodology", "approach", "result"),
    )


def extract_abstract(markdown: str) -> str:
    """Extract abstract only as last resort."""
    result = extract_sections(markdown, sections=("abstract",))
    if len(result) < 100:
        # Fallback: take first 2000 characters
        return markdown[:2000]
    return result


def compact_paper(
    markdown: str,
    max_tokens: int,
) -> tuple[str, CompactionLevel]:
    """Apply 5-stage adaptive content compaction to a paper.

    Returns the compacted text and the compaction level applied.
    Provenance: caller should record the level in the ExecutionRecord.
    """
    # Level 1: Full paper
    tokens = estimate_tokens(markdown)
    if tokens <= max_tokens:
        return markdown, CompactionLevel.FULL

    # Level 2: Section extraction
    sectioned = extract_sections(markdown)
    if estimate_tokens(sectioned) <= max_tokens:
        logger.info(
            "Compacted paper from %d to %d tokens (sections)",
            tokens,
            estimate_tokens(sectioned),
        )
        return sectioned, CompactionLevel.SECTIONS

    # Level 3: Aggressive pruning
    aggressive = extract_aggressive(markdown)
    if estimate_tokens(aggressive) <= max_tokens:
        logger.info(
            "Compacted paper from %d to %d tokens (aggressive)",
            tokens,
            estimate_tokens(aggressive),
        )
        return aggressive, CompactionLevel.AGGRESSIVE

    # Level 4: Abstract only
    abstract = extract_abstract(markdown)
    if estimate_tokens(abstract) <= max_tokens:
        logger.info(
            "Compacted paper from %d to %d tokens (abstract only)",
            tokens,
            estimate_tokens(abstract),
        )
        return abstract, CompactionLevel.ABSTRACT_ONLY

    # Level 5: Truncate to budget
    truncated = abstract[: max_tokens * CHARS_PER_TOKEN]
    logger.warning(
        "Paper exceeds budget even with abstract — truncated to %d tokens",
        estimate_tokens(truncated),
    )
    return truncated, CompactionLevel.ABSTRACT_ONLY
