"""Phase E E06 — dossier-specific validation.

Extends ``validate.validate_dossier_report`` with explicit length budget,
required-section coverage, evidence-URL presence, factuality-label
presence, and supports the manual-only Phase E workflow.
"""

from __future__ import annotations

import re

from research_pipeline.briefing.validate import (
    REQUIRED_DOSSIER_SECTIONS,
    ValidationResult,
)

# Phase E manual dossier focuses on a single topic. Set generous link budget
# so well-cited dossiers still pass, but reject obvious link-spam runaways.
DEFAULT_MAX_LINKS = 30
DEFAULT_MAX_WORDS = 1500
DEFAULT_MIN_WORDS = 80


def validate_dossier_markdown(
    markdown: str,
    *,
    max_links: int = DEFAULT_MAX_LINKS,
    max_words: int = DEFAULT_MAX_WORDS,
    min_words: int = DEFAULT_MIN_WORDS,
) -> ValidationResult:
    """Validate dossier markdown for Phase E acceptance gates.

    Checks (errors):
      - all required sections present
      - at least one http(s) evidence URL
      - link count below ``max_links``
      - word count between ``min_words`` and ``max_words``
      - factuality label present (``factuality_label=supported_fact``)
    """
    errors: list[str] = []
    warnings: list[str] = []

    for section in REQUIRED_DOSSIER_SECTIONS:
        if section not in markdown:
            errors.append(f"missing required section: {section}")

    links = re.findall(r"https?://[^)\s]+", markdown)
    if not links:
        errors.append("dossier requires at least one evidence URL")
    if len(links) > max_links:
        errors.append(f"dossier link count {len(links)} exceeds budget {max_links}")

    words = re.findall(r"\b\w+\b", markdown)
    if len(words) > max_words:
        errors.append(f"dossier word count {len(words)} exceeds budget {max_words}")
    if len(words) < min_words:
        errors.append(f"dossier word count {len(words)} below minimum {min_words}")

    if "factuality_label=supported_fact" not in markdown:
        errors.append(
            "dossier must include factuality labels in Agent Notes "
            "(factuality_label=supported_fact)"
        )

    # Single-topic invariant: only one topic_id line allowed in frontmatter.
    topic_lines = re.findall(r"^topic_id:\s*\S+", markdown, re.MULTILINE)
    if len(topic_lines) > 1:
        errors.append("dossier must focus on a single topic")

    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics={
            "link_count": len(links),
            "word_count": len(words),
        },
    )
