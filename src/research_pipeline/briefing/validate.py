"""Deterministic validators for briefing artifacts."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from research_pipeline.briefing.models import BriefingCluster
from research_pipeline.briefing.normalize import normalize_title

REQUIRED_DAILY_SECTIONS = (
    "## 🔥 Executive Signal",
    "## ⭐ Top Items",
    "## 🗒️ Feedback Targets",
)

REQUIRED_DOSSIER_SECTIONS = (
    "## Agent Read Map",
    "## One-paragraph Summary",
    "## What Changed",
    "## Why It Matters Technically",
    "## Evidence Timeline",
    "## Artifacts To Open",
    "## Open Questions",
    "## Agent Notes",
)


class ValidationResult(BaseModel):
    """Briefing validation result."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, int] = Field(default_factory=dict)


def validate_daily_report(
    markdown: str,
    clusters: list[BriefingCluster],
    *,
    max_links: int = 30,
    active_min_words: int = 400,
    active_max_words: int = 1400,
) -> ValidationResult:
    """Validate daily report shape, budgets, duplicates, and evidence links."""
    errors: list[str] = []
    warnings: list[str] = []
    for section in REQUIRED_DAILY_SECTIONS:
        if section not in markdown:
            errors.append(f"missing required section: {section}")
    links = re.findall(r"https?://[^)\s]+", markdown)
    words = re.findall(r"\b\w+\b", markdown)
    item_count = len(clusters)
    if len(links) > max_links:
        errors.append(f"link count {len(links)} exceeds budget {max_links}")
    if item_count >= 6 and not (active_min_words <= len(words) <= active_max_words):
        errors.append(
            f"active-day word count {len(words)} outside "
            f"{active_min_words}-{active_max_words}"
        )
    title_counts = Counter(normalize_title(cluster.title) for cluster in clusters)
    duplicates = [title for title, count in title_counts.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate cluster titles: {', '.join(duplicates)}")
    for cluster in clusters:
        if not cluster.canonical_urls:
            errors.append(f"cluster {cluster.cluster_id} has no evidence URL")
        elif str(cluster.canonical_urls[0]) not in markdown:
            errors.append(
                f"cluster {cluster.cluster_id} evidence URL missing from report"
            )
        if cluster.evidence_type == "supported_fact" and "[FACT]" not in markdown:
            errors.append("supported facts must be labeled [FACT]")
        if cluster.evidence_type == "inference" and "[INFERENCE]" not in markdown:
            errors.append("inferences must be labeled [INFERENCE]")
        if (
            cluster.evidence_type == "speculation_or_watch_item"
            and "[WATCH]" not in markdown
        ):
            errors.append("watch/speculation items must be labeled [WATCH]")
    if clusters and "Duplicate release mention" in markdown:
        errors.append("report used duplicate-source boilerplate as primary summary")
    if clusters and "Previous release." in markdown:
        errors.append("report included low-information release filler")
    if item_count == 0:
        warnings.append("no material updates found")
    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics={
            "word_count": len(words),
            "link_count": len(links),
            "item_count": item_count,
        },
    )


def validate_obsidian_path(path: Path, vault_root: Path) -> None:
    """Ensure generated notes stay under the configured vault root."""
    root = vault_root.resolve()
    target = path.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"obsidian path escapes vault root: {path}")


def validate_dossier_report(markdown: str, *, max_links: int = 30) -> ValidationResult:
    """Validate hot-topic dossier shape and evidence links."""
    errors: list[str] = []
    for section in REQUIRED_DOSSIER_SECTIONS:
        if section not in markdown:
            errors.append(f"missing required section: {section}")
    links = re.findall(r"https?://[^)\s]+", markdown)
    if not links:
        errors.append("dossier requires at least one evidence URL")
    if len(links) > max_links:
        errors.append(f"dossier link count {len(links)} exceeds budget {max_links}")
    if "factuality_label=supported_fact" not in markdown:
        errors.append("dossier must include factuality labels in Agent Notes")
    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        metrics={
            "link_count": len(links),
            "word_count": len(re.findall(r"\b\w+\b", markdown)),
        },
    )


def validate_obsidian_note(markdown: str, required_type: str) -> ValidationResult:
    """Validate generated Obsidian note frontmatter and Agent Read Map."""
    errors: list[str] = []
    if not markdown.startswith("---\n"):
        errors.append("missing YAML frontmatter")
    if f"type: {required_type}" not in markdown:
        errors.append(f"missing frontmatter type: {required_type}")
    if "## Agent Read Map" not in markdown:
        errors.append("missing Agent Read Map")
    return ValidationResult(passed=not errors, errors=tuple(errors))


def validation_to_json(result: ValidationResult) -> dict[str, Any]:
    """Serialize validation result."""
    return result.model_dump(mode="json")
