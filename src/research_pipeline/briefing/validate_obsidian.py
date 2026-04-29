"""Phase C C06 validators for exported Obsidian notes.

Post-export checks confirm a note on disk:

* parses cleanly,
* carries the expected ``type`` and ``generated_id`` frontmatter,
* contains the required body headings,
* preserves Obsidian wiki-link syntax (no ``[[x]]`` → ``[x](...)``
  conversions).

These validators read the file from disk so the CLI can re-validate
existing notes without re-rendering.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from research_pipeline.briefing.obsidian_notes import (
    GENERATED_ID_KEY,
    NOTE_TYPES,
    TYPE_KEY,
    parse_frontmatter,
)
from research_pipeline.briefing.validate import ValidationResult

REQUIRED_HEADINGS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "briefing-daily": ("## Agent Read Map",),
    "briefing-topic": ("## Agent Read Map",),
    "briefing-source": ("## Agent Read Map",),
}

# Heuristic for accidental wiki-to-markdown rewrites: ``[Topic](path.md)``
# combined with the absence of any matching ``[[Topic]]`` is suspicious.
import re  # noqa: E402

_MARKDOWN_LINK_RE = re.compile(r"(?<!\!)\[([^\[\]]+)\]\(([^)]+\.md)\)")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+)(?:\|[^\[\]]+)?\]\]")


def validate_obsidian_note_file(
    path: Path,
    *,
    expected_note_type: str,
    expected_generated_id: str | None = None,
) -> ValidationResult:
    """Validate a single exported note file."""
    if expected_note_type not in NOTE_TYPES:
        raise ValueError(f"unknown expected_note_type {expected_note_type!r}")

    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, int] = {}

    if not path.exists():
        return ValidationResult(
            passed=False,
            errors=(f"missing exported note: {path}",),
        )

    text = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    if fm is None:
        return ValidationResult(
            passed=False,
            errors=(f"{path}: missing or unparsable frontmatter",),
        )

    note_type = fm.get(TYPE_KEY)
    if note_type != expected_note_type:
        errors.append(
            f"{path}: expected {TYPE_KEY}={expected_note_type!r}, got {note_type!r}"
        )

    generated_id = fm.get(GENERATED_ID_KEY)
    if not generated_id:
        errors.append(f"{path}: missing {GENERATED_ID_KEY} in frontmatter")
    elif expected_generated_id is not None and generated_id != expected_generated_id:
        errors.append(
            f"{path}: expected {GENERATED_ID_KEY}={expected_generated_id!r}, "
            f"got {generated_id!r}"
        )

    for heading in REQUIRED_HEADINGS_BY_TYPE.get(expected_note_type, ()):
        if heading not in body:
            errors.append(f"{path}: missing required heading {heading!r}")

    wikilinks = _WIKILINK_RE.findall(body)
    metrics["wikilink_count"] = len(wikilinks)

    md_links = _MARKDOWN_LINK_RE.findall(body)
    rewritten = [
        target
        for (target, _href) in md_links
        if not any(target == wl for wl in wikilinks)
    ]
    if rewritten:
        errors.append(
            f"{path}: wiki-links appear converted to markdown links: "
            f"{', '.join(sorted(set(rewritten)))}"
        )

    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
    )


def validate_obsidian_export(
    *,
    daily_path: Path | None = None,
    topic_paths: Iterable[Path] = (),
    source_paths: Iterable[Path] = (),
) -> ValidationResult:
    """Validate a full export bundle (daily + topic + source notes)."""
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, int] = {
        "daily_count": 0,
        "topic_count": 0,
        "source_count": 0,
        "wikilink_count": 0,
    }

    if daily_path is not None:
        result = validate_obsidian_note_file(
            daily_path, expected_note_type="briefing-daily"
        )
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        metrics["daily_count"] = 1
        metrics["wikilink_count"] += result.metrics.get("wikilink_count", 0)

    for topic_path in topic_paths:
        result = validate_obsidian_note_file(
            topic_path, expected_note_type="briefing-topic"
        )
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        metrics["topic_count"] += 1
        metrics["wikilink_count"] += result.metrics.get("wikilink_count", 0)

    for source_path in source_paths:
        result = validate_obsidian_note_file(
            source_path, expected_note_type="briefing-source"
        )
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        metrics["source_count"] += 1
        metrics["wikilink_count"] += result.metrics.get("wikilink_count", 0)

    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
    )
