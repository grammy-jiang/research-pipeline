"""Topic Obsidian note export (Phase C ticket C04).

Topic notes live at ``<vault>/<subdir>/Topics/<topic_id>.md``. Each note
encodes the durable ``TopicMemory`` snapshot in its frontmatter so external
tooling can join on ``topic_id`` without re-parsing the body. Wiki-links in
the body are passed through verbatim — link generation belongs to ticket
C05 (``obsidian_links``).

Ownership rules mirror :mod:`research_pipeline.briefing.obsidian_daily`:
existing notes without a matching ``generated_id`` are refused.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from research_pipeline.briefing.obsidian import (
    ObsidianConfig,
    is_owned_generated_note,
    validate_vault_path,
)
from research_pipeline.briefing.obsidian_notes import (
    ObsidianNote,
    compose_note,
)

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def topic_note_path(config: ObsidianConfig, topic_id: str) -> Path:
    """Return the canonical path for a topic note in the configured vault."""
    if not _SLUG_RE.match(topic_id):
        raise ValueError(f"invalid topic_id (must be slug-like): {topic_id!r}")
    return config.namespace_root / "Topics" / f"{topic_id}.md"


def build_topic_note(
    *,
    topic_id: str,
    name: str,
    body: str,
    status: str | None = None,
    fatigue_score: float | None = None,
    last_reported_at: str | None = None,
    aliases: tuple[str, ...] | list[str] = (),
    extra: dict[str, Any] | None = None,
) -> ObsidianNote:
    """Build a topic :class:`ObsidianNote` from durable topic memory fields."""
    if not name.strip():
        raise ValueError("topic name is required")
    fm: dict[str, Any] = {"topic_id": topic_id, "name": name}
    if status:
        fm["status"] = status
    if fatigue_score is not None:
        fm["fatigue_score"] = float(fatigue_score)
    if last_reported_at:
        fm["last_reported_at"] = last_reported_at
    if aliases:
        fm["aliases"] = list(aliases)
    if extra:
        for key, value in extra.items():
            if key in fm:
                continue
            fm[key] = value
    return ObsidianNote(
        note_type="briefing-topic",
        generated_id=f"topic-{topic_id}",
        extra=fm,
        body=body,
    )


def export_topic_note(note: ObsidianNote, config: ObsidianConfig) -> Path | None:
    """Write ``note`` to the topics directory, honouring ownership/dry-run."""
    if note.note_type != "briefing-topic":
        raise ValueError(
            f"export_topic_note expects briefing-topic, got {note.note_type!r}"
        )
    topic_id = str(note.extra.get("topic_id") or "")
    if not topic_id:
        raise ValueError("topic note frontmatter is missing topic_id")
    target = topic_note_path(config, topic_id)
    resolved = validate_vault_path(target, config)
    if resolved.exists() and not is_owned_generated_note(resolved, note.generated_id):
        raise ValueError(
            "refusing to overwrite Obsidian topic note without matching "
            f"generated_id at {resolved}"
        )
    text = compose_note(note)
    if config.dry_run:
        return None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved
